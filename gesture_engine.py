import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import yaml
import threading
from collections import Counter
from typing import Tuple, List, Optional, Dict

class GestureEngine:
    """
    Handles MediaPipe Hands in LIVE_STREAM mode with temporal gesture smoothing.
    """
    
    def __init__(self, config_path: str = "config.yaml", model_path: str = "hand_landmarker.task"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.lock = threading.Lock()
        self.last_result = None
        self.start_time = time.time()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=self.config['mediapipe']['max_num_hands'],
            min_hand_detection_confidence=self.config['mediapipe']['min_detection_confidence'],
            min_hand_presence_confidence=self.config['mediapipe']['min_tracking_confidence'],
            min_tracking_confidence=self.config['mediapipe']['min_tracking_confidence'],
            result_callback=self._on_result
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.hand_data = {} # {hand_idx: {'prev_xh': ..., 'holds': ..., 'history': []}}
        
    def _on_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        with self.lock:
            self.last_result = result

    def send_frame(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        self.detector.detect_async(mp_image, timestamp_ms)

    def get_latest_results(self) -> List[Dict]:
        with self.lock:
            result = self.last_result
            if not result or not result.hand_landmarks:
                return []
            
        hand_results = []
        for i, landmarks in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            hand_id = f"{label}_{i}"
            
            if hand_id not in self.hand_data:
                self.hand_data[hand_id] = {
                    'prev_xh': None, 
                    'holds': {}, 
                    'cooldowns': {},
                    'history': []
                }
            
            raw_gesture = self._classify_gesture(landmarks, label, self.hand_data[hand_id])
            
            # Temporal Smoothing (Majority Vote)
            history = self.hand_data[hand_id]['history']
            history.append(raw_gesture)
            if len(history) > 5: # 5-frame window
                history.pop(0)
            
            smooth_gesture = Counter(history).most_common(1)[0][0]
            
            hand_results.append({
                'gesture': smooth_gesture,
                'landmarks': landmarks,
                'label': label,
                'id': hand_id
            })
            
        return hand_results

    def _is_finger_up(self, lms, finger_idx: int) -> bool:
        tip_idx = finger_idx * 4 + 8
        pip_idx = finger_idx * 4 + 6
        return lms[tip_idx].y < lms[pip_idx].y

    def _is_thumb_up(self, lms, hand_label: str) -> bool:
        tip = lms[4]
        ip = lms[3]
        if hand_label == "Right":
            return tip.x < ip.x
        else:
            return tip.x > ip.x

    def _classify_gesture(self, lms, hand_label: str, state: dict) -> str:
        f_up = [self._is_finger_up(lms, i) for i in range(4)]
        thumb_up = self._is_thumb_up(lms, hand_label)
        
        # 🖐️ CLEAR / HOVER_PALM
        if (sum(f_up) + (1 if thumb_up else 0)) >= 4:
            return "CLEAR" if self._check_hold(state, "CLEAR", 1.0) else "HOVER_PALM"

        # ✊ PAUSE
        if not any(f_up) and not thumb_up:
            return "PAUSE"
            
        # ☝️ DRAW
        if f_up[0] and not any(f_up[1:]):
            return "DRAW"

        # ✌️ COLOR / UNDO / REDO
        if f_up[0] and f_up[1] and not f_up[2] and not f_up[3]:
            dx = lms[8].x - (state['prev_xh'] if state['prev_xh'] is not None else lms[8].x)
            state['prev_xh'] = lms[8].x
            
            threshold = 0.05 
            if dx < -threshold: return "UNDO"
            if dx > threshold: return "REDO"
            
            dist_tips = np.sqrt((lms[8].x - lms[12].x)**2 + (lms[8].y - lms[12].y)**2)
            if dist_tips < self.config['gestures']['snap_dist_threshold']:
                return "SHAPE_SNAP"
                
            return "COLOR_PICK"

        # 🤏 BRUSH_SIZE gesture (detection only, main.py will ignore logic)
        dist_pinch = np.sqrt((lms[4].x - lms[8].x)**2 + (lms[4].y - lms[8].y)**2)
        if dist_pinch < 0.05 and not f_up[1] and not f_up[2] and not f_up[3]:
            return "BRUSH_SIZE"

        # 🤙 SWITCH_LAYER
        if thumb_up and f_up[3] and not f_up[0] and not f_up[1] and not f_up[2]:
            return "SWITCH_LAYER"

        # 👌 EXPORT
        if dist_pinch < 0.04 and f_up[1] and f_up[2] and f_up[3]:
             return "EXPORT" if self._check_hold(state, "EXPORT", 2.0) else "HOVER_OK"

        return "UNKNOWN"

    def _check_hold(self, state, gesture_name: str, duration: float) -> bool:
        now = time.time()
        holds = state['holds']
        cooldowns = state['cooldowns']
        
        if gesture_name not in holds:
            holds[gesture_name] = now
            return False
            
        if now - holds[gesture_name] > duration:
            if now - cooldowns.get(gesture_name, 0) > 1.5:
                cooldowns[gesture_name] = now
                return True
        return False

    def reset_holds(self, hand_id=None):
        if hand_id and hand_id in self.hand_data:
            self.hand_data[hand_id]['holds'] = {}
            self.hand_data[hand_id]['prev_xh'] = None
        else:
            for hid in self.hand_data:
                self.hand_data[hid]['holds'] = {}
                self.hand_data[hid]['prev_xh'] = None
