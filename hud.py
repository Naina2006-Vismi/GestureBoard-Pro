import cv2
import numpy as np
import yaml
import time
from typing import Dict, Any, List

class HUDRenderer:
    """
    Renders the GestureBoard Pro HUD, including the new interactive color arc and help table.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.hud_cfg = self.config['hud']
        self.palette_cfg = self.hud_cfg['palette']
        
        # Gesture definitions for help table
        self.gestures_help = [
            ("DRAW", "☝️", "Index Finger Up"),
            ("COLOR", "✌️", "Index + Middle Up"),
            ("SIZE", "🤏", "Pinch Thumb + Index"),
            ("PAUSE", "✊", "Closed Fist"),
            ("CLEAR", "🖐️", "Open Palm (Hold 1s)"),
            ("UNDO", "⬅️", "Swipe Left (✌️)"),
            ("REDO", "➡️", "Swipe Right (✌️)"),
            ("SNAP", "🫰", "Cross Fingers"),
            ("LAYER", "🤙", "Thumb + Pinky"),
            ("EXPORT", "👌", "OK Sign (Hold 2s)")
        ]
        
    def draw_hud(self, frame: np.ndarray, stats: Dict[str, Any], state: Dict[str, Any], hand_results: List[Dict], show_help: bool = False):
        """
        Draw all HUD components.
        """
        h, w = frame.shape[:2]
        tb_h = self.hud_cfg['top_bar_height']
        
        # 1. Top Bar Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, tb_h), self.hud_cfg['colors']['background'], -1)
        cv2.addWeighted(overlay, self.hud_cfg['bg_opacity'], frame, 1 - self.hud_cfg['bg_opacity'], 0, frame)
        
        # 2. Performance Stats
        fps_text = f"FPS: {stats['fps']:.1f}"
        lat_text = f"LATENCY: {stats['latency']:.1f}ms"
        perf_color = self.hud_cfg['colors']['good_perf']
        if stats['fps'] < 25: perf_color = self.hud_cfg['colors']['warn_perf']
        
        cv2.putText(frame, fps_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, perf_color, 1, cv2.LINE_AA)
        cv2.putText(frame, lat_text, (120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "GestureBoard Pro", (w // 2 - 80, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'H' for Help", (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # 3. Dynamic Color Arc
        is_picking = any(hr['gesture'] == "COLOR_PICK" for hr in hand_results)
        active_point = None
        for hr in hand_results:
            if hr['gesture'] == "COLOR_PICK":
                lms = hr['landmarks']
                active_point = (int(lms[8].x * w), int(lms[8].y * h))
                break
                
        if is_picking:
            self._draw_color_arc(frame, active_point)

        # 4. Multi-Hand Status Badges
        for i, hr in enumerate(hand_results):
            gesture = hr['gesture']
            label = hr['label']
            y_off = h - 30 - (i * 40)
            
            pulse = int(abs(np.sin(time.time() * 5)) * 5)
            dot_color = (0, 255, 255) if label == "Right" else (255, 0, 255)
            cv2.circle(frame, (30, y_off), 8 + pulse, dot_color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{label}: {gesture}", (55, y_off + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Brush Preview
        cv2.circle(frame, (w // 2, h - 40), state['brush_size'], state['brush_color'], -1, cv2.LINE_AA)
        cv2.putText(frame, f"{state['brush_size']}px", (w // 2 - 20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 6. Help Table Sidebar
        if show_help:
            self._draw_help_table(frame)

    def _draw_help_table(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        table_w = 260
        table_h = 320
        x_start = w - table_w - 10
        y_start = 50
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + table_w, y_start + table_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "GESTURE GUIDE", (x_start + 60, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        for i, (action, emoji, desc) in enumerate(self.gestures_help):
            y_pos = y_start + 65 + (i * 25)
            # We can't draw actual emojis in OpenCV text without complex setup, 
            # so we'll just use the action names and simple text markers.
            cv2.putText(frame, f"{action}", (x_start + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{desc}", (x_start + 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_color_arc(self, frame: np.ndarray, active_point: tuple):
        h, w = frame.shape[:2]
        center = (w // 2, 0)
        radius = self.palette_cfg['radius']
        inner_radius = self.palette_cfg['inner_radius']
        colors = self.palette_cfg['colors']
        
        num_colors = len(colors)
        angle_step = 180 / num_colors
        
        finger_angle = None
        if active_point:
            dx = active_point[0] - center[0]
            dy = active_point[1] - center[1]
            finger_angle = np.degrees(np.arctan2(dy, dx))
            if finger_angle < 0: finger_angle += 360
            
        overlay = frame.copy()
        for i, color in enumerate(colors):
            start_angle = i * angle_step
            end_angle = (i + 1) * angle_step
            
            alpha = 0.6
            if finger_angle is not None and start_angle <= finger_angle <= end_angle:
                alpha = 0.9
                cv2.ellipse(overlay, center, (radius + 5, radius + 5), 0, start_angle, end_angle, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.ellipse(overlay, center, (radius, radius), 0, start_angle, end_angle, color, -1, cv2.LINE_AA)
            
        cv2.circle(overlay, center, inner_radius, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def get_color_from_point(self, point: tuple, w: int) -> tuple:
        center = (w // 2, 0)
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0: angle += 360
        
        colors = self.palette_cfg['colors']
        angle_step = 180 / len(colors)
        
        if 0 <= angle <= 180:
            idx = int(angle // angle_step)
            return tuple(colors[min(idx, len(colors)-1)])
        return None
