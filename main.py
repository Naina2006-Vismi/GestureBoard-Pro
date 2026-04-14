import cv2
import time
import threading
import numpy as np
import yaml
from flask import Flask, Response
from gesture_engine import GestureEngine
from canvas import CanvasManager
from hud import HUDRenderer
from smoothing import PointSmoother
from export import Exporter

# Shared state for Flask stream
stream_frame = None
lock = threading.Lock()

app = Flask(__name__)

class WebcamStream:
    """Threaded camera reader to maximize FPS and unblock the main thread."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(1)
            
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if stream_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', stream_frame)
                frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    camera_stream = WebcamStream(src=0).start()
    time.sleep(1.0)
    test_frame = camera_stream.read()
    if test_frame is None:
        print("Error: Could not access built-in camera.")
        return

    h, w = test_frame.shape[:2]
    
    # Load config for fixed brush size
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    engine = GestureEngine()
    canvas = CanvasManager(w, h)
    hud = HUDRenderer()
    smoother = PointSmoother()
    exporter = Exporter()
    
    # State
    brush_color = (0, 255, 255)
    brush_size = config['gestures'].get('fixed_brush_size', 8) # Load fixed size
    frames_for_gif = []
    frame_count = 0 
    prev_time = time.time()
    last_ai_time = 0
    ai_interval = 1.0 / 60.0 # Increased AI throttle to 60 FPS for "react fast"
    show_help = True
    
    print("GestureBoard Pro Started. Press 'H' for Help, 'q' to quit.")
    
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()

    layer_debounce = False

    while True:
        loop_start = time.time()
        
        frame = camera_stream.read()
        if frame is None: continue
        frame = cv2.flip(frame, 1)
        
        # 1. Throttled AI Detection (60 FPS)
        curr_time = time.time()
        if curr_time - last_ai_time > ai_interval:
            engine.send_frame(frame)
            last_ai_time = curr_time
        
        # 2. Get latest detection results
        hand_results = engine.get_latest_results()
        active_hands = [hr['id'] for hr in hand_results]
        
        # 3. Process Each Hand
        for hr in hand_results:
            gesture = hr['gesture']
            landmarks = hr['landmarks']
            hand_id = hr['id']
            
            it = landmarks[8]
            sx, sy = smoother.update(hand_id, int(it.x * w), int(it.y * h))
            curr_p = (sx, sy)
            
            if gesture == "DRAW":
                last_p = canvas.hand_last_points.get(hand_id)
                if last_p:
                    canvas.draw_stroke(hand_id, last_p, curr_p, brush_color, brush_size)
                canvas.hand_last_points[hand_id] = curr_p
            
            elif gesture == "COLOR_PICK":
                new_color = hud.get_color_from_point(curr_p, w)
                if new_color: brush_color = new_color
                
            # BRUSH_SIZE gesture logic REMOVED. brush_size stays constant.
                
            elif gesture == "CLEAR":
                canvas.clear_active_layer()
                engine.reset_holds(hand_id)
                
            elif gesture == "UNDO":
                canvas.undo()
                engine.reset_holds(hand_id)
                
            elif gesture == "REDO":
                canvas.redo()
                engine.reset_holds(hand_id)
                
            elif gesture == "EXPORT":
                img = canvas.get_merged_canvas()
                exporter.export_png(img)
                exporter.export_svg(canvas.strokes, w, h)
                exporter.export_gif(frames_for_gif)
                frames_for_gif = []
                engine.reset_holds(hand_id)

            if gesture != "DRAW":
                if hand_id in canvas.hand_last_points and canvas.hand_last_points[hand_id]:
                    canvas.finalize_stroke(hand_id)

        disappeared = set(canvas.hand_last_points.keys()) - set(active_hands)
        for d_id in disappeared:
            canvas.finalize_stroke(d_id)
            smoother.reset(d_id)

        # 4. Final Compositing & HUD
        latency = (time.time() - loop_start) * 1000
        fps = 1.0 / (time.time() - prev_time)
        prev_time = time.time()
        
        composited = canvas.composite(frame)
        
        hud_state = {
            'brush_size': brush_size,
            'brush_color': brush_color,
            'active_layer': canvas.layer_configs[canvas.active_layer_idx]['name'],
            'undo_v': len(canvas.history) > 0,
            'redo_v': len(canvas.redo_stack) > 0,
            'gesture': "MULTI" if len(hand_results) > 1 else (hand_results[0]['gesture'] if hand_results else "NONE")
        }
        hud_stats = {'fps': fps, 'latency': latency}
        
        hud.draw_hud(composited, hud_stats, hud_state, hand_results, show_help)
        
        if frame_count % config['export']['gif_interval'] == 0:
            frames_for_gif.append(composited.copy())
            if len(frames_for_gif) > 100: frames_for_gif.pop(0)

        with lock:
            stream_frame = composited.copy()
            
        cv2.imshow("GestureBoard Pro", composited)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
            
        frame_count += 1

    camera_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
