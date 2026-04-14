# GestureBoard Pro 🎨✨

GestureBoard Pro is a real-time, AI-powered virtual whiteboard controlled entirely by hand gestures via webcam. It uses advanced computer vision to translate the movement of your fingers into precise drawing strokes and intuitive tool controls.

## Features

- **Liquid-Smooth Interaction:** Powered by a customized Kalman filter system for zero-lag, jitter-free brush strokes.
- **Dual-Hand Tracking:** Perform actions simultaneously with both hands for maximum creative speed.
- **Visual Color Arc:** Seamlessly pick from a beautiful semi-circular color wheel hovering at the top of your screen.
- **3-Layer Compositing Engine:** Draw across three independent layers (Sketch, Annotation, Shapes) without destroying underlying work.
- **High-Performance Architecture:** Employs an asynchronous AI pipeline and vectorized alpha-blending to maintain rock-solid 30-60 FPS on macOS.
- **Export Powerhouse:** Instantly capture your work as SVG vectors, PNG images, and Timelapse GIFs to share with the world.
- **Local Web Stream:** Optionally stream your drawing session over a local network using Flask MJPEG streaming.

## 🛠 Tech Stack

- **Python 3.11**
- **MediaPipe Tasks API**: For high-speed, robust hand landmark tracking.
- **OpenCV**: For the compositing engine and high-performance canvas rendering.
- **Flask**: For local network streaming (Port 8080).
- **NumPy**: For optimized math and vector operations.
- **YAML**: Configuration management without hardcoding.

## ✨ 10-Gesture Control System

Control your entire canvas without ever touching your mouse or keyboard!

| Action | Emoji | Hand Gesture | Digital Meaning |
| :--- | :---: | :--- | :--- |
| **DRAW** | ☝️ | Index Finger Up | Draws freehand strokes on the current layer. |
| **COLOR_PICK** | ✌️ | Index + Middle Up | Opens the Color Arc. Hover over segments to pick. |
| **BRUSH_SIZE** | 🤏 | Thumb + Index Pinch | Constant 8px professional brush size lock. |
| **PAUSE** | ✊ | Closed Fist | Stop drawing and hover without making marks. |
| **CLEAR** | 🖐️ | Open Palm (1s) | Instantly erases the active layer. |
| **UNDO** | ⬅️ | Swipe Left (✌️) | Revert your last drawing action. |
| **REDO** | ➡️ | Swipe Right (✌️) | Redo an action you previously undid. |
| **SHAPE_SNAP** | 🫰 | Crossed Fingers | Snaps a perfect circle to your finger tip. |
| **SWITCH_LAYER** | 🤙 | Thumb + Pinky Up | Cycles between Sketch, Annotation, and Shapes layers. |
| **EXPORT** | 👌 | OK Sign (2s) | Saves your work as PNG, SVG, and Timelapse GIF. |

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Naina2006-Vismi/GestureBoard-Pro.git
   cd GestureBoard-Pro
   ```

2. **Set up a Virtual Environment (Optional but recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python main.py
   ```

Your built-in FaceTime camera will initialize. You can view the Live Feed directly in the OpenCV window, or connect to the Stream at `http://localhost:8080/video_feed`.

*PRO TIP: Press `H` at any time while the application is running to pull up the on-screen Gesture Guide!*
