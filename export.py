import cv2
import numpy as np
import imageio
import yaml
import os
from datetime import datetime
from typing import List, Dict, Any

class Exporter:
    """
    Handles PNG, SVG, and GIF exports for GestureBoard Pro.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.exp_cfg = self.config['export']
        
        if not os.path.exists("exports"):
            os.makedirs("exports")
            
    def export_png(self, image: np.ndarray, prefix: str = "whiteboard"):
        filename = f"exports/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, image)
        return filename

    def export_svg(self, strokes: List[Dict[str, Any]], width: int, height: int):
        filename = f"exports/whiteboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
        
        with open(filename, 'w') as f:
            f.write(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
            f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
            f.write(f'<rect width="100%" height="100%" fill="black" />\n')
            
            for stroke in strokes:
                points = stroke['points']
                if len(points) < 2: continue
                
                color = stroke['color'] # BGR
                rgb_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
                thickness = stroke['thickness']
                
                # Simple path implementation using quadratic beziers
                # Format: M x1 y1 Q x1 y1, (x1+x2)/2 (y1+y2)/2 T ...
                path_data = f'M {points[0][0]} {points[0][1]}'
                
                for i in range(1, len(points) - 1):
                    p1 = points[i]
                    p2 = points[i+1]
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    path_data += f' Q {p1[0]} {p1[1]}, {mid_x} {mid_y}'
                
                # Close with final point
                path_data += f' L {points[-1][0]} {points[-1][1]}'
                
                f.write(f'  <path d="{path_data}" stroke="{rgb_hex}" stroke-width="{thickness}" fill="none" stroke-linecap="round" stroke-linejoin="round" />\n')
            
            f.write('</svg>')
        return filename

    def export_gif(self, frames: List[np.ndarray]):
        if not frames: return None
        filename = f"exports/timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        
        # Resize frames for GIF performance
        processed = []
        for frame in frames:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scaled = cv2.resize(rgb, (int(w * self.exp_cfg['gif_scale']), int(h * self.exp_cfg['gif_scale'])))
            processed.append(scaled)
            
        duration = int(1000 / self.exp_cfg['gif_fps'])
        imageio.mimsave(filename, processed, duration=duration, loop=0)
        return filename
