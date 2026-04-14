import cv2
import numpy as np
import yaml
from typing import List, Tuple, Dict, Optional

class CanvasManager:
    """
    Manages 3-layer BGRA canvas with multi-hand stroke tracking and undo/redo.
    """
    def __init__(self, width: int, height: int, config_path: str = "config.yaml"):
        self.w = width
        self.h = height
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.layer_configs = self.config['layers']
        self.layers = [np.zeros((height, width, 4), dtype=np.uint8) for _ in range(len(self.layer_configs))]
        self.active_layer_idx = 0
        
        # Multi-hand stroke tracking
        self.hand_last_points = {} # {hand_id: (x, y)}
        self.active_strokes = {}   # {hand_id: {'points': [], 'color': (), 'thickness': int}}
        
        # History for active layer
        self.history = []
        self.redo_stack = []
        self.strokes = [] # List of completed {'points': [], 'color': (), 'thickness': int}
        
    def draw_stroke(self, hand_id: str, p1: tuple, p2: tuple, color: tuple, size: int):
        """Draw a line segment on the active layer."""
        alpha_color = (*color, 255)
        cv2.line(self.layers[self.active_layer_idx], p1, p2, alpha_color, size, cv2.LINE_AA)
        
        # Track for SVG
        if hand_id not in self.active_strokes:
            self.active_strokes[hand_id] = {'points': [p1], 'color': color, 'thickness': size}
        self.active_strokes[hand_id]['points'].append(p2)

    def finalize_stroke(self, hand_id: str):
        """Called when a hand stops drawing."""
        if hand_id in self.active_strokes:
            self.strokes.append(self.active_strokes.pop(hand_id))
            
        if hand_id in self.hand_last_points:
            # Simple undo checkpoint: Save state when any hand finishes a stroke
            self.history.append([layer.copy() for layer in self.layers])
            if len(self.history) > 20: self.history.pop(0)
            self.redo_stack.clear()
            self.hand_last_points[hand_id] = None

    def composite(self, bg_frame: np.ndarray) -> np.ndarray:
        """Composite all layers onto the BGR background frame."""
        output = bg_frame.astype(float)
        
        for i, layer in enumerate(self.layers):
            alpha_multiplier = self.layer_configs[i]['opacity']
            layer_alpha = (layer[:, :, 3].astype(float) / 255.0) * alpha_multiplier
            alpha_mask = layer_alpha[..., np.newaxis]
            
            layer_rgb = layer[:, :, :3].astype(float)
            output = (1.0 - alpha_mask) * output + alpha_mask * layer_rgb
                
        return np.clip(output, 0, 255).astype(np.uint8)

    def cycle_layer(self):
        self.active_layer_idx = (self.active_layer_idx + 1) % len(self.layers)

    def undo(self):
        if self.history:
            self.redo_stack.append([layer.copy() for layer in self.layers])
            last_state = self.history.pop()
            self.layers = last_state

    def redo(self):
        if self.redo_stack:
            self.history.append([layer.copy() for layer in self.layers])
            next_state = self.redo_stack.pop()
            self.layers = next_state

    def clear_active_layer(self):
        self.history.append([layer.copy() for layer in self.layers])
        self.layers[self.active_layer_idx].fill(0)

    def draw_shape(self, shape_type: str, pos: tuple, size: int, color: tuple):
        alpha_color = (*color, 255)
        layer = self.layers[2] # Shapes Layer
        if shape_type == "circle":
            cv2.circle(layer, pos, size, alpha_color, -1, cv2.LINE_AA)
        self.history.append([layer.copy() for layer in self.layers])

    def get_merged_canvas(self, bg_color=(0, 0, 0)):
        result = np.full((self.h, self.w, 3), bg_color, dtype=np.uint8)
        return self.composite(result)
