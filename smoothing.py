import numpy as np

class KalmanSmoother:
    """1D Kalman filter to smooth coordinates."""
    def __init__(self, process_noise=0.01, measurement_noise=0.01):
        # Increased process_noise and decreased measurement_noise for zero-lag responsiveness
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = None
        self.error_cov = 1.0

    def update(self, measurement):
        if self.state is None:
            self.state = measurement
            return measurement

        # Prediction
        self.error_cov += self.process_noise

        # Measurement Update
        kalman_gain = self.error_cov / (self.error_cov + self.measurement_noise)
        self.state = self.state + kalman_gain * (measurement - self.state)
        self.error_cov = (1 - kalman_gain) * self.error_cov

        return self.state

class PointSmoother:
    """Manages separate Kalman filters for multiple hands."""
    def __init__(self, process_noise=0.01, measurement_noise=0.01):
        self.params = (process_noise, measurement_noise)
        self.filters = {} # {hand_id: (kx, ky)}

    def update(self, hand_id, x: int, y: int) -> tuple:
        if hand_id not in self.filters:
            self.filters[hand_id] = (
                KalmanSmoother(*self.params),
                KalmanSmoother(*self.params)
            )
        
        kx, ky = self.filters[hand_id]
        sx = int(kx.update(x))
        sy = int(ky.update(y))
        return (sx, sy)

    def reset(self, hand_id=None):
        if hand_id is None:
            self.filters = {}
        elif hand_id in self.filters:
            del self.filters[hand_id]
