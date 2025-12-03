def compute_servo_targets_from_telemetry(telemetry, frame_shape, pan_angle, tilt_angle):
    """
    Compute servo target angles based on detected person telemetry.
    
    Args:
        telemetry: Dict with keys like 'cx', 'cy' (center), 'tracking' (bool)
        frame_shape: (height, width, channels)
        pan_angle: Current pan angle (0–180°)
        tilt_angle: Current tilt angle (0–180°)
    
    Returns:
        (new_pan, new_tilt): Computed servo angles
    """
    frame_h, frame_w = frame_shape[:2]
    center_x, center_y = frame_w // 2, frame_h // 2
    
    # If not tracking, hold current angles
    if not telemetry['tracking']:
        return pan_angle, tilt_angle
    
    # Get detected person center
    person_cx = telemetry.get('cx', center_x)
    person_cy = telemetry.get('cy', center_y)
    
    # ────────────────────────────────────
    # PAN COMPUTATION (Horizontal)
    # ────────────────────────────────────
    # Error: how far person is from center (pixels)
    pan_error_px = person_cx - center_x
    
    # Convert pixel error to angle (proportional control)
    # Example: assume 30° per 320 pixels width
    pan_gain = 30.0 / (frame_w / 2)  # degrees per pixel
    pan_delta = pan_error_px * pan_gain
    
    # Smoothing: blend old and new (low-pass filter)
    alpha = 0.3  # 0.0 = no movement, 1.0 = instant
    new_pan = pan_angle + (pan_delta * alpha)
    new_pan = max(0, min(180, new_pan))  # Clamp to 0–180°
    
    # ────────────────────────────────────
    # TILT COMPUTATION (Vertical)
    # ────────────────────────────────────
    # Error: how far person is from center (pixels)
    tilt_error_px = person_cy - center_y
    
    # Convert pixel error to angle (proportional control)
    # Example: assume 22.5° per 240 pixels height
    tilt_gain = 22.5 / (frame_h / 2)  # degrees per pixel
    tilt_delta = tilt_error_px * tilt_gain
    
    # Smoothing: blend old and new
    new_tilt = tilt_angle + (tilt_delta * alpha)
    new_tilt = max(0, min(180, new_tilt))  # Clamp to 0–180°
    
    return new_pan, new_tilt


# ========================================
# ADVANCED: PID Controller (Optional Enhancement)
# ========================================
# For even better tracking, consider implementing full PID:

class PIDController:
    """
    Full PID controller for smoother tracking.
    
    P (Proportional): Responds to current error
    I (Integral): Eliminates steady-state error
    D (Derivative): Reduces overshoot/oscillation
    
    Use when:
    - P-controller causes oscillation
    - Target tracking has lag
    - Need zero steady-state error
    """
    def __init__(self, kp, ki, kd, output_min, output_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute(self, error, dt):
        """
        Compute PID output.
        
        error: Current error value
        dt: Time since last update (seconds)
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (accumulated error over time)
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term (rate of change of error)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative
        
        # Combine terms
        output = p_term + i_term + d_term
        
        # Clamp output
        output = max(self.output_min, min(self.output_max, output))
        
        # Update state
        self.prev_error = error
        
        return output