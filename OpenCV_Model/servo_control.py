def compute_servo_targets_from_telemetry(telemetry, frame_shape, pan_angle, tilt_angle):
    """
    Proportional controller to compute servo angles that center the target.
    
    CONTROL THEORY:
    - Implements a P-controller (Proportional control)
    - Error = distance from frame center to target center
    - Output = angle adjustment proportional to error
    - Formula: new_angle = current_angle + Kp * error
    
    WHY PROPORTIONAL CONTROL:
    1. Simple and stable for visual servoing
    2. Large errors → fast correction
    3. Small errors → gentle adjustment (prevents oscillation)
    4. No overshoot risk with proper Kp tuning
    
    COORDINATE SYSTEM:
    - Image origin (0,0) is TOP-LEFT
    - X increases RIGHT, Y increases DOWN
    - Frame center = (W/2, H/2)
    - Positive error: target is right/below center
    
    PARAMETERS:
    - telemetry: Dict with tracking=1/0, cx, cy
    - frame_shape: (H, W, C) for computing center
    - pan_angle: Current pan servo angle (degrees)
    - tilt_angle: Current tilt servo angle (degrees)
    
    RETURNS:
    (new_pan, new_tilt) - Updated angles in degrees
    
    TUNING GUIDE:
    - Increase Kp → faster response, risk of oscillation
    - Decrease Kp → smoother motion, slower tracking
    - Deadzone prevents micro-jitter when nearly centered
    """
    
    h, w = frame_shape[:2]
    
    # Configuration constants
    KP_X = 0.05         # Pan gain: degrees per pixel error
    KP_Y = 0.05         # Tilt gain: degrees per pixel error
    DEADZONE_X = 10     # Ignore errors within ±10 pixels (horizontal)
    DEADZONE_Y = 10     # Ignore errors within ±10 pixels (vertical)
    PAN_MIN, PAN_MAX = 20, 160    # Physical servo limits
    TILT_MIN, TILT_MAX = 30, 150

    # If no person detected, maintain current orientation
    if telemetry["tracking"] == 0:
        return pan_angle, tilt_angle

    # ========================================
    # STEP 1: Calculate Pixel Error
    # ========================================
    # Error = target position - desired position (frame center)
    # Positive cx_err means target is RIGHT of center → pan right
    # Positive cy_err means target is BELOW center → tilt down
    
    frame_center_x = w / 2.0
    frame_center_y = h / 2.0
    
    cx_err = telemetry["cx"] - frame_center_x  
    cy_err = telemetry["cy"] - frame_center_y
    
    # ========================================
    # STEP 2: Apply Deadzone
    # ========================================
    # Prevents constant micro-adjustments when target is nearly centered
    # Example: If target is 5px off-center but deadzone=10, ignore it
    
    if abs(cx_err) < DEADZONE_X:
        cx_err = 0.0
    if abs(cy_err) < DEADZONE_Y:
        cy_err = 0.0
    
    # ========================================
    # STEP 3: Proportional Control
    # ========================================
    # Convert pixel error to angle adjustment
    # Sign convention depends on your mechanical setup
    
    # PAN CONTROL:
    # - If person is RIGHT (+cx_err), we want to pan RIGHT (increase angle)
    # - The negative sign may need adjustment based on your servo orientation
    # - Test and flip sign if servo moves opposite direction
    pan_delta = -KP_X * cx_err
    
    # TILT CONTROL:
    # - If person is BELOW (+cy_err), we want to tilt DOWN (increase angle)
    # - Again, verify sign with your hardware
    tilt_delta = KP_Y * cy_err
    
    # ========================================
    # STEP 4: Update Angles & Clamp
    # ========================================
    # Add computed deltas to current angles
    new_pan = pan_angle + pan_delta
    new_tilt = tilt_angle + tilt_delta
    
    # Enforce physical servo limits to prevent damage
    new_pan = clamp(new_pan, PAN_MIN, PAN_MAX)
    new_tilt = clamp(new_tilt, TILT_MIN, TILT_MAX)
    
    return new_pan, new_tilt


def clamp(val, vmin, vmax):
    """
    Constrain value to range [vmin, vmax].
    
    REASONING:
    - Prevents servo commands outside safe operating range
    - Protects hardware from mechanical damage
    - Ensures smooth behavior at limits (no sudden stops)
    """
    return max(vmin, min(vmax, val))


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