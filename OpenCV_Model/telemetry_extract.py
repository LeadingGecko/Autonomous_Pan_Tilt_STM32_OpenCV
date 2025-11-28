def extract_person_telemetry(result, frame_shape, mode, servo_pan, servo_tilt, fps):
    """
    Extracts bounding box telemetry from YOLO detection result.
    
    LOGIC:
    1. Parse YOLO result object to extract bounding boxes
    2. Select highest confidence detection (YOLO sorts by confidence)
    3. Compute bounding box center (cx, cy) - our tracking target
    4. Calculate box dimensions for distance estimation
    5. Package all data for downstream control and transmission
    
    PARAMETERS:
    - result: Ultralytics YOLO result object containing detections
    - frame_shape: Tuple (H, W, C) - needed to normalize coordinates
    - mode: Current system mode ("AUTO" or "MANUAL")
    - servo_pan/tilt: Current servo angles (degrees)
    - fps: Frame processing rate
    
    RETURNS:
    Dictionary with tracking state and measurements:
    {
        "mode": str,           # System operating mode
        "tracking": 0/1,       # 1 if person detected, 0 otherwise
        "cx": int,             # Bounding box center X (pixels)
        "cy": int,             # Bounding box center Y (pixels)
        "bw": int,             # Box width (for distance estimation)
        "bh": int,             # Box height (for distance estimation)
        "conf": float,         # Detection confidence [0.0-1.0]
        "servo_pan": float,    # Current pan angle (degrees)
        "servo_tilt": float,   # Current tilt angle (degrees)
        "fps": float,          # Processing framerate
    }
    
    REASONING:
    - Using bounding box CENTER rather than corners ensures smooth tracking
    - Box dimensions (bw, bh) can estimate person distance (larger box = closer)
    - Confidence score allows filtering unreliable detections
    - Including servo state enables closed-loop verification
    """
    
    h, w = frame_shape[:2]

    # Default telemetry when no detection
    telemetry = {
        "mode": mode,
        "tracking": 0,        # No person found
        "cx": -1,             # Invalid coordinate marker
        "cy": -1,
        "bw": 0,
        "bh": 0,
        "conf": 0.0,
        "servo_pan": servo_pan,
        "servo_tilt": servo_tilt,
        "fps": fps,
    }

    # Extract bounding boxes from YOLO result
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return telemetry  # No detections - return default

    # Get first detection (highest confidence due to YOLO sorting)
    # Handle both tensor and numpy array formats
    try:
        xyxy = boxes.xyxy[0].cpu().numpy()  # GPU tensor case
    except Exception:
        xyxy = boxes.xyxy[0]  # Already numpy or CPU tensor

    # Parse bounding box coordinates
    # xyxy format: [x1, y1, x2, y2] where (x1,y1) = top-left, (x2,y2) = bottom-right
    x1, y1, x2, y2 = xyxy
    
    # Calculate box dimensions
    bw = x2 - x1  # Width in pixels
    bh = y2 - y1  # Height in pixels
    
    # Compute CENTER of bounding box (our tracking target point)
    # This is more stable than tracking corners
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    # Extract confidence score
    try:
        conf = float(boxes.conf[0])
    except Exception:
        conf = 0.0

    # Update telemetry with detection data
    telemetry.update({
        "tracking": 1,      # Person successfully detected
        "cx": int(cx),      # Cast to int for transmission efficiency
        "cy": int(cy),
        "bw": int(bw),
        "bh": int(bh),
        "conf": conf,       # Keep as float for precision
    })
    
    return telemetry