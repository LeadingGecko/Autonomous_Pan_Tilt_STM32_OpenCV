import time
import cv2
from ultralytics import YOLO


from uart_communication import init_serial, send_servo_command, poll_mode_from_serial, encode_telemetry_uart
from telemetry_extract import extract_person_telemetry
from servo_control import compute_servo_targets_from_telemetry

# ========================================
# MAIN CONTROL LOOP
# ========================================
"""
SYSTEM TIMING ANALYSIS:

Typical Frame Pipeline:
1. Capture frame from webcam: ~5-10ms
2. YOLO inference: ~30-100ms (depends on GPU/CPU)
3. Telemetry extraction: <1ms
4. Servo computation: <1ms
5. UART transmission: ~1ms
6. Display rendering: ~5-10ms
Total: ~50-120ms → 8-20 FPS

BOTTLENECKS:
- YOLO inference is slowest component
- Use smaller model (11n) for speed
- GPU acceleration critical
- Reduce image resolution if needed

CONTROL LOOP CHARACTERISTICS:
- Event-driven: Processes each frame as available
- State-based: Tracks mode (AUTO/MANUAL)
- Non-blocking I/O: Doesn't wait for serial responses
"""


def main():
    """
    Main execution loop for gimbal tracking system.
    
    ARCHITECTURE:
    ┌─────────────┐      UART       ┌─────────────┐      BLE        ┌─────────────┐
    │   Laptop    │ ←─────────────→ │   STM32     │ ←─────────────→ │    Phone    │
    │   (YOLO)    │  Commands/Data  │  (Servos)   │   Mode Control  │   (UI App)  │
    └─────────────┘                 └─────────────┘                 └─────────────┘
    FLOW:
    1. Initialize hardware (webcam, serial, YOLO)
    2. Loop:
       a. Check for mode updates from phone (via STM32)
       b. Capture webcam frame
       c. Run YOLO detection
       d. Extract telemetry
       e. Compute servo angles (if AUTO mode)
       f. Send commands to STM32
       g. Send telemetry to STM32 (for BLE forwarding)
       h. Display annotated frame
    3. Cleanup on exit
    """
    
    # ========================================
    # Configuration
    # ========================================
    SERIAL_PORT = "COM6"  # TODO: Update for your system
    BAUD_RATE = 115200
    MODEL_PATH = "yolo11n-seg.pt"
    CONF_THRESH = 0.8     # Detection confidence threshold
    PERSON_CLASS = 0      # COCO class ID for "person"
    IMGSZ = 480           # Inference image size (lower = faster)
    
    # Servo initialization
    INITIAL_PAN = 90
    INITIAL_TILT = 90
    MODE_AUTO = "AUTO"
    MODE_MANUAL = "MANUAL"
    
    # ========================================
    # Hardware Initialization
    # ========================================
    
    # Initialize serial connection
    ser = init_serial(SERIAL_PORT, BAUD_RATE)
    if ser is None:
        print("[WARN] Running in simulation mode (no serial)")
    
    # Load YOLO model
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded. Classes: {model.names}")
    
    # Initialize webcam
    print("[INFO] Opening webcam...")
    webcamera = cv2.VideoCapture(1)
    if not webcamera.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    # Optional: Set camera resolution for better quality
    # webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # ========================================
    # Initialize Tracking State
    # Need to setup mode changes to determine current mode between AUTO and MANUAL 
    # Need to update pan_angle and pan_tilt based on predictions to display on laptop 
    # ========================================
    current_mode = MODE_AUTO  # Default to autonomous tracking
    pan_angle = INITIAL_PAN   # Start centered
    tilt_angle = INITIAL_TILT
    
    # Timing variables for FPS calculation
    last_time = time.time()
    fps = 0.0
    
    # Initialize computed servo angles (will be updated in main loop)
    new_pan = INITIAL_PAN
    new_tilt = INITIAL_TILT
    
    # Send initial servo position
    if ser is not None:
        send_servo_command(ser, pan_angle, tilt_angle)
        time.sleep(0.1)  # Allow servos to reach position
    
    # Debug / Test Statements 
    print("[INFO] System initialized. Starting main loop...")
    print("[INFO] Press 'q' to quit")
    
    # ========================================
    # Main Control Loop
    # ========================================
    try:
        while True:
            # --------------------------------
            # STEP 1: Check Mode Updates
            # --------------------------------
            # Non-blocking check for mode changes from phone
            if ser is not None:
                current_mode = poll_mode_from_serial(ser, current_mode)
            
            # --------------------------------
            # STEP 2: Capture Frame
            # --------------------------------
            success, frame = webcamera.read()
            if not success:
                print("[WARN] Failed to capture frame")
                break
            
            frame_h, frame_w = frame.shape[:2]
            
            # --------------------------------
            # STEP 3: YOLO Detection
            # --------------------------------
            # Run tracking on current frame
            # .track() maintains object IDs across frames (vs .predict())
            results = model.track(
                source=frame,
                classes=[PERSON_CLASS],  # Only detect persons
                conf=CONF_THRESH,        # Confidence threshold
                imgsz=IMGSZ,             # Resize for inference
                verbose=False,           # Suppress per-frame logging
                persist=True             # Maintain tracking IDs
            )
            
            result = results[0]  # Single frame result
            
            # --------------------------------
            # STEP 4: Compute FPS
            # Time Library to calculate FPS
            # --------------------------------
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 1.0 / dt
            
            # --------------------------------
            # STEP 5: Extract Telemetry
            # From telemetry_extract.py need to use to determine angles, tracking status, etc.
            # --------------------------------
            telemetry = extract_person_telemetry(
                result=result,
                frame_shape=frame.shape,
                mode=current_mode,
                servo_pan=pan_angle,
                servo_tilt=tilt_angle,
                fps=fps
            )
            
            # --------------------------------
            # STEP 6: Servo Control Logic
            # --------------------------------
            # Always compute new servo targets (for display in all modes)
            new_pan, new_tilt = compute_servo_targets_from_telemetry(
                telemetry=telemetry,
                frame_shape=frame.shape,
                pan_angle=pan_angle,
                tilt_angle=tilt_angle
            )
            
            # Only send commands in AUTO mode with serial connection
            if ser is not None and current_mode == MODE_AUTO:
                # Send computed angles to STM32
                send_servo_command(ser, new_pan, new_tilt)
                # Update tracking state with sent angles
                pan_angle, tilt_angle = new_pan, new_tilt
            
            # In MANUAL mode, servos controlled by phone via STM32
            # Computed angles still displayed for reference
            
            # --------------------------------
            # STEP 7: Transmit Telemetry
            # --------------------------------
            # Send telemetry to STM32 for BLE forwarding to phone
            if ser is not None:
                telem_line = encode_telemetry_uart(telemetry)
                ser.write(telem_line.encode("ascii"))
            
            # --------------------------------
            # STEP 8: Visualization
            # Display annotated frame with info overlays
            # FPS / MODE / TRACKING STATUS / SERVO ANGLES 
            # --------------------------------
            # Draw bounding boxes and tracking info
            display_frame = result.plot()  # YOLO's built-in visualization
            
            # Overlay mode indicator
            mode_color = (0, 255, 0) if current_mode == MODE_AUTO else (0, 255, 255)
            cv2.putText(
                display_frame,
                f"Mode: {current_mode}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                mode_color,
                2,
                cv2.LINE_AA
            )
            
            # Tracking status and FPS
            cv2.putText(
                display_frame,
                f"Tracking: {telemetry['tracking']}  FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Servo angles
            # UPDATE pan_angle and tilt_angle based on predictions to display on laptop
            cv2.putText(
                display_frame,
                f"Pan: {new_pan:.1f}°  Tilt: {new_tilt:.1f}°",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
            
            # Draw center crosshair
            center_x, center_y = frame_w // 2, frame_h // 2
            cv2.drawMarker(
                display_frame,
                (center_x, center_y),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                20,
                2
            )
            
            # Show frame
            cv2.imshow("Gimbal Tracking System", display_frame)
            
            # --------------------------------
            # STEP 9: Check for Exit
            # --------------------------------
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User requested exit")
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ========================================
        # Cleanup
        # ========================================
        print("[INFO] Cleaning up...")
        
        # Release webcam
        webcamera.release()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Close serial port
        if ser is not None and ser.is_open:
            # Optional: Send center command before closing
            send_servo_command(ser, INITIAL_PAN, INITIAL_TILT)
            time.sleep(0.1)
            ser.close()
        
        print("[INFO] Shutdown complete")


# ========================================
# PERFORMANCE OPTIMIZATION TIPS
# ========================================
"""
1. GPU ACCELERATION:
   - Install CUDA + cuDNN for PyTorch
   - Verify: print(torch.cuda.is_available())
   - 5-10x speedup over CPU

2. MODEL OPTIMIZATION:
   - Use YOLO11n (nano) for speed
   - Consider INT8 quantization
   - Export to TensorRT for max performance

3. RESOLUTION TUNING:
   - Lower imgsz: 320 (faster) vs 640 (accurate)
   - Reduce webcam resolution if needed
   - Balance detection range vs speed

4. MULTI-THREADING (Advanced):
   - Separate threads for capture, inference, display
   - Use queue for inter-thread communication
   - Prevents blocking in any single operation

5. FRAME SKIPPING:
   - Skip inference on every Nth frame
   - Interpolate servo commands between detections
   - Maintains smooth motion with less compute
"""

if __name__ == "__main__":
    main()