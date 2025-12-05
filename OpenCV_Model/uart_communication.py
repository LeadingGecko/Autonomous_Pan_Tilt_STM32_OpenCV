import serial

# ========================================
# UART PROTOCOL DESIGN
# ========================================
"""
PROTOCOL SPECIFICATION:

1. TELEMETRY (Laptop → STM32 → BLE → Phone)
   Format: T,<mode>,<tracking>,<cx>,<cy>,<bw>,<bh>,<conf>,<pan>,<tilt>,<fps>
   Example: T,AUTO,1,312,245,150,280,0.92,123.0,88.0,9.7
   
   Field breakdown:
   - T: Message type identifier
   - mode: AUTO or MANUAL
   - tracking: 1=person detected, 0=no detection
   - cx,cy: Bounding box center (pixels)
   - bw,bh: Box dimensions (pixels)
   - conf: Detection confidence 0.0-1.0
   - pan,tilt: Current servo angles (degrees)
   - fps: Processing framerate

2. SERVO COMMAND (Laptop → STM32)
   Format: C,<pan>,<tilt>
   Example: C,120,95
   
   - C: Command identifier
   - pan: Pan servo angle 20-160 degrees
   - tilt: Tilt servo angle 30-150 degrees

3. STATUS UPDATE (Phone → BLE → STM32 → Laptop)
   Format: STATUS,MODE,<mode>
   Example: STATUS,MODE,AUTO
   
   - Sent when user changes mode on phone
   - Laptop uses this to decide whether to send servo commands

WHY THIS PROTOCOL:
- ASCII format: Human-readable, easy to debug
- CSV structure: Simple parsing on MCU (no JSON overhead)
- Fixed message types: Fast identification
- Compact: Efficient bandwidth usage
- Extensible: Easy to add fields
"""


def encode_telemetry_uart(t):
    """
    Encode telemetry dictionary into UART-transmittable ASCII string.
    
    REASONING:
    - CSV format minimizes parsing complexity on resource-constrained MCU
    - Fixed field order ensures consistent parsing
    - Newline delimiter allows line-buffered reading
    - Decimal precision balanced for accuracy vs bandwidth
    
    PARAMETERS:
    t: Telemetry dictionary (output of extract_person_telemetry)
    
    RETURNS:
    ASCII string ending with '\n', ready for serial.write()
    
    EXAMPLE OUTPUT:
    "T,AUTO,1,312,245,150,280,0.92,123.0,88.0,9.7\n"
    """
    # To make parsing on resource-constrained MCUs simple and avoid
    # floating-point scanf/printf requirements, we serialize numeric
    # values as integers where reasonable:
    #  - cx,cy,bw,bh: integers (pixels)
    #  - conf: integer percent [0..100]
    #  - servo_pan, servo_tilt: integer degrees
    #  - fps: integer (rounded)
    line = (
        f"T,"                      # Message type
        f"{t['mode']},"           # AUTO or MANUAL
        f"{t['tracking']},"       # 0 or 1
        f"{int(t['cx'])},{int(t['cy'])},"   # Bounding box center
        f"{int(t['bw'])},{int(t['bh'])},"   # Box dimensions
        f"{int(round(t['conf'] * 100))},"    # Confidence as percent (0-100)
        f"{int(round(t['servo_pan']))},"     # Pan angle (degrees)
        f"{int(round(t['servo_tilt']))},"    # Tilt angle (degrees)
        f"{int(round(t['fps']))}\n"         # Framerate (rounded)
    )
    return line


def send_servo_command(ser, pan_angle, tilt_angle):
    """
    Send servo position command to STM32 over UART.
    
    PROTOCOL: C,<pan>,<tilt>
    *** Change to just <pan><tilt> what is C Prefix for ? 
    
    TIMING CONSIDERATIONS:
    - STM32 should acknowledge or implement command buffering
    - At 115200 baud, transmission time ~1ms per command
    - Send rate should match YOLO inference rate (typically 10-30 Hz)
    
    PARAMETERS:
    ser: pyserial Serial object
    pan_angle: Target pan angle (degrees)
    tilt_angle: Target tilt angle (degrees)
    
    ERROR HANDLING:
    - Angles pre-clamped by compute_servo_targets()
    - Cast to int to reduce transmission size
    - STM32 should validate and clamp again (defense in depth)
    """
    # Format command string
    cmd = f"C,{int(pan_angle)},{int(tilt_angle)}\n"
    
    # Transmit over UART
    ser.write(cmd.encode("ascii"))
    
    # Optional: Read acknowledgment from STM32
    # ack = ser.readline().decode('ascii').strip()
    # if ack != "OK":
    #     print(f"[WARN] Servo command not acknowledged")


def poll_mode_from_serial(ser, current_mode):
    """
    Non-blocking read of mode updates from STM32.
    
    COMMUNICATION FLOW:
    1. User changes mode on phone app
    2. Phone sends BLE command to STM32
    3. STM32 forwards "STATUS,MODE,<mode>" over UART
    4. This function catches it and updates tracking state
    
    NON-BLOCKING DESIGN:
    - Uses ser.in_waiting to check buffer before read
    - Prevents blocking main tracking loop
    - Processes all pending messages in one call
    
    PARAMETERS:
    ser: pyserial Serial object
    current_mode: Last known mode ("AUTO" or "MANUAL")
    
    RETURNS:
    Updated mode string
    
    ROBUSTNESS:
    - Malformed messages ignored
    - Case-insensitive parsing
    - Exception handling prevents crashes
    """
    MODE_AUTO = "AUTO"
    MODE_MANUAL = "MANUAL"
    
    try:
        # Process all pending messages in serial buffer
        while ser.in_waiting:
            # Read one line (terminated by '\n')
            raw = ser.readline().decode("ascii", errors="ignore").strip()
            
            if not raw:
                continue  # Empty line, skip
            
            # Debug logging (optional)
            # print(f"[UART RX] {raw}")
            
            # Parse status message
            if raw.startswith("STATUS,MODE,"):
                try:
                    # Split: "STATUS,MODE,AUTO" → ["STATUS", "MODE", "AUTO"]
                    parts = raw.split(",", 2)
                    if len(parts) >= 3:
                        mode_str = parts[2].upper()
                        
                        # Validate mode string
                        if mode_str.startswith("AUTO"):
                            current_mode = MODE_AUTO
                            print(f"[INFO] Mode changed to AUTO")
                        elif mode_str.startswith("MANUAL"):
                            current_mode = MODE_MANUAL
                            print(f"[INFO] Mode changed to MANUAL")
                        
                except (ValueError, IndexError):
                    # Malformed message, ignore
                    print(f"[WARN] Malformed status message: {raw}")
                    pass
    
    except serial.SerialException as e:
        # Handle disconnection gracefully
        print(f"[ERROR] Serial error: {e}")
        pass
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"[ERROR] Unexpected error in poll_mode_from_serial: {e}")
        pass
    
    return current_mode


# ========================================
# Serial Port Initialization
# ========================================

def init_serial(port, baudrate, timeout=0.01):
    """
    Initialize UART connection to STM32.
    
    CONFIGURATION:
    - Baud rate: 115200 (good balance of speed and reliability)
    - Timeout: 0.01s (10ms) for non-blocking reads
    - Data format: 8N1 (8 bits, no parity, 1 stop bit) - default
    
    TROUBLESHOOTING:
    - Linux: Port typically /dev/ttyACM0 or /dev/ttyUSB0
    - Windows: Port typically COM3, COM4, etc.
    - Mac: Port typically /dev/cu.usbmodem*
    - Check: ls /dev/tty* (Linux/Mac) or Device Manager (Windows)
    
    PARAMETERS:
    port: String port identifier (e.g., "COM8" or "/dev/ttyACM0")
    baudrate: Communication speed (must match STM32 config)
    timeout: Read timeout in seconds
    
    RETURNS:
    pyserial Serial object or None on failure
    """
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        
        # Flush any stale data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        print(f"[INFO] Serial port {port} opened at {baudrate} baud")
        return ser
        
    except serial.SerialException as e:
        print(f"[ERROR] Failed to open {port}: {e}")
        print("[INFO] Check:")
        print("  1. STM32 is connected")
        print("  2. Correct port name")
        print("  3. No other program using port")
        print("  4. USB drivers installed")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None


# ========================================
# EXAMPLE: STM32 Parsing Code (C)
# ========================================
"""
// STM32 C code to parse incoming commands

void parse_uart_command(char* buffer) {
    if (buffer[0] == 'C') {
        // Servo command: "C,120,95"
        int pan, tilt;
        if (sscanf(buffer, "C,%d,%d", &pan, &tilt) == 2) {
            // Validate ranges
            pan = clamp(pan, PAN_MIN, PAN_MAX);
            tilt = clamp(tilt, TILT_MIN, TILT_MAX);
            
            // Update servo positions
            set_servo_pan(pan);
            set_servo_tilt(tilt);
        }
    }
    else if (buffer[0] == 'T') {
        // Telemetry - forward to BLE
        ble_transmit(buffer);
    }
}

// Helper function
int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}
"""
