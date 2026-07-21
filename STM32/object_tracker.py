import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time

# Servo protocol expects ASCII commands in the format:
# "C,<pan>,<tilt>\n" with angles clamped to 0-180 degrees.
# The STM32 firmware echoes the command or returns an error string.

class ObjectTracker:
    def __init__(self, model_path='yolo11n.pt', serial_port='COM3', baud_rate=115200,
                 pan_center=90.0, tilt_center=90.0, deg_per_pixel=0.05):
        """
        Initialize the object tracker with YOLO11n model
        
        Args:
            model_path: Path to YOLO11n model
            serial_port: Serial port for STM32 communication
            baud_rate: Serial communication baud rate
        """
        # Load YOLO11n model
        self.model = YOLO(model_path)
        
        # Initialize serial connection to STM32
        try:
            self.serial = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            print(f"Connected to STM32 on {serial_port}")
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            self.serial = None
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Frame center
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Tracking parameters
        self.target_class = 0  # 0 = person, can be changed
        self.confidence_threshold = 0.5

        # Servo state
        self.pan_angle = pan_center
        self.tilt_angle = tilt_center
        self.deg_per_pixel = deg_per_pixel  # proportional constant to convert pixel error to servo delta
        
        # Dead zone (pixels around center where no adjustment is needed)
        self.dead_zone_x = 50
        self.dead_zone_y = 50
        
    def calculate_error(self, bbox_center_x, bbox_center_y):
        """
        Calculate error from frame center
        
        Returns:
            error_x, error_y: Pixel errors from center
        """
        error_x = bbox_center_x - self.center_x
        error_y = bbox_center_y - self.center_y
        
        # Apply dead zone
        if abs(error_x) < self.dead_zone_x:
            error_x = 0
        if abs(error_y) < self.dead_zone_y:
            error_y = 0
            
        return error_x, error_y

    def update_angles_from_error(self, error_x, error_y):
        """
        Convert pixel error into servo angles and clamp to valid range.
        """
        delta_pan = -error_x * self.deg_per_pixel
        delta_tilt = error_y * self.deg_per_pixel

        self.pan_angle = np.clip(self.pan_angle + delta_pan, 0, 180)
        self.tilt_angle = np.clip(self.tilt_angle + delta_tilt, 0, 180)
    
    def send_to_stm32(self, pan_angle, tilt_angle):
        """
        Send pan/tilt angles to STM32 using the ASCII protocol.
        """
        if self.serial is None:
            return

        pan_angle = int(np.clip(pan_angle, 0, 180))
        tilt_angle = int(np.clip(tilt_angle, 0, 180))

        command = f"C,{pan_angle},{tilt_angle}\n"

        try:
            self.serial.write(command.encode("ascii"))
            # Attempt to read echo/ack but ignore timeouts for robustness
            if self.serial.in_waiting:
                response = self.serial.readline().decode(errors="ignore").strip()
                if response:
                    print(f"STM32: {response}")
        except Exception as e:
            print(f"Serial communication error: {e}")
    
    def track_objects(self):
        """
        Main tracking loop
        """
        print("Starting object tracking...")
        print(f"Target class: {self.target_class}")
        print(f"Frame size: {self.frame_width}x{self.frame_height}")
        print("Press 'q' to quit, 'c' to change target class")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run YOLO detection with tracking
            results = self.model.track(frame, persist=True, conf=self.confidence_threshold)
            
            # Draw frame center
            cv2.circle(frame, (self.center_x, self.center_y), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, 
                         (self.center_x - self.dead_zone_x, self.center_y - self.dead_zone_y),
                         (self.center_x + self.dead_zone_x, self.center_y + self.dead_zone_y),
                         (0, 255, 0), 2)
            
            # Process detections
            target_found = False
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                # Find target class with highest confidence
                best_box = None
                best_conf = 0
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == self.target_class and conf > best_conf:
                        best_conf = conf
                        best_box = box
                
                if best_box is not None:
                    target_found = True
                    
                    # Get bounding box
                    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                    
                    # Calculate center
                    bbox_center_x = int((x1 + x2) / 2)
                    bbox_center_y = int((y1 + y2) / 2)
                    
                    # Calculate error and update servo targets
                    error_x, error_y = self.calculate_error(bbox_center_x, bbox_center_y)
                    self.update_angles_from_error(error_x, error_y)

                    # Send to STM32
                    self.send_to_stm32(self.pan_angle, self.tilt_angle)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
                    
                    # Draw line from center to target
                    cv2.line(frame, (self.center_x, self.center_y), 
                            (bbox_center_x, bbox_center_y), (255, 0, 0), 2)
                    
                    # Display info
                    label = f"{self.model.names[self.target_class]} {best_conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Error X: {error_x}, Y: {error_y}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If no target found, keep servos steady at last value
            if not target_found:
                self.send_to_stm32(self.pan_angle, self.tilt_angle)
                cv2.putText(frame, "No target detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('YOLO11n Object Tracking', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Available classes:")
                for idx, name in self.model.names.items():
                    print(f"  {idx}: {name}")
                try:
                    new_class = int(input("Enter class ID: "))
                    if new_class in self.model.names:
                        self.target_class = new_class
                        print(f"Target changed to: {self.model.names[new_class]}")
                except:
                    print("Invalid input")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.serial:
            self.serial.close()

def main():
    # Configuration
    MODEL_PATH = 'yolo11n.pt'  # Path to YOLO11n model
    SERIAL_PORT = 'COM3'        # Change to your STM32 port (e.g., '/dev/ttyACM0' on Linux)
    BAUD_RATE = 115200
    
    tracker = ObjectTracker(MODEL_PATH, SERIAL_PORT, BAUD_RATE)
    tracker.track_objects()

if __name__ == "__main__":
    main()
