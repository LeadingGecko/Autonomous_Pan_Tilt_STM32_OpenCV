"""
========================================================================
Title: YOLO Object Detection and Tracking Model
Author: Giancarlo Passanante
========================================================================

Description:
------------
This script implements a YOLO (You Only Look Once) object detection and 
tracking system using OpenCV. It can detect multiple objects in real-time
and track their movements across frames.

Dependencies:
------------
- OpenCV (cv2)
- NumPy
- Time

Features:
---------
1. YOLO object detection
2. Real-time object tracking
3. Bounding box visualization
4. FPS calculation
5. Confidence threshold adjustment
"""

import cv2
import numpy as np
import time
import os

class YOLOTracker:
    """
    A class to handle YOLO object detection and tracking
    """
    
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4,
                 model_cfg: str = None, model_weights: str = None, names_path: str = None):
        """
        Initialize the YOLO tracker with model configurations
        
        Args:
            confidence_threshold (float): Minimum confidence for detection (0-1)
            nms_threshold (float): Non-maximum suppression threshold (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Determine base directory (script directory) to resolve relative paths
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Allow overriding paths via constructor args; otherwise use files in the same folder
        self.model_weights = model_weights if model_weights is not None else os.path.join(base_dir, "yolov4-tiny.weights")
        self.model_cfg = model_cfg if model_cfg is not None else os.path.join(base_dir, "yolov4-tiny.cfg")
        self.names_path = names_path if names_path is not None else os.path.join(base_dir, "coco.names")

        # Load YOLO model configurations
        self.net = None
        self.output_layers = None
        self.classes = []
        loaded = self.load_model()
        if not loaded:
            raise RuntimeError("Failed to load YOLO model. See earlier messages for details.")
        
    def load_model(self):
        """
        Load the YOLO model weights and configurations
        """
        # Load YOLO weights and cfg file (paths resolved in __init__)
        weights_path = self.model_weights
        config_path = self.model_cfg

        # Check that files exist and provide helpful messages
        missing = []
        if not os.path.exists(config_path):
            missing.append(config_path)
        if not os.path.exists(weights_path):
            missing.append(weights_path)
        if not os.path.exists(self.names_path):
            missing.append(self.names_path)
        if missing:
            print("Error loading YOLO model: the following required files were not found:")
            for p in missing:
                print(f"  - {p}")
            print("Place the model files in the same folder as this script, or pass explicit paths to YOLOTracker(model_cfg=..., model_weights=..., names_path=...)")
            return False

        try:
            # Use the Darknet reader explicitly: cfg then weights
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
            
        # Load class names
        try:
            with open(self.names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading class names: {e}")
            return False
            
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        return True
        
    def preprocess_image(self, frame):
        """
        Preprocess the image for YOLO model
        
        Args:
            frame (numpy.ndarray): Input frame from video/camera
            
        Returns:
            blob: Preprocessed image blob
        """
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                   swapRB=True, crop=False)
        return blob
        
    def detect_objects(self, frame):
        """
        Detect objects in the frame using YOLO
        
        Args:
            frame (numpy.ndarray): Input frame from video/camera
            
        Returns:
            list: List of detected objects with their properties
        """
        height, width = frame.shape[:2]
        
        # Preprocess image
        blob = self.preprocess_image(frame)
        self.net.setInput(blob)
        
        # Get detections
        outs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected objects
        class_ids = []
        confidences = []
        boxes = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 
                                 self.confidence_threshold, 
                                 self.nms_threshold)
        
        detections = []
        if len(indexes) > 0:
            indexes = indexes.flatten()
            for i in indexes:
                detection = {
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]]
                }
                detections.append(detection)
                
        return detections
        
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels for detected objects
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detected objects
            
        Returns:
            numpy.ndarray: Frame with drawn detections
        """
        for detection in detections:
            x, y, w, h = detection['box']
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
        return frame

    def process_video(self, source=0):
        """
        Process video stream and detect/track objects
        
        Args:
            source: Video source (0 for webcam, or video file path)
        """
        cap = cv2.VideoCapture(source)
        fps = 0
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("YOLO Object Tracking", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize tracker
    tracker = YOLOTracker(confidence_threshold=0.5, nms_threshold=0.4)
    
    # Start video processing (use 0 for webcam or provide video file path)
    tracker.process_video(0)