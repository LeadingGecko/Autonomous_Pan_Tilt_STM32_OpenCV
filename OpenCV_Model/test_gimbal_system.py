# test_gimbal_system.py
import math
import numpy as np
import time

from servo_control import compute_servo_targets_from_telemetry, PIDController  # :contentReference[oaicite:1]{index=1}
from telemetry_extract import extract_person_telemetry                          # :contentReference[oaicite:2]{index=2}
from uart_communication import encode_telemetry_uart                            # :contentReference[oaicite:3]{index=3}


# ─────────────────────────────────────────────
# Helpers for fake YOLO result
# ─────────────────────────────────────────────

class DummyXYXY:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class DummyBoxes:
    def __init__(self, xyxy_arr, conf_val):
        # xyxy_arr: [x1, y1, x2, y2]
        self.xyxy = [DummyXYXY(xyxy_arr)]
        self.conf = [conf_val]


class DummyResult:
    def __init__(self, boxes):
        self.boxes = boxes


# ─────────────────────────────────────────────
# Tests for compute_servo_targets_from_telemetry
# ─────────────────────────────────────────────

def test_servo_holds_position_when_not_tracking():
    frame_shape = (480, 640, 3)
    pan, tilt = 90.0, 90.0
    telemetry = {
        "tracking": 0,   # not tracking
        "cx": -1,
        "cy": -1,
    }

    new_pan, new_tilt = compute_servo_targets_from_telemetry(
        telemetry, frame_shape, pan, tilt
    )

    assert new_pan == pan
    assert new_tilt == tilt


def test_servo_moves_towards_right_when_person_on_right():
    frame_shape = (480, 640, 3)
    pan, tilt = 90.0, 90.0
    center_x = frame_shape[1] // 2

    telemetry = {
        "tracking": 1,
        "cx": center_x + 50,  # person to the right of center
        "cy": frame_shape[0] // 2,
    }

    new_pan, _ = compute_servo_targets_from_telemetry(
        telemetry, frame_shape, pan, tilt
    )

    # if person is right, pan angle should increase
    assert new_pan > pan


def test_servo_moves_towards_left_when_person_on_left():
    frame_shape = (480, 640, 3)
    pan, tilt = 90.0, 90.0
    center_x = frame_shape[1] // 2

    telemetry = {
        "tracking": 1,
        "cx": center_x - 50,  # person to the left of center
        "cy": frame_shape[0] // 2,
    }

    new_pan, _ = compute_servo_targets_from_telemetry(
        telemetry, frame_shape, pan, tilt
    )

    # if person is left, pan angle should decrease
    assert new_pan < pan


def test_servo_output_clamped_to_valid_range():
    frame_shape = (480, 640, 3)
    pan, tilt = 180.0, 0.0  # extremes

    telemetry = {
        "tracking": 1,
        "cx": frame_shape[1] * 2,   # very far right
        "cy": -frame_shape[0],      # very far up
    }

    new_pan, new_tilt = compute_servo_targets_from_telemetry(
        telemetry, frame_shape, pan, tilt
    )

    assert 0.0 <= new_pan <= 180.0
    assert 0.0 <= new_tilt <= 180.0


# ─────────────────────────────────────────────
# Tests for PIDController
# ─────────────────────────────────────────────

def test_pid_pure_proportional():
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0,
                        output_min=-100.0, output_max=100.0)

    out = pid.compute(error=10.0, dt=1.0)
    assert out == 10.0      # kp * error
    assert pid.prev_error == 10.0


def test_pid_with_integral_and_derivative():
    pid = PIDController(kp=0.5, ki=0.1, kd=0.0,
                        output_min=-100.0, output_max=100.0)

    # First call
    out1 = pid.compute(error=10.0, dt=1.0)
    # second call, same error → integral accumulates
    out2 = pid.compute(error=10.0, dt=1.0)

    assert out2 > out1  # integral term should increase output


# ─────────────────────────────────────────────
# Tests for extract_person_telemetry
# ─────────────────────────────────────────────

def test_extract_person_telemetry_no_detections():
    frame_shape = (480, 640, 3)
    result = DummyResult(boxes=None)

    telemetry = extract_person_telemetry(
        result=result,
        frame_shape=frame_shape,
        mode="AUTO",
        servo_pan=90.0,
        servo_tilt=90.0,
        fps=20.0,
    )

    assert telemetry["tracking"] == 0
    assert telemetry["cx"] == -1
    assert telemetry["cy"] == -1


def test_extract_person_telemetry_with_detection():
    frame_shape = (480, 640, 3)
    # bounding box from (100, 50) to (300, 250)
    boxes = DummyBoxes([100, 50, 300, 250], conf_val=0.9)
    result = DummyResult(boxes=boxes)

    telemetry = extract_person_telemetry(
        result=result,
        frame_shape=frame_shape,
        mode="AUTO",
        servo_pan=90.0,
        servo_tilt=90.0,
        fps=15.0,
    )

    assert telemetry["tracking"] == 1
    # center of [100,50,300,250] is (200,150)
    assert telemetry["cx"] == 200
    assert telemetry["cy"] == 150
    assert telemetry["bw"] == 200
    assert telemetry["bh"] == 200
    assert abs(telemetry["conf"] - 0.9) < 1e-6


# ─────────────────────────────────────────────
# Tests for encode_telemetry_uart
# ─────────────────────────────────────────────

def test_encode_telemetry_uart_format():
    t = {
        "mode": "AUTO",
        "tracking": 1,
        "cx": 320,
        "cy": 240,
        "bw": 100,
        "bh": 200,
        "conf": 0.9234,
        "servo_pan": 123.456,
        "servo_tilt": 78.9,
        "fps": 12.3456,
    }

    line = encode_telemetry_uart(t)
    # Expect: "T,AUTO,1,320,240,100,200,0.92,123.5,78.9,12.35\n"
    assert line.startswith("T,AUTO,1,320,240,100,200,")
    assert line.endswith("\n")

    parts = line.strip().split(",")
    assert parts[0] == "T"
    assert parts[1] == "AUTO"
    assert parts[2] == "1"
    assert parts[3] == "320"
    assert parts[4] == "240"
    assert parts[7] == "0.92"  # two decimal places
