## STM32 pan/tilt debugging

The STM32 firmware expects ASCII commands on USART2 using the format:

```
C,<pan>,<tilt>\n
```

Angles must be between 0–180 degrees. The board echoes the command or an
error string when parsing fails.

### Quick host-side serial test

Use the helper script to confirm that commands can be transmitted and
parsed before running the full OpenCV pipeline:

```
python debug_serial_test.py --port COM6 --pan 120 --tilt 80
python debug_serial_test.py --port /dev/ttyACM0 --sweep
```

### YOLO object tracker

`object_tracker.py` now converts pixel error into servo angles and sends
ASCII commands that match the firmware parser. Adjust `deg_per_pixel` in
the constructor to tune how aggressively the servos move in response to
object displacement.

