/**
 * @file servo_control.h
 * @brief Servo motor control via PWM
 */

#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H

#include <stdint.h>

/* ========================================
   SERVO CONFIGURATION
   ======================================== */

#define SERVO_MIN_ANGLE     0.0f
#define SERVO_MAX_ANGLE     180.0f
#define SERVO_CENTER_ANGLE  90.0f

#define SERVO_MIN_PULSE_US  1000   /* 1.0 ms @ 0° */
#define SERVO_MAX_PULSE_US  2000   /* 2.0 ms @ 180° */
#define SERVO_FREQ_HZ       50     /* Standard servo frequency */

/* ========================================
   FUNCTION DECLARATIONS
   ======================================== */

/**
 * Initialize servo PWM generation
 * Must be called once before using servo functions
 */
void servo_init(void);

/**
 * Set pan servo angle (horizontal)
 * @param angle: 0-180 degrees
 */
void servo_set_pan(float angle);

/**
 * Set tilt servo angle (vertical)
 * @param angle: 0-180 degrees
 */
void servo_set_tilt(float angle);

/**
 * Center both servos at 90 degrees
 */
void servo_center(void);

/**
 * Get current pan angle
 * @return: Pan angle in degrees
 */
float servo_get_pan(void);

/**
 * Get current tilt angle
 * @return: Tilt angle in degrees
 */
float servo_get_tilt(void);

#endif /* SERVO_CONTROL_H */
