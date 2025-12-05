/**
 * @file servo_control.c
 * @brief Servo motor control implementation
 *
 * HARDWARE MAPPING:
 *   TIM2_CH1 (PA0)  → Pan servo
 *   TIM2_CH2 (PA1)  → Tilt servo
 *
 * PWM CHARACTERISTICS:
 *   Frequency: 50 Hz (20 ms period)
 *   Timer Clock: 1 MHz (1 µs resolution)
 *   Pulse Range: 1000-2000 µs (0-180°)
 */

#include "main.h"
#include "servo_control.h"
#include <math.h>

/* ========================================
   HARDWARE ABSTRACTIONS
   ======================================== */

extern TIM_HandleTypeDef htim2;

#define SERVO_TIMER        (&htim2)
#define PAN_CHANNEL        TIM_CHANNEL_1
#define TILT_CHANNEL       TIM_CHANNEL_2
#define TIMER_FREQ_HZ      1000000  /* 1 MHz */

/* ========================================
   STATIC STATE
   ======================================== */

static float g_pan_angle = SERVO_CENTER_ANGLE;
static float g_tilt_angle = SERVO_CENTER_ANGLE;
static uint8_t g_initialized = 0;

/* ========================================
   INTERNAL FUNCTIONS
   ======================================== */

/**
 * Clamp angle to valid range [0, 180]
 */
static float clamp_angle(float angle)
{
    if (angle < SERVO_MIN_ANGLE) return SERVO_MIN_ANGLE;
    if (angle > SERVO_MAX_ANGLE) return SERVO_MAX_ANGLE;
    return angle;
}

/**
 * Convert angle (0-180°) to PWM pulse width in timer ticks
 *
 * Linear mapping:
 *   0°   → 1000 µs
 *   90°  → 1500 µs
 *   180° → 2000 µs
 *
 * With 1 MHz timer: 1 tick = 1 µs
 */
static uint32_t angle_to_pulse_ticks(float angle)
{
    angle = clamp_angle(angle);

    /* Linear interpolation */
    float pulse_range = SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US;
    float normalized = angle / SERVO_MAX_ANGLE;
    float pulse_us = SERVO_MIN_PULSE_US + (normalized * pulse_range);

    return (uint32_t)pulse_us;
}

/* ========================================
   PUBLIC API
   ======================================== */

void servo_init(void)
{
    if (g_initialized) return;

    /* TIM2 already initialized by CubeMX */
    /* Start PWM on both channels */
    HAL_TIM_PWM_Start(SERVO_TIMER, PAN_CHANNEL);
    HAL_TIM_PWM_Start(SERVO_TIMER, TILT_CHANNEL);

    /* Initialize to center position */
    servo_center();

    g_initialized = 1;
}

void servo_set_pan(float angle)
{
    angle = clamp_angle(angle);
    uint32_t pulse_ticks = angle_to_pulse_ticks(angle);

    __HAL_TIM_SET_COMPARE(SERVO_TIMER, PAN_CHANNEL, pulse_ticks);

    g_pan_angle = angle;
}

void servo_set_tilt(float angle)
{
    angle = clamp_angle(angle);
    uint32_t pulse_ticks = angle_to_pulse_ticks(angle);

    __HAL_TIM_SET_COMPARE(SERVO_TIMER, TILT_CHANNEL, pulse_ticks);

    g_tilt_angle = angle;
}

void servo_center(void)
{
    servo_set_pan(SERVO_CENTER_ANGLE);
    servo_set_tilt(SERVO_CENTER_ANGLE);
}

float servo_get_pan(void)
{
    return g_pan_angle;
}

float servo_get_tilt(void)
{
    return g_tilt_angle;
}
