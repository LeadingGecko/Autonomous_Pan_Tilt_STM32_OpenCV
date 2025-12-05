/**
 * @file uart_comm.h
 * @brief UART communication interface for YOLO servo commands
 *
 * PROTOCOL:
 *   C,<pan>,<tilt>    - Servo command (e.g., C,120,95)
 *   T,<mode>,<track>  - Telemetry from YOLO
 *
 * UART: USART2 @ 115200 baud
 * STM32 NUCLEO-L073:
 *   - TX: PA2 (to ST-LINK)
 *   - RX: PA3 (from ST-LINK)
 *   - USB virtual COM port (auto-detected by Windows)
 */

#ifndef UART_COMM_H
#define UART_COMM_H

#include <stdint.h>

/* ========================================
   CONFIGURATION
   ======================================== */

#define UART_BAUD_RATE      115200
#define UART_RX_BUFFER_SIZE 256
#define CMD_TIMEOUT_MS      5000

/* Simple RX buffer type used by uart_comm implementation */
typedef struct {
    char buffer[UART_RX_BUFFER_SIZE];
    uint16_t index;
} UartRxBuffer;

/* ========================================
   CALLBACK DEFINITION

   User must provide this function to handle
   parsed servo commands
   ======================================== */

/**
 * Servo command callback
 * Called when valid "C,pan,tilt" command is received
 *
 * @param pan: Pan angle (0-180)
 * @param tilt: Tilt angle (0-180)
 */
typedef void (*uart_servo_callback_t)(float pan, float tilt);

/**
 * Telemetry callback (optional)
 * Called when telemetry data is received
 *
 * @param tracking: 1 if person detected, 0 otherwise
 * @param pan: Computed pan angle from YOLO
 * @param tilt: Computed tilt angle from YOLO
 * @param fps: Processing framerate
 */
typedef void (*uart_telemetry_callback_t)(uint8_t tracking, float pan, float tilt, float fps);

/* ========================================
   FUNCTION DECLARATIONS
   ======================================== */

/**
 * Initialize UART communication
 * Must be called before any UART operations
 */
void uart_init(void);

/**
 * Send string over UART
 * @param str: Null-terminated string
 */
void uart_send_string(const char* str);

/**
 * Send formatted string over UART
 * @param fmt: printf-style format string
 */
void uart_printf(const char* fmt, ...);

/**
 * Register servo command callback
 * @param callback: Function to call on servo command
 */
void uart_set_servo_callback(uart_servo_callback_t callback);

/**
 * Register telemetry callback (optional)
 * @param callback: Function to call on telemetry
 */
void uart_set_telemetry_callback(uart_telemetry_callback_t callback);

/**
 * Check if UART is connected (for debugging)
 * @return: 1 if connected, 0 otherwise
 */
uint8_t uart_is_connected(void);

/**
 * Get last command timestamp
 * @return: HAL tick time of last command
 */
uint32_t uart_get_last_command_time(void);

#endif /* UART_COMM_H */
