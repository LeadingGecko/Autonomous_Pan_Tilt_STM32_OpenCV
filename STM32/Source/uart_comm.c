#include "main.h"
#include "../Includes/uart_comm.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* Local RX buffer */
static UartRxBuffer uart_rx = {0};

/* Callback for parsed servo commands */
static uart_servo_callback_t servo_callback = NULL;

/* Last command timestamp (HAL ticks) */
static uint32_t last_command_time = 0;

void uart_init(void)
{
    /* Start interrupt-based receive for first byte */
    HAL_UART_Receive_IT(&huart2, (uint8_t*)&uart_rx.buffer[uart_rx.index], 1);
}

void uart_send_string(const char* str)
{
    if (!str) return;
    HAL_UART_Transmit(&huart2, (uint8_t*)str, strlen(str), 100);
}

void uart_printf(const char* fmt, ...)
{
    char buf[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    uart_send_string(buf);
}

void uart_set_servo_callback(uart_servo_callback_t callback)
{
    servo_callback = callback;
}

uint8_t uart_is_connected(void)
{
    /* Basic heuristic: if UART instance is initialized, return 1 */
    return (huart2.Instance != NULL) ? 1 : 0;
}

uint32_t uart_get_last_command_time(void)
{
    return last_command_time;
}

/* Internal: parse a completed line */
static void uart_handle_line(char* line)
{
    if (!line || strlen(line) < 2) return;

    /* Trim trailing whitespace */
    char* end = line + strlen(line) - 1;
    while (end >= line && (*end == '\r' || *end == '\n' || *end == ' ')) {
        *end-- = 0;
    }

    if (line[0] == 'C') {
        int pan, tilt;
        if (sscanf(line, "C,%d,%d", &pan, &tilt) == 2) {
            if (pan >= 0 && pan <= 180 && tilt >= 0 && tilt <= 180) {
                last_command_time = HAL_GetTick();
                if (servo_callback) {
                    servo_callback((float)pan, (float)tilt);
                }
                char log[80];
                snprintf(log, sizeof(log), "[CMD] Pan: %d, Tilt: %d\r\n", pan, tilt);
                uart_send_string(log);
                return;
            } else {
                uart_send_string("[ERROR] Servo angles out of range (0-180)\r\n");
                return;
            }
        }
        uart_send_string("[ERROR] Invalid command format. Use: C,pan,tilt\r\n");
    }
    else if (line[0] == 'T') {
        /* Telemetry messages are ignored at MCU side for now */
        uart_send_string("[INFO] Telemetry received\r\n");
    } else {
        /* Echo unknown */
        char log[128];
        snprintf(log, sizeof(log), "[ECHO] %s\r\n", line);
        uart_send_string(log);
    }
}

/**
 * HAL UART receive callback - buffer characters until newline
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart != &huart2) return;

    char c = uart_rx.buffer[uart_rx.index];

    if (c == '\n' || c == '\r') {
        if (uart_rx.index > 0) {
            uart_rx.buffer[uart_rx.index] = 0;
            uart_handle_line(uart_rx.buffer);
            uart_rx.index = 0;
        }
    } else if (uart_rx.index >= UART_RX_BUFFER_SIZE - 1) {
        uart_send_string("[ERROR] RX buffer overflow\r\n");
        uart_rx.index = 0;
    } else {
        uart_rx.buffer[uart_rx.index++] = c;
    }

    /* Continue receiving next byte */
    HAL_UART_Receive_IT(&huart2, (uint8_t*)&uart_rx.buffer[uart_rx.index], 1);
}
