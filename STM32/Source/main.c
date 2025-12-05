/**
 * ========================================
 * STM32L0 SERVO CONTROL SYSTEM
 * ========================================
 *
 * ARCHITECTURE OVERVIEW:
 *
 * [LAPTOP PC]                 [STM32 NUCLEO-L073]          [SERVO MOTORS]
 *     |                              |                           |
 *     | Python: YOLO detection       |                           |
 *     | Computes servo angles        |                           |
 *     |                              |                           |
 *     +--UART over Virtual COM6----->+ USART2 (PA2 TX, PA3 RX)   |
 *                                    |                           |
 *                            Parses commands (C,pan,tilt)        |
 *                            Updates servo targets               |
 *                                    |                           |
 *                                    +--PWM on TIM2 CH1/CH2----->+ Pan Servo (PA0)
 *                                    |                      +---->+ Tilt Servo (PA1)
 *
 * ========================================
 * KEY POINTS:
 *
 * 1. NUCLEO BOARDS HAVE STLINK BRIDGE:
 *    - The onboard ST-LINK MCU bridges UART2 to the USB port
 *    - When you connect the Nucleo board via USB, Windows automatically
 *      assigns a virtual COM port (e.g., COM6, COM7, COM8)
 *    - NO external USB-to-UART converter needed!
 *    - This virtual COM port connects to UART2 pins (PA2, PA3)
 *
 * 2. COMMUNICATION FLOW:
 *    - Laptop Python code sends: "C,120,95\n"
 *    - USB → Virtual COM port (COM6) → ST-LINK → UART2 → STM32
 *    - STM32 receives via UART2 interrupt
 *    - Parses command and updates PWM outputs
 *    - Servos move to pan=120°, tilt=95°
 *
 * 3. WHY SERVOS NOT MOVING:
 *    - UART data not being received (check connection)
 *    - Parsing failed (check format)
 *    - PWM not initialized (check TIM2 setup)
 *    - Wrong pins or timer (verify PA0, PA1, TIM2)
 *
 * ========================================
 * HARDWARE CONNECTIONS:
 *
 * STM32 NUCLEO-L073 Board:
 *   Pin PA0 ──────> TIM2_CH1 ──> Pan Servo Signal
 *   Pin PA1 ──────> TIM2_CH2 ──> Tilt Servo Signal
 *   GND     ──────> Servo GND
 *   5V(*)   ──────> Servo VCC (via separate power supply!)
 *
 *   (*) IMPORTANT: Use external servo power supply!
 *       DO NOT power servos from STM32 board (insufficient current)
 *       Common: GND to servos and board together
 *
 * ========================================
 */

#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================
   SERVO CONFIGURATION
   ======================================== */

/* PWM Timer Parameters */
#define SERVO_TIMER            TIM2
#define SERVO_TIMER_FREQ       1000000   /* 1 MHz timer clock */
#define SERVO_PWM_FREQ         50        /* 50 Hz (20 ms period) */

/* Servo Pulse Range (microseconds) */
#define SERVO_MIN_PULSE        1000      /* 1.0 ms  = 0° */
#define SERVO_CENTER_PULSE     1500      /* 1.5 ms  = 90° */
#define SERVO_MAX_PULSE        2000      /* 2.0 ms  = 180° */

/* Channel assignments */
#define PAN_CHANNEL            TIM_CHANNEL_1   /* PA0 */
#define TILT_CHANNEL           TIM_CHANNEL_2   /* PA1 */

/* ========================================
   UART PROTOCOL CONFIGURATION
   ======================================== */

#define UART_BAUD_RATE         115200
#define RX_BUFFER_SIZE         256
#define CMD_TIMEOUT_MS         5000

/* Protocol message types */
#define MSG_SERVO_CMD          'C'   /* C,pan,tilt */
#define MSG_TELEMETRY          'T'   /* T,mode,tracking,... */
#define MSG_STATUS             'S'   /* STATUS,MODE,AUTO/MANUAL */

/* ========================================
   DATA STRUCTURES
   ======================================== */

typedef struct {
    char buffer[RX_BUFFER_SIZE];
    uint16_t index;
} UartRxBuffer;

typedef struct {
    float pan_angle;      /* 0-180 degrees */
    float tilt_angle;     /* 0-180 degrees */
    uint8_t valid;        /* 1 if new command received */
    uint32_t timestamp;   /* Time of last update */
} ServoCommand;

/* ========================================
   GLOBAL STATE
   ======================================== */

/* Peripheral handle definitions (one definition required for linker) */
TIM_HandleTypeDef htim2;
UART_HandleTypeDef huart2;

static ServoCommand servo_cmd = {90.0f, 90.0f, 0, 0};
static uint8_t system_mode = 0;  /* 0=AUTO, 1=MANUAL */

/* ========================================
   FUNCTION DECLARATIONS
   ======================================== */

/* Servo control - use dedicated module */
#include "../Includes/servo_control.h"

/* UART communication - use dedicated module */
#include "../Includes/uart_comm.h"

/* Helpers */
static uint32_t angle_to_pwm_ticks(float angle);

/* ISR callbacks are implemented in uart_comm.c */

/* ========================================
    MAIN ENTRY POINT
    ======================================== */
static void SystemClock_Config();
static void MX_GPIO_Init();
static void MX_USART2_UART_Init();
static void MX_TIM2_Init();

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART2_UART_Init();
    MX_TIM2_Init();

    /* Initialize subsystems */
    servo_init();
    servo_center();

    /* Initialize UART module (starts receive IRQ) */
    uart_init();
    uart_set_servo_callback(on_servo_command);
    uart_send_string("\r\n[SYSTEM] STM32 Servo Controller Started\r\n");
    uart_send_string("[INFO] Listening for YOLO commands on UART2 (115200 baud)\r\n");
    uart_send_string("[INFO] Send format: C,<pan>,<tilt>  (e.g., C,120,95)\r\n\r\n");

    uint32_t last_heartbeat = HAL_GetTick();

    /* ========================================
       MAIN CONTROL LOOP
       ======================================== */
    while (1)
    {
        uint32_t now = HAL_GetTick();

        /* Check for command timeout */
        if (system_mode == 0) {  /* AUTO mode */
            uint32_t elapsed = now - servo_cmd.timestamp;
            if (elapsed > CMD_TIMEOUT_MS && servo_cmd.valid) {
                uart_send_string("[TIMEOUT] No command received, centering servos\r\n");
                servo_center();
                servo_cmd.valid = 0;
            }
        }

        /* Apply latest servo command */
        if (servo_cmd.valid) {
            servo_set_pan(servo_cmd.pan_angle);
            servo_set_tilt(servo_cmd.tilt_angle);
            servo_cmd.valid = 0;
        }

        /* Heartbeat LED (blink every 500ms) */
        if (now - last_heartbeat > 500) {
            HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
            last_heartbeat = now;
        }

        HAL_Delay(10);
    }
}

/* Servo control implementation moved to servo_control.c */

/* UART implementation moved to `uart_comm.c` */

/* Servo command callback: called by the UART module when a valid C,pan,tilt command arrives */
static void on_servo_command(float pan, float tilt)
{
    servo_cmd.pan_angle = pan;
    servo_cmd.tilt_angle = tilt;
    servo_cmd.valid = 1;
    servo_cmd.timestamp = HAL_GetTick();

    /* Log receipt */
    char log[80];
    snprintf(log, sizeof(log), "[CMD_CB] Pan: %d, Tilt: %d\r\n", (int)pan, (int)tilt);
    uart_send_string(log);
}

/* Register main's servo callback with UART module */
/* (registration performed after uart_init() in main) */

/* ========================================
   MCU INITIALIZATION (from CubeMX)
   ======================================== */

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
    RCC_OscInitStruct.MSIState = RCC_MSI_ON;
    RCC_OscInitStruct.MSICalibrationValue = 0;
    RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_5;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                  RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_MSI;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK) Error_Handler();

    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART2;
    PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK) Error_Handler();
}

static void MX_TIM2_Init(void)
{
    TIM_MasterConfigTypeDef sMasterConfig = {0};
    TIM_OC_InitTypeDef sConfigOC = {0};

    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 0;
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 19999;  /* 20 ms @ 1 MHz = 50 Hz */
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_PWM_Init(&htim2) != HAL_OK) Error_Handler();

    sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
        Error_Handler();

    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 1500;  /* Start at 90° (center) */
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

    if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
        Error_Handler();
    if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
        Error_Handler();

    HAL_TIM_MspPostInit(&htim2);
}

static void MX_USART2_UART_Init(void)
{
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) Error_Handler();
}

static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

    GPIO_InitStruct.Pin = LD2_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);
}

void Error_Handler(void)
{
    __disable_irq();
    while (1) {
        HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
        HAL_Delay(100);
    }
}
