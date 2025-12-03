/* main.c - STM32 Nucleo-L053R8 Pan-Tilt Servo Controller
 * 
 * Hardware Configuration:
 * - USART2 (PA2/PA3): Serial communication with PC
 * - TIM2_CH1 (PA0): Pan servo PWM output
 * - TIM2_CH2 (PA1): Tilt servo PWM output
 * - TIM21: PID control loop timer
 * 
 * Author: Generated for STM32L053R8
 * Date: 2024
 */

#include "stm32l0xx_hal.h"
#include <string.h>
#include <stdlib.h>

/* Private defines */
#define SERVO_PWM_FREQ      50      // 50Hz for servo
#define SERVO_MIN_PULSE     500     // 0.5ms (0 degrees)
#define SERVO_MAX_PULSE     2500    // 2.5ms (180 degrees)
#define SERVO_CENTER_PULSE  1500    // 1.5ms (90 degrees)

#define PAN_MIN_ANGLE       0
#define PAN_MAX_ANGLE       180
#define TILT_MIN_ANGLE      0
#define TILT_MAX_ANGLE      180

#define UART_RX_BUFFER_SIZE 128
#define PACKET_SIZE         6

/* PID Controller parameters */
#define KP_PAN              0.08f
#define KD_PAN              0.02f
#define KP_TILT             0.08f
#define KD_TILT             0.02f

#define MAX_ERROR_ACCUM     1000

/* Private variables */
UART_HandleTypeDef huart2;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim21;

uint8_t uart_rx_buffer[UART_RX_BUFFER_SIZE];
uint8_t uart_rx_index = 0;

/* Servo positions (in microseconds) */
volatile uint16_t pan_pulse = SERVO_CENTER_PULSE;
volatile uint16_t tilt_pulse = SERVO_CENTER_PULSE;

/* PID variables */
volatile int16_t error_x = 0;
volatile int16_t error_y = 0;
volatile int16_t prev_error_x = 0;
volatile int16_t prev_error_y = 0;

/* Function prototypes */
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM21_Init(void);
void Error_Handler(void);
void process_packet(uint8_t* packet);
void update_servos(void);
uint16_t calculate_pid_pan(int16_t error, int16_t prev_error);
uint16_t calculate_pid_tilt(int16_t error, int16_t prev_error);
uint16_t constrain_pulse(uint16_t pulse);

int main(void)
{
    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* Configure the system clock */
    SystemClock_Config();

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_USART2_UART_Init();
    MX_TIM2_Init();
    MX_TIM21_Init();

    /* Start PWM for servos */
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);  // Pan servo
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);  // Tilt servo

    /* Set servos to center position */
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pan_pulse);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, tilt_pulse);

    /* Start control timer */
    HAL_TIM_Base_Start_IT(&htim21);

    /* Enable UART receive interrupt */
    HAL_UART_Receive_IT(&huart2, uart_rx_buffer, 1);

    /* Main loop */
    while (1)
    {
        /* Control is handled in timer interrupt */
        HAL_Delay(10);
    }
}

/**
  * @brief System Clock Configuration
  */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    /* Configure the main internal regulator output voltage */
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    /* Initializes the CPU, AHB and APB busses clocks */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLMUL = RCC_PLLMUL_4;
    RCC_OscInitStruct.PLL.PLLDIV = RCC_PLLDIV_2;
    
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }

    /* Initializes the CPU, AHB and APB busses clocks */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief USART2 Initialization Function
  */
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
    
    if (HAL_UART_Init(&huart2) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief TIM2 Initialization Function (PWM for servos)
  * Timer configured for 50Hz PWM (20ms period)
  */
static void MX_TIM2_Init(void)
{
    TIM_OC_InitTypeDef sConfigOC = {0};
    TIM_MasterConfigTypeDef sMasterConfig = {0};

    /* TIM2 clock = 32MHz, prescaler = 32-1, ARR = 20000-1 gives 50Hz */
    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 31;  // 32MHz / 32 = 1MHz
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 19999;  // 1MHz / 20000 = 50Hz
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    
    if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
    {
        Error_Handler();
    }

    sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    
    if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
    {
        Error_Handler();
    }

    /* Configure PWM channel 1 (Pan) */
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = SERVO_CENTER_PULSE;
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
    
    if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
    {
        Error_Handler();
    }

    /* Configure PWM channel 2 (Tilt) */
    if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief TIM21 Initialization Function (Control loop timer - 20Hz)
  */
static void MX_TIM21_Init(void)
{
    /* Timer for 20Hz control loop */
    htim21.Instance = TIM21;
    htim21.Init.Prescaler = 31999;  // 32MHz / 32000 = 1kHz
    htim21.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim21.Init.Period = 49;        // 1kHz / 50 = 20Hz
    htim21.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    
    if (HAL_TIM_Base_Init(&htim21) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief GPIO Initialization Function
  */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* Configure GPIO pin : LED (PA5) */
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

/**
  * @brief Calculate PID for pan servo
  */
uint16_t calculate_pid_pan(int16_t error, int16_t prev_error)
{
    float p_term = KP_PAN * error;
    float d_term = KD_PAN * (error - prev_error);
    
    int16_t adjustment = (int16_t)(p_term + d_term);
    
    int16_t new_pulse = pan_pulse - adjustment;
    
    return constrain_pulse(new_pulse);
}

/**
  * @brief Calculate PID for tilt servo
  */
uint16_t calculate_pid_tilt(int16_t error, int16_t prev_error)
{
    float p_term = KP_TILT * error;
    float d_term = KD_TILT * (error - prev_error);
    
    int16_t adjustment = (int16_t)(p_term + d_term);
    
    int16_t new_pulse = tilt_pulse + adjustment;  // Note: inverted for tilt
    
    return constrain_pulse(new_pulse);
}

/**
  * @brief Constrain pulse width to valid range
  */
uint16_t constrain_pulse(uint16_t pulse)
{
    if (pulse < SERVO_MIN_PULSE)
        return SERVO_MIN_PULSE;
    if (pulse > SERVO_MAX_PULSE)
        return SERVO_MAX_PULSE;
    return pulse;
}

/**
  * @brief Update servo positions based on PID control
  */
void update_servos(void)
{
    /* Calculate new pulse widths */
    uint16_t new_pan_pulse = calculate_pid_pan(error_x, prev_error_x);
    uint16_t new_tilt_pulse = calculate_pid_tilt(error_y, prev_error_y);
    
    /* Update pulse widths */
    pan_pulse = new_pan_pulse;
    tilt_pulse = new_tilt_pulse;
    
    /* Apply to PWM */
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pan_pulse);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, tilt_pulse);
    
    /* Store previous errors */
    prev_error_x = error_x;
    prev_error_y = error_y;
}

/**
  * @brief Process received packet from UART
  * Packet format: <0xAA><X_HIGH><X_LOW><Y_HIGH><Y_LOW><0x55>
  */
void process_packet(uint8_t* packet)
{
    if (packet[0] == 0xAA && packet[5] == 0x55)
    {
        /* Extract error values */
        uint16_t error_x_u = (packet[1] << 8) | packet[2];
        uint16_t error_y_u = (packet[3] << 8) | packet[4];
        
        /* Convert to signed */
        error_x = (int16_t)error_x_u;
        error_y = (int16_t)error_y_u;
        
        /* Toggle LED to indicate packet received */
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
    }
}

/**
  * @brief UART receive callback
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        static uint8_t packet_buffer[PACKET_SIZE];
        static uint8_t packet_index = 0;
        
        uint8_t received_byte = uart_rx_buffer[0];
        
        /* State machine for packet reception */
        if (packet_index == 0 && received_byte == 0xAA)
        {
            /* Start of packet */
            packet_buffer[packet_index++] = received_byte;
        }
        else if (packet_index > 0 && packet_index < PACKET_SIZE)
        {
            /* Middle of packet */
            packet_buffer[packet_index++] = received_byte;
            
            if (packet_index == PACKET_SIZE)
            {
                /* Complete packet received */
                process_packet(packet_buffer);
                packet_index = 0;
            }
        }
        else
        {
            /* Invalid state, reset */
            packet_index = 0;
        }
        
        /* Re-enable receive interrupt */
        HAL_UART_Receive_IT(&huart2, uart_rx_buffer, 1);
    }
}

/**
  * @brief Timer interrupt callback (Control loop)
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM21)
    {
        /* Update servos at 20Hz */
        update_servos();
    }
}

/**
  * @brief Error Handler
  */
void Error_Handler(void)
{
    /* Blink LED rapidly to indicate error */
    while (1)
    {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
        HAL_Delay(100);
    }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    /* User can add implementation to report error */
}
#endif
