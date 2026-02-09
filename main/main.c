#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "vl53l5cx_api.h"
#include "kalman.h"
#include "ai_inference.h"

//------------------- Compilation Flags ----------------------/
#define IS_ENVELOPE_ENABLED     true
#define IS_KALMAN_ENABLED       true
#define PRINT_MATRIX_ENABLED    false

//----------------------- Constants --------------------------/
#define KALMAN_INITIAL_VALUE    400
#define MATRIX_PIXEL_SIZE       64
#define ENVELOPE_FILTER_VALUE   400

//----------------------- Variables --------------------------/
#if IS_KALMAN_ENABLED
static KALMAN_STRUCT_T kalman[MATRIX_PIXEL_SIZE];
#endif

static uint16_t frame[8][8];

//------------------ Main --------------------/
void app_main(void)
{
    // -------------------------------
    // I2C init
    // -------------------------------
    i2c_port_t i2c_port = I2C_NUM_0;
    i2c_config_t i2c_config = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = 5,
        .scl_io_num = 4,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = VL53L5CX_MAX_CLK_SPEED,
    };

    i2c_param_config(i2c_port, &i2c_config);
    i2c_driver_install(i2c_port, i2c_config.mode, 0, 0, 0);

    // -------------------------------
    // VL53L5CX
    // -------------------------------
    uint8_t status, isAlive, isReady;
    VL53L5CX_Configuration Dev;
    VL53L5CX_ResultsData Results;

    Dev.platform.address = VL53L5CX_DEFAULT_I2C_ADDRESS;
    Dev.platform.port = i2c_port;

    status = vl53l5cx_is_alive(&Dev, &isAlive);
    if (!isAlive || status)
    {
        printf("VL53L5CX not detected at requested address\n");
        return;
    }

    status = vl53l5cx_init(&Dev);
    if (status)
    {
        printf("VL53L5CX ULD Loading failed\n");
        return;
    }

    printf("VL53L5CX ULD ready ! (Version : %s)\n", VL53L5CX_API_REVISION);

    vl53l5cx_set_resolution(&Dev, VL53L5CX_RESOLUTION_8X8);
    vl53l5cx_set_ranging_frequency_hz(&Dev, 5);
    vl53l5cx_set_ranging_mode(&Dev, VL53L5CX_RANGING_MODE_CONTINUOUS);
    vl53l5cx_set_sharpener_percent(&Dev, 20);

    status = vl53l5cx_start_ranging(&Dev);

#if IS_KALMAN_ENABLED
    for (int i = 0; i < MATRIX_PIXEL_SIZE; i++)
    {
        Kalman_Initialize(&kalman[i], KALMAN_INITIAL_VALUE);
    }
#endif

    // -------------------------------
    // AI init
    // -------------------------------
    bool ai_ok = false;

    ai_ok = ai_init();

    if (ai_ok)
        printf("AI initialized successfully\n");
    else
        printf("AI init failed, running without AI\n");

    // -------------------------------
    // Main loop
    // -------------------------------
    while (1)
    {
        status = vl53l5cx_check_data_ready(&Dev, &isReady);

        if (isReady)
        {
            vl53l5cx_get_ranging_data(&Dev, &Results);

#if PRINT_MATRIX_ENABLED
            printf("Print data no : %3u\n", Dev.streamcount);
#endif

            for (int i = 0; i < MATRIX_PIXEL_SIZE; i++)
            {
                float filtered_value = Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE * i];

#if IS_ENVELOPE_ENABLED
                if (filtered_value > ENVELOPE_FILTER_VALUE)
                    filtered_value = ENVELOPE_FILTER_VALUE;
#endif

#if IS_KALMAN_ENABLED
                filtered_value = Kalman_Update(&kalman[i], filtered_value);
#endif

                frame[i / 8][i % 8] = (uint16_t)filtered_value;

#if PRINT_MATRIX_ENABLED
                printf("%d", (int)filtered_value);
                if (i % 8 == 7) printf("\n");
                else printf(",");
#endif
            }

            // -------------------------------
            // AI inference
            // -------------------------------
            if (ai_ok)
            {
                float valid = 0.0f;
                float xpos = 0.0f;

                if (ai_run(frame, &valid, &xpos))
                {
                    printf("AI => valid = %.3f | x = %.3f\n", valid, xpos);
                }
                else
                {
                    printf("AI run failed\n");
                }
            }
        }

        WaitMs(&(Dev.platform), 5);
    }
}
