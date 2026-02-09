#ifndef KALMAN_H
#define KALMAN_H

//----------------------- Includes ---------------------------/
#include <stdio.h>

//----------------------- Constants --------------------------/
/**
 * Variance of the model noise.
 * The value is proportional to how noisier the model is.
 */
#define Q 0.5
/**
 * Variance of the measurement noise.
 * The value is proportional to how noisier the measurement is.
 */
#define R 4.00

//------------------------ Structs ---------------------------/
typedef struct {
    float x_hat;    // Estimated value
    float P;        // Estimated uncertainty
} KALMAN_STRUCT_T;

//------------------ Function Declarations -------------------/
/**
 * @brief Initialize an instance of the Kalmann filter.
 * 
 * @param data Pointer to a struct with the initial data.
 * @param initial_value The initial value to be considered.
 */
void Kalman_Initialize(KALMAN_STRUCT_T* data, float initial_value);

/**
 * @brief Updates the estimative of a filter instance with a new reading.
 * 
 * @param data Pointer to a struct with the previous data.
 * @param current_reading The current value measured by the sensor.
 * @return The filtered value.
 */
float Kalman_Update(KALMAN_STRUCT_T* data, float current_reading);

#endif // KALMAN_H
