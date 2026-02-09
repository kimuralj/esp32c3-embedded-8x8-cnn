//----------------------- Includes ---------------------------/
#include "kalman.h"

//------------------ Function Definitions --------------------/
/**
 * @brief Initialize an instance of the Kalmann filter.
 * 
 * @param data Pointer to a struct with the initial data.
 * @param initial_value The initial value to be considered.
 */
void Kalman_Initialize(KALMAN_STRUCT_T* data, float initial_value) {
    // Set the previous value as the initial value
    data->x_hat = initial_value;
    // High uncertainty in the initialization
    data->P = 1.0;
}

/**
 * @brief Updates the estimative of a filter instance with a new reading.
 * 
 * @param data Pointer to a struct with the previous data.
 * @param current_reading The current value measured by the sensor.
 * @return The filtered value.
 */
float Kalman_Update(KALMAN_STRUCT_T* data, float current_reading) {
    /* 
     * 1st step: Predict the new value.
     * This model considers that the object is static, so the new
     * value is considered the same as the previous one.
     */
    float predicted_value = data->x_hat;

    /*
     * 2nd step: Predict the error covariance.
     */
    float predicted_uncertainty = data->P + Q;

    /*
     * 3rd step: Calculate the Kalman gain
     */
    float kalman_gain = predicted_uncertainty / (predicted_uncertainty + R);

    /*
     * 4th step: Update the predicted value
     */
    data->x_hat = predicted_value + kalman_gain * (current_reading - predicted_value);

    /* 
     * 5th step: Upadte the error covariance.
     */
    data->P = (1.0f - kalman_gain) * predicted_uncertainty;

    /*
     * Returns the filtered value.
     */
    return data->x_hat;
}
