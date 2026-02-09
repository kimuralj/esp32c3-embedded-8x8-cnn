#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============ BUILD SWITCH ============
#define USE_INT8_MODEL   1   // 1 = INT8, 0 = FLOAT
// =====================================

bool ai_init(void);

/**
 * @brief Run AI model
 * @param frame 8x8 in milimeters
 * @param out_valid output [0,1]
 * @param out_x output [0,1]
 */
bool ai_run(const uint16_t frame[8][8], float* out_valid, float* out_x);

#ifdef __cplusplus
}
#endif
