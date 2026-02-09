#include "ai_inference.h"
#include "background.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "esp_timer.h"


#if USE_INT8_MODEL
  #include "model_int8.h"
#else
  #include "model_float.h"
#endif

#define TENSOR_ARENA_SIZE (80 * 1024)

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter = NULL;
static TfLiteTensor* input_tensor = NULL;
static TfLiteTensor* out_valid_tensor = NULL;

bool ai_init(void)
{
    #if USE_INT8_MODEL
        const tflite::Model* model = tflite::GetModel(tof_model_int8_tflite);
    #else
        const tflite::Model* model = tflite::GetModel(tof_model_float_tflite);
    #endif

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        MicroPrintf("Model schema mismatch");
        return false;
    }

    static tflite::MicroMutableOpResolver<8> resolver;

    // Register what the model uses:
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddLogistic();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        MicroPrintf("AllocateTensors failed");
        return false;
    }

    size_t arena_used = interpreter->arena_used_bytes();
    printf("Arena used: %u bytes\n", (unsigned int)arena_used);

    input_tensor = interpreter->input(0);
    out_valid_tensor = interpreter->output(0);

    MicroPrintf("AI initialized");

    return true;
}

bool ai_run(const uint16_t frame[8][8], float* out_valid, float* out_x)
{
    if (!interpreter) return false;

    float energy[8] = {0};

    // -------------------------------
    // Preprocess + quantization + energy sum
    // -------------------------------
    for (int y = 0; y < 8; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            int d = (int)background[y][x] - (int)frame[y][x];
            if (d < 0) d = 0;
            if (d > 400) d = 400;

            float norm = (float)d / 400.0f;

            // energy sum per column
            energy[x] += norm;

            #if USE_INT8_MODEL
                float scale = input_tensor->params.scale;
                int zero = input_tensor->params.zero_point;

                int q = (int)roundf(norm / scale) + zero;
                if (q < -128) q = -128;
                if (q > 127) q = 127;

                input_tensor->data.int8[y * 8 + x] = (int8_t)q;
            #else
                input_tensor->data.f[y * 8 + x] = norm;
            #endif
        }
    }

    // -------------------------------
    // Run inference
    // -------------------------------
    int64_t t0 = esp_timer_get_time();

    if (interpreter->Invoke() != kTfLiteOk)
    {
        return false;
    }

    int64_t t1 = esp_timer_get_time();

    printf("Inference time: %lld us\n", (long long)(t1 - t0));

    // -------------------------------
    // Output
    // -------------------------------
    #if USE_INT8_MODEL
        int8_t q = out_valid_tensor->data.int8[0];
        float valid = (q - out_valid_tensor->params.zero_point) *
                    out_valid_tensor->params.scale;
    #else
        float valid = out_valid_tensor->data.f[0];
    #endif

    *out_valid = valid;

    // -------------------------------
    // Compute centroid
    // -------------------------------
    if (valid > 0.5f)
    {
        float sum = 0;
        float weighted = 0;

        for (int i = 0; i < 8; i++)
        {
            sum += energy[i];
            weighted += energy[i] * i;
        }

        if (sum > 1e-6f)
            *out_x = (weighted / sum) / 7.0f;   // normalize [0,1]
        else
            *out_x = 0.5f;
    }
    else
    {
        *out_x = 0.5f; // default
    }

    return true;
}
