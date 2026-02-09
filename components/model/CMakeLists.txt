idf_component_register(
    SRCS "ai_inference.cpp"
    INCLUDE_DIRS "include"
    REQUIRES
        esp-tflite-micro
        esp_timer
)