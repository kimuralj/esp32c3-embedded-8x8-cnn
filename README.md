# esp32c3-embedded-8x8-cnn
This repository implements a lightweight Convolutional Neural Network (CNN) running on the ESP32-C3 for embedded object detection/classification using 8×8 sensor data.

## Overview

This project aims at reading the data of a Time-of-Flight VL53L5CX sensor from STMicroelectronics. The sensor is connected through I2C and the data is sent via serial.

### Hardware Required

Two basic things are required for this project to run: a VL53L5CX ToF sensor and an ESP32-C3 (it could work with other ESP32 models as well).


### Build and Flash

Enter `idf.py -p PORT flash monitor` to build, flash and monitor the project.

**Note:** It is necessary to increase value of CONFIG_MAIN_TASK_STACK_SIZE, if the system is not behaving as expect, try doubling this value.
