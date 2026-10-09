#pragma once
#include "model.h"
#define MODEL_DATA traffic_sign_int8_tflite
#define IMG_W 32
#define IMG_H 32
#define IMG_CHANNELS 3
#define NUM_CLASSES 10
#define TENSOR_ARENA_KB 200
#define MODEL_DTYPE_UINT8
#define RESOLVER_SIZE 11

#define REGISTER_OPS(r) do { \
    (r).AddConv2D(); \
    (r).AddDepthwiseConv2D(); \
    (r).AddAveragePool2D(); \
    (r).AddReshape(); \
    (r).AddFullyConnected(); \
    (r).AddSoftmax(); \
    (r).AddQuantize(); \
    (r).AddMean(); \
    (r).AddPad(); \
    (r).AddTranspose(); \
    (r).AddMaxPool2D(); \
} while (0)
