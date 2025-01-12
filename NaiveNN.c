// Neural Network implementation in C with MNIST loader
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// MNIST file format constants
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_LABEL_MAGIC 0x00000801

// Structure definitions
typedef struct
{
    float *data;
    int rows;
    int cols;
} Matrix;

typedef struct
{
    Matrix *weights1;
    Matrix *bias1;
    Matrix *weights2;
    Matrix *bias2;
    // Cache for backpropagation
    Matrix *input;
    Matrix *layer1;
    Matrix *relu_output;
    Matrix *layer2;
    Matrix *probs;
    // Gradients
    Matrix *grad_weights1;
    Matrix *grad_bias1;
    Matrix *grad_weights2;
    Matrix *grad_bias2;
    // Dimensions
    int input_size;
    int hidden_size;
    int output_size;
} NeuralNetwork;
