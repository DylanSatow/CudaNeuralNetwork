#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#endif

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Data Structures
typedef struct
{
    int inputSize;
    int outputSize;
    float *weights;
    float *biases;
    float *d_weights;
    float *d_biases;
} Layer;

typedef struct
{
    int numLayers;
    Layer *layers;
    float learningRate;
} NeuralNetwork;

void initLayer(Layer *layer, int inputSize, int outputSize);
void initNetwork(NeuralNetwork *network, int *layerSizes, int numLayers, float learningRate);
void freeLayer(Layer *layer);
void freeNetwork(NeuralNetwork *network);
void forwardLayer(const Layer *layer, const float *input, float *output, int batchSize);
void forwardNetwork(const NeuralNetwork *network, const float *input, float *output, int batchSize);
