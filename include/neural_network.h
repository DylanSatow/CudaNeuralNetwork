#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#endif

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int inputSize;
    int outputSize;

    float *weights;   // Device memory
    float *biases;    // Device memory
    float *d_weights; // Device memory (gradients)
    float *d_biases;  // Device memory (gradients)

    float *weight_momentum; // Device memory
    float *bias_momentum;   // Device memory
} Layer;

typedef struct
{
    Layer *layers;
    int numLayers;
    float learningRate;
    float **activations; // an array of pointers to device memory
} NeuralNetwork;

void initLayer(Layer *layer, int inputSize, int outputSize);
void initNetwork(NeuralNetwork *network, int *layerSizes, int numLayers, float learningRate);
void freeLayer(Layer *layer);
void freeNetwork(NeuralNetwork *network);
void forwardLayer(const Layer *layer,
                  const float *input,
                  float *output,
                  int batchSize,
                  bool isLastLayer);

void forwardNetwork(const NeuralNetwork *network, const float *input, float *output, int batchSize);

float trainNetwork(NeuralNetwork *network,
                   const float *d_trainImages,
                   const float *d_trainLabels,
                   int numSamples,
                   int batchSize);