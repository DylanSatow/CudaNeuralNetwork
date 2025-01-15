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

// In your NeuralNetwork structure, add something like:
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
void forwardLayer(const Layer *layer, const float *input, float *output, int batchSize);
void forwardNetwork(const NeuralNetwork *network, const float *input, float *output, int batchSize);

void trainNetwork(NeuralNetwork *network,
                  const float *d_trainImages, // all training images on GPU
                  const float *d_trainLabels, // all training labels on GPU (one-hot or not)
                  int numSamples,
                  int batchSize,
                  int epochs);