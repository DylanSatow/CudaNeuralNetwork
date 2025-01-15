#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>   // for rand(), srand()
#include <time.h>     // for time(NULL)
#include <math.h>     // for sqrtf
#include "neural_network.h"
#include "utils.h"

// Data Structures
typedef struct {
    int inputSize;
    int outputSize;
    float *weights;
    float *biases;
    float *d_weights;
    float *d_biases;
} Layer;

typedef struct {
    int numLayers;
    Layer *layers;
    float learningRate;
} NeuralNetwork;

// Initializations
void initLayer(Layer *layer, int inputSize, int outputSize) {
    
    // Store dimensions
    layer->inputSize = inputSize;
    layer->outputSize = outputSize;

    // Calculate parameter sizes
    size_t weightBytes = inputSize * outputSize * sizeof(float);
    size_t biasBytes = outputSize * sizeof(float);

    // Allocate GPU memory
    checkCudaError(cudaMalloc((void **)&(layer->weights), weightBytes), "cudaMalloc layer->weights");
    checkCudaError(cudaMalloc((void **)&(layer->biases), biasBytes), "cudaMalloc layer->biases");
    checkCudaError(cudaMalloc((void **)&(layer->d_weights), weightBytes), "cudaMalloc layer->d_weights");
    checkCudaError(cudaMalloc((void **)&(layer->d_biases), biasBytes), "cudaMalloc layer->d_biases");

    // Allocate temporary host arrays for initialization
    float *h_weights = (float *)malloc(weightBytes);
    float *h_biases = (float *)malloc(biasBytes);

    if (!h_weights || !h_biases)
    {
        fprintf(stderr, "Host memory allocation failed in initLayer\n");
        exit(EXIT_FAILURE);
    }

    // Seed the random generator
    static int seedInitialized = 0;
    if (!seedInitialized)
    {
        srand((unsigned int)time(NULL));
        seedInitialized = 1;
    }

    // Simple random init -- Maybe make more sophisticated?
    float stddev = sqrtf(2.0f / (float)inputSize);
    for (int i = 0; i < inputSize * outputSize; i++)
    {
        // random float in [-1,1]
        float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_weights[i] = r * stddev;
    }

    // Initialize biases to zero (or small random)
    for (int i = 0; i < outputSize; i++)
    {
        h_biases[i] = 0.0f;
    }

    // Copy from host to device
    checkCudaError(cudaMemcpy(layer->weights, h_weights, weightBytes, cudaMemcpyHostToDevice), "cudaMemcpy weights");
    checkCudaError(cudaMemcpy(layer->biases, h_biases, biasBytes, cudaMemcpyHostToDevice), "cudaMemcpy biases");

    // Initialize gradients to zero
    checkCudaError(cudaMemset(layer->d_weights, 0, weightBytes), "cudaMemset d_weights");
    checkCudaError(cudaMemset(layer->d_biases, 0, biasBytes), "cudaMemset d_biases");

    // Free host memory
    free(h_weights);
    free(h_biases);
}

void initNetwork(NeuralNetwork *network, int *layerSizes, int numLayers, float learningRate) {
    // Typically, numLayers might be the count of the *layerSizes array* - 1 if it includes input.
    // E.g.: layerSizes = [784, 128, 10] => numLayers = 2 fully-connected layers.
    // We'll interpret numLayers as "number of layers" in the sense of how many times we call initLayer.

    network->numLayers = numLayers - 1; // e.g. if layerSizes has 3 elements => 2 layers
    network->learningRate = learningRate;

    // Allocate host array for the layers
    network->layers = (Layer *)malloc(network->numLayers * sizeof(Layer));
    if (!network->layers)
    {
        fprintf(stderr, "Host memory allocation failed in initNetwork\n");
        exit(EXIT_FAILURE);
    }

    // Initialize each layer
    for (int i = 0; i < network->numLayers; i++) {
        int inSize = layerSizes[i];
        int outSize = layerSizes[i + 1];

        initLayer(&(network->layers[i]), inSize, outSize);
    }

    // Print Summary Info
    printf("[initNetwork] Created %d layers.\n", network->numLayers);
    for (int i = 0; i < network->numLayers; i++)
    {
        printf("  Layer %d: inputSize=%d, outputSize=%d\n",
               i, network->layers[i].inputSize, network->layers[i].outputSize);
    }
}

void freeLayer(Layer *layer) {
    if (layer->weights)
        cudaFree(layer->weights);
    if (layer->biases)
        cudaFree(layer->biases);
    if (layer->d_weights)
        cudaFree(layer->d_weights);
    if (layer->d_biases)
        cudaFree(layer->d_biases);

    layer->weights = NULL;
    layer->biases = NULL;
    layer->d_weights = NULL;
    layer->d_biases = NULL;
}

void freeNetwork(NeuralNetwork *network)
{
    for (int i = 0; i < network->numLayers; i++)
    {
        freeLayer(&(network->layers[i]));
    }
    free(network->layers);
    network->layers = NULL;
}
