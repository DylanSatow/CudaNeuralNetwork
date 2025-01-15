#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>   // for rand(), srand()
#include <time.h>     // for time(NULL)
#include <math.h>     // for sqrtf
#include "include/neural_network.h"
#include "include/utils.h"

/*
***Initialization***
*/
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

void freeNetwork(NeuralNetwork *network) {
    for (int i = 0; i < network->numLayers; i++)
    {
        freeLayer(&(network->layers[i]));
    }
    free(network->layers);
    network->layers = NULL;
}

/*
***Forward Pass***
*/

__global__ void forwardLayerKernel(const float *weights,
                                   const float *biases,
                                   const float *input,
                                   float *output,
                                   int inputSize,
                                   int outputSize,
                                   int batchSize)
{
    // 2D indexing:
    //   row = index over batch
    //   col = index over output neurons
    int row = blockIdx.y * blockDim.y + threadIdx.y; // which sample in the batch
    int col = blockIdx.x * blockDim.x + threadIdx.x; // which output neuron

    if (row < batchSize && col < outputSize)
    {
        // Dot product: input[row, :] (length inputSize) with weights[:, col]
        float val = 0.0f;
        for (int i = 0; i < inputSize; i++)
        {
            val += input[row * inputSize + i] * weights[i * outputSize + col];
        }

        // Add bias
        val += biases[col];

        // (Optional) activation, e.g. ReLU:
        val = fmaxf(0.0f, val);

        // Store result
        output[row * outputSize + col] = val;
    }
}

void forwardLayer(const Layer *layer, 
                  const float *input, 
                  float *output, 
                  int batchSize)
{
    // layer->weights:   (inputSize x outputSize)
    // layer->biases:    (outputSize)
    // input:            (batchSize x inputSize)
    // output:           (batchSize x outputSize)

    const int inputSize  = layer->inputSize;
    const int outputSize = layer->outputSize;

    // Configure a 2D block and grid. 
    // E.g. 16x16 threads per block (tune as needed).
    dim3 blockDim(16, 16);
    dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x,
                 (batchSize  + blockDim.y - 1) / blockDim.y);

    // Launch our kernel
    forwardLayerKernel<<<gridDim, blockDim>>>(
        layer->weights,
        layer->biases,
        input,
        output,
        inputSize,
        outputSize,
        batchSize
    );
    checkCudaError(cudaGetLastError(), "forwardLayerKernel launch");
    checkCudaError(cudaDeviceSynchronize(), "forwardLayerKernel sync");
}

void forwardNetwork(const NeuralNetwork *network,
                    const float *input,
                    float *output,
                    int batchSize)
{
    // We assume 'input' is on GPU of shape (batchSize x layerSizes[0])
    // We want 'output' on GPU of shape (batchSize x layerSizes[last])

    const int numLayers = network->numLayers;

    // If there's no layer, just copy input to output (edge case)
    if (numLayers == 0)
    {
        // Possibly just do memcpy if same shape
        return;
    }

    // We’ll use a double-buffer approach for intermediate results:
    //   bufferA and bufferB both allocated up to the max layer-size needed.
    // We'll flip between them as we go forward layer by layer.
    // So we never do extra malloc/free inside the loop.

    // 1) Determine the largest possible output size among all layers
    int maxOutputNeurons = 0;
    for (int i = 0; i < numLayers; i++)
    {
        if (network->layers[i].outputSize > maxOutputNeurons)
            maxOutputNeurons = network->layers[i].outputSize;
    }
    size_t maxBytes = maxOutputNeurons * batchSize * sizeof(float);

    // 2) Allocate two temp buffers
    float *bufferA = nullptr;
    float *bufferB = nullptr;
    checkCudaError(cudaMalloc(&bufferA, maxBytes), "cudaMalloc bufferA");
    checkCudaError(cudaMalloc(&bufferB, maxBytes), "cudaMalloc bufferB");

    // currentInput points to the input buffer for the current layer
    // currentOutput points to the output buffer for the current layer
    const float *currentInput = input; // first layer sees the user input
    float *currentOutput = bufferA;    // it can write to bufferA initially

    for (int i = 0; i < numLayers; i++)
    {
        const Layer *layer = &(network->layers[i]);

        // If this is the last layer, we'll write the result directly into "output"
        // to avoid an extra copy.
        if (i == numLayers - 1)
            currentOutput = output;

        // Forward pass for the i-th layer
        forwardLayer(layer, currentInput, currentOutput, batchSize);

        // Prepare for next iteration:
        // swap buffers so next layer reads from what we just wrote
        if (i < numLayers - 1)
        {
            // The next layer's input will be the current layer's output
            // Flip the pointer so we don't keep overwriting the same buffer
            if (currentOutput == bufferA)
            {
                currentInput = bufferA;
                currentOutput = bufferB;
            }
            else
            {
                currentInput = bufferB;
                currentOutput = bufferA;
            }
        }
    }

    // Free temp buffers
    checkCudaError(cudaFree(bufferA), "cudaFree bufferA");
    checkCudaError(cudaFree(bufferB), "cudaFree bufferB");
}

// /* --------------------- *
//  *  Backward Pass        *
//  * --------------------- */
// void backwardLayer(Layer *layer,
//                    const float *inputActivations,
//                    const float *outputGradients,
//                    float *inputGradients,
//                    int batchSize)
// {
//     // compute d_weights, d_biases, inputGradients
// }

// void backwardNetwork(Network *network,
//                      const float *input,
//                      const float *target,
//                      int batchSize)
// {
//     // compute dLoss/dOutput
//     // for each layer in reverse order:
//     //    backwardLayer(...)
// }

// /* --------------------- *
//  *  Update Parameters    *
//  * --------------------- */
// __global__ void updateParametersKernel(float *params, const float *grads, float lr, int size)
// {
//     // subtract lr * grad
// }

// void updateLayerParameters(Layer *layer, float lr)
// {
//     // Launch updateParametersKernel for weights, biases
// }

// void updateNetworkParameters(Network *network)
// {
//     // For each layer, call updateLayerParameters
// }

// /* --------------------- *
//  *  Training (Optional)  *
//  * --------------------- */
// void trainNetwork(Network *network,
//                   const float *d_trainImages,
//                   const float *d_trainLabels,
//                   int numSamples,
//                   int batchSize,
//                   int epochs)
// {
//     // implement training loop
// }