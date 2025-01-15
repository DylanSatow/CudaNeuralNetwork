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

void initNetwork(NeuralNetwork *network, int *layerSizes, int numLayers, float learningRate)
{
    network->numLayers = numLayers - 1;
    network->learningRate = learningRate;

    // Allocate host array for Layers
    network->layers = (Layer *)malloc(network->numLayers * sizeof(Layer));
    if (!network->layers)
    {
        fprintf(stderr, "Host memory allocation failed in initNetwork\n");
        exit(EXIT_FAILURE);
    }

    // Initialize each layer
    for (int i = 0; i < network->numLayers; i++)
    {
        initLayer(&(network->layers[i]), layerSizes[i], layerSizes[i + 1]);
    }

    // Allocate space for storing intermediate activations
    // We have network->numLayers + 1 arrays: index 0 for the original input, then 1..numLayers for each layer’s output
    network->activations = (float **)malloc((network->numLayers + 1) * sizeof(float *));
    if (!network->activations)
    {
        fprintf(stderr, "Host memory allocation failed for activations pointers\n");
        exit(EXIT_FAILURE);
    }
    // We only allocate the actual memory for each forward call,
    // or we can do it here if we know a fixed batchSize in advance.

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
    const int numLayers = network->numLayers;
    if (numLayers == 0)
    {
        // edge case: no layers => nothing to do
        return;
    }

    // 1) Allocate GPU buffers for each layer's output
    //    network->activations[0] is the input
    network->activations[0] = (float *)input; // we do NOT own this memory; user provided

    for (int i = 0; i < numLayers; i++)
    {
        int outSize = network->layers[i].outputSize;
        size_t bytes = batchSize * outSize * sizeof(float);

        // Allocate a buffer for this layer’s output
        checkCudaError(cudaMalloc(&(network->activations[i + 1]), bytes), "cudaMalloc layer output");
    }
    // 2) Forward pass layer by layer
    for (int i = 0; i < numLayers; i++)
    {
        const Layer *layer = &(network->layers[i]);
        const float *currentInput = network->activations[i];
        float *currentOutput = network->activations[i + 1];

        forwardLayer(layer, currentInput, currentOutput, batchSize);
    }

    // 3) The final layer’s output is in network->activations[numLayers].

    if (output != nullptr)
    {
        float *finalOutput = network->activations[numLayers];
        int finalSize = network->layers[numLayers - 1].outputSize;

        checkCudaError(cudaMemcpy(output,
                                  finalOutput,
                                  batchSize * finalSize * sizeof(float),
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy final layer output");
    }
}

/* --------------------- *
 *  Backward Pass        *
 * --------------------- */

__global__ void computeOutputGradientKernel(const float *pred,
                                            const float *target,
                                            float *dOut,
                                            int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        dOut[idx] = (pred[idx] - target[idx]); // dLoss/dPred for MSE
    }
}

__global__ void backwardLayerKernel(const float *inputActivations,  // (batchSize, inputSize)
                                    const float *outputActivations, // (batchSize, outputSize) after ReLU
                                    const float *outputGradients,   // (batchSize, outputSize) dL/dY
                                    float *d_weights,               // (inputSize, outputSize)
                                    float *d_biases,                // (outputSize)
                                    float *inputGradients,          // (batchSize, inputSize) dL/dX
                                    const float *weights,           // (inputSize, outputSize)
                                    int inputSize,
                                    int outputSize,
                                    int batchSize)
{
    int outNeuron = blockIdx.x * blockDim.x + threadIdx.x; // which output neuron
    int sample = blockIdx.y * blockDim.y + threadIdx.y;    // which batch sample
    if (outNeuron < outputSize && sample < batchSize)
    {
        // Compute dZ = dY * ReLU'(Z).  Z>0 iff outputActivations>0.
        // We'll get the post-activation from 'outputActivations'.
        float dY = outputGradients[sample * outputSize + outNeuron];
        float a = outputActivations[sample * outputSize + outNeuron];
        float dZ = (a > 0.0f) ? dY : 0.0f;

        // Accumulate bias gradient
        atomicAdd(&d_biases[outNeuron], dZ);

        // For each input neuron, update d_weights and inputGradients
        for (int inNeuron = 0; inNeuron < inputSize; inNeuron++)
        {
            float xval = inputActivations[sample * inputSize + inNeuron];
            // d_weight = xval * dZ
            atomicAdd(&d_weights[inNeuron * outputSize + outNeuron], xval * dZ);

            // dX = sum_j( dZ_j * W_{inNeuron, j} )
            // but here j is just outNeuron in this loop
            float wval = weights[inNeuron * outputSize + outNeuron];
            atomicAdd(&inputGradients[sample * inputSize + inNeuron], dZ * wval);
        }
    }
}

void backwardLayer(Layer *layer,
                   const float *inputActivations,  // a^{l-1}
                   const float *outputActivations, // a^l
                   const float *outputGradients,   // dL/d(a^l)
                   float *inputGradients,          // dL/d(a^{l-1})
                   int batchSize)
{
    // First, zero out the d_weights, d_biases in this layer
    checkCudaError(cudaMemset(layer->d_weights, 0,
                              layer->inputSize * layer->outputSize * sizeof(float)),
                   "cudaMemset d_weights");
    checkCudaError(cudaMemset(layer->d_biases, 0,
                              layer->outputSize * sizeof(float)),
                   "cudaMemset d_biases");

    // Also zero out inputGradients (the dL/dX) we will produce
    checkCudaError(cudaMemset(inputGradients, 0,
                              batchSize * layer->inputSize * sizeof(float)),
                   "cudaMemset inputGradients");

    // Configure the kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((layer->outputSize + blockDim.x - 1) / blockDim.x,
                 (batchSize + blockDim.y - 1) / blockDim.y);

    // Launch
    backwardLayerKernel<<<gridDim, blockDim>>>(
        inputActivations,
        outputActivations,
        outputGradients,
        layer->d_weights,
        layer->d_biases,
        inputGradients,
        layer->weights,
        layer->inputSize,
        layer->outputSize,
        batchSize);
    checkCudaError(cudaGetLastError(), "backwardLayerKernel launch");
    checkCudaError(cudaDeviceSynchronize(), "backwardLayerKernel sync");
}

void backwardNetwork(NeuralNetwork *network,
                     const float *target, // (batchSize x lastLayerSize)
                     int batchSize)
{
    int L = network->numLayers;
    if (L == 0)
        return;

    // 1) Allocate device memory for gradient wrt each layer's activation
    float **gradBuffer = (float **)malloc((L + 1) * sizeof(float *));
    for (int i = 0; i <= L; i++)
    {
        // The i-th activation has shape (batchSize, layerSizes[i])
        int size = (i == 0) ? network->layers[0].inputSize
                            : network->layers[i - 1].outputSize;
        checkCudaError(cudaMalloc(&gradBuffer[i], batchSize * size * sizeof(float)),
                       "cudaMalloc gradBuffer");
    }

    // 2) Compute dLoss/dOutput for the last layer
    {
        // last layer activation is network->activations[L]
        int finalSize = network->layers[L - 1].outputSize;
        int total = batchSize * finalSize;

        dim3 blockDim(256);
        dim3 gridDim((total + blockDim.x - 1) / blockDim.x);
        computeOutputGradientKernel<<<gridDim, blockDim>>>(
            network->activations[L], // pred
            target,                  // target
            gradBuffer[L],           // dLoss/dPred
            total);
        checkCudaError(cudaGetLastError(), "computeOutputGradientKernel launch");
        checkCudaError(cudaDeviceSynchronize(), "computeOutputGradientKernel sync");
    }

    // 3) Now go in reverse order of layers
    for (int i = L - 1; i >= 0; i--)
    {
        Layer *layer = &network->layers[i];

        // activation of i-th layer input: network->activations[i]
        // activation of i-th layer output: network->activations[i+1]
        // dL/d(a^l) is in gradBuffer[i+1]
        // we want to produce dL/d(a^{l-1}) in gradBuffer[i]

        backwardLayer(layer,
                      network->activations[i],     // inputActivations
                      network->activations[i + 1], // outputActivations
                      gradBuffer[i + 1],           // outputGradients
                      gradBuffer[i],               // inputGradients
                      batchSize);
    }

    // 4) We now have layer->d_weights, layer->d_biases for each layer.
    //    Next step is to update them using the chosen learning rate.
    //    (We'll do that in updateNetworkParameters().)

    // Cleanup
    for (int i = 0; i <= L; i++)
    {
        cudaFree(gradBuffer[i]);
    }
    free(gradBuffer);
}

__global__ void updateParametersKernel(float *params,
                                       const float *grads,
                                       float lr,
                                       int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        params[idx] -= lr * grads[idx];
    }
}

void updateLayerParameters(Layer *layer, float lr)
{
    // weights
    {
        int size = layer->inputSize * layer->outputSize;
        dim3 blockDim(256);
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
        updateParametersKernel<<<gridDim, blockDim>>>(
            layer->weights,
            layer->d_weights,
            lr,
            size);
        checkCudaError(cudaGetLastError(), "updateParametersKernel for weights");
        checkCudaError(cudaDeviceSynchronize(), "updateParametersKernel for weights sync");
    }
    // biases
    {
        int size = layer->outputSize;
        dim3 blockDim(256);
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
        updateParametersKernel<<<gridDim, blockDim>>>(
            layer->biases,
            layer->d_biases,
            lr,
            size);
        checkCudaError(cudaGetLastError(), "updateParametersKernel for biases");
        checkCudaError(cudaDeviceSynchronize(), "updateParametersKernel for biases sync");
    }
}

void updateNetworkParameters(NeuralNetwork *network)
{
    for (int i = 0; i < network->numLayers; i++)
    {
        updateLayerParameters(&network->layers[i], network->learningRate);
    }
}

void trainNetwork(NeuralNetwork *network,
                  const float *d_trainImages, // all training images on GPU
                  const float *d_trainLabels, // all training labels on GPU (one-hot or not)
                  int numSamples,
                  int batchSize,
                  int epochs)
{
    int stepsPerEpoch = numSamples / batchSize;

    for (int e = 0; e < epochs; e++)
    {
        float epochLoss = 0.0f; // accumulate if you like

        for (int s = 0; s < stepsPerEpoch; s++)
        {
            // 1) Slice out mini-batch from d_trainImages, d_trainLabels
            //    We assume you have a function that returns pointers to the
            //    batch in device memory: d_batchImages, d_batchLabels
            const float *d_batchImages = d_trainImages + (s * batchSize * network->layers[0].inputSize);
            const float *d_batchLabels = d_trainLabels + (s * batchSize * network->layers[network->numLayers - 1].outputSize);

            // 2) Forward
            forwardNetwork(network, d_batchImages, /*out=*/nullptr, batchSize);

            // Optionally compute the loss here if you want to measure epochLoss

            // 3) Backward
            backwardNetwork(network, d_batchLabels, batchSize);

            // 4) Update
            updateNetworkParameters(network);

            // 5) Free intermediate activations that were allocated in forwardNetwork
            //    (except the user-provided input pointer)
            for (int i = 1; i <= network->numLayers; i++)
            {
                cudaFree(network->activations[i]);
            }
        }
        printf("Epoch %d done.\n", e);
    }
}
