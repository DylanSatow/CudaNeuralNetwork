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

float float_rand(float min, float max) // see stack overflow
{
    float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
    return min + scale * (max - min);       /* [min, max] */
}

void initLayer(Layer *layer, int inputSize, int outputSize)
{
    // Store dimensions
    layer->inputSize = inputSize;
    layer->outputSize = outputSize;

    // Calculate sizes
    size_t weightBytes = inputSize * outputSize * sizeof(float);
    size_t biasBytes = outputSize * sizeof(float);

    // Allocate GPU memory for parameters and gradients
    checkCudaError(cudaMalloc((void **)&(layer->weights), weightBytes), "cudaMalloc weights");
    checkCudaError(cudaMalloc((void **)&(layer->biases), biasBytes), "cudaMalloc biases");
    checkCudaError(cudaMalloc((void **)&(layer->d_weights), weightBytes), "cudaMalloc d_weights");
    checkCudaError(cudaMalloc((void **)&(layer->d_biases), biasBytes), "cudaMalloc d_biases");

    // Allocate GPU memory for momentum buffers
    checkCudaError(cudaMalloc((void **)&(layer->weight_momentum), weightBytes), "cudaMalloc weight_momentum");
    checkCudaError(cudaMalloc((void **)&(layer->bias_momentum), biasBytes), "cudaMalloc bias_momentum");

    // Initialize momentum buffers to zero
    checkCudaError(cudaMemset(layer->weight_momentum, 0, weightBytes), "cudaMemset weight_momentum");
    checkCudaError(cudaMemset(layer->bias_momentum, 0, biasBytes), "cudaMemset bias_momentum");

    // Allocate temporary host arrays for initialization
    float *h_weights = (float *)malloc(weightBytes);
    float *h_biases = (float *)malloc(biasBytes);

    if (!h_weights || !h_biases)
    {
        fprintf(stderr, "Host memory allocation failed in initLayer\n");
        exit(EXIT_FAILURE);
    }

    // Initialize weights with He initialization
    float stddev = sqrtf(2.0f / (float)inputSize);
    for (int i = 0; i < inputSize * outputSize; i++)
    {
        float r = float_rand(-2.0f, 2.0f);
        h_weights[i] = r * stddev;
    }

    // Initialize biases to zero instead of 0.1
    for (int i = 0; i < outputSize; i++)
    {
        h_biases[i] = 0.0f;
    }

    // Copy initialized values to GPU
    checkCudaError(cudaMemcpy(layer->weights, h_weights, weightBytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy weights");
    checkCudaError(cudaMemcpy(layer->biases, h_biases, biasBytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy biases");

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

void freeLayer(Layer *layer)
{
    if (layer->weights)
        cudaFree(layer->weights);
    if (layer->biases)
        cudaFree(layer->biases);
    if (layer->d_weights)
        cudaFree(layer->d_weights);
    if (layer->d_biases)
        cudaFree(layer->d_biases);
    if (layer->weight_momentum)
        cudaFree(layer->weight_momentum);
    if (layer->bias_momentum)
        cudaFree(layer->bias_momentum);

    layer->weights = NULL;
    layer->biases = NULL;
    layer->d_weights = NULL;
    layer->d_biases = NULL;
    layer->weight_momentum = NULL;
    layer->bias_momentum = NULL;
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

__global__ void softmaxKernel(float *output, int batchSize, int outputSize)
{
    int row = blockIdx.x; // batch item
    if (row < batchSize)
    {
        float maxVal = -INFINITY;
        int offset = row * outputSize;

        // Find max value for numerical stability
        for (int i = 0; i < outputSize; i++)
        {
            maxVal = fmaxf(maxVal, output[offset + i]);
        }

        float sum = 0.0f;
        // Compute exp(x - max) and sum
        for (int i = 0; i < outputSize; i++)
        {
            float val = expf(output[offset + i] - maxVal);
            output[offset + i] = val;
            sum += val;
        }

        // Normalize
        for (int i = 0; i < outputSize; i++)
        {
            output[offset + i] /= sum;
        }
    }
}

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
                  int batchSize,
                  bool isLastLayer) // New parameter
{
    // Original matrix multiplication
    dim3 blockDim(16, 16);
    dim3 gridDim((layer->outputSize + blockDim.x - 1) / blockDim.x,
                 (batchSize + blockDim.y - 1) / blockDim.y);

    forwardLayerKernel<<<gridDim, blockDim>>>(
        layer->weights,
        layer->biases,
        input,
        output,
        layer->inputSize,
        layer->outputSize,
        batchSize);

    // Apply softmax only for the last layer
    if (isLastLayer)
    {
        softmaxKernel<<<batchSize, 1>>>(output, batchSize, layer->outputSize);
    }

    checkCudaError(cudaGetLastError(), "forwardLayer kernels");
    checkCudaError(cudaDeviceSynchronize(), "forwardLayer sync");
}

void forwardNetwork(const NeuralNetwork *network,
                    const float *input,
                    float *output,
                    int batchSize)
{
    const int numLayers = network->numLayers;
    if (numLayers == 0)
        return;

    // First activation is the input
    network->activations[0] = (float *)input;

    // Allocate memory for each layer's output
    for (int i = 0; i < numLayers; i++)
    {
        int outSize = network->layers[i].outputSize;
        size_t bytes = batchSize * outSize * sizeof(float);
        checkCudaError(cudaMalloc(&(network->activations[i + 1]), bytes),
                       "cudaMalloc layer output");
    }

    // Forward pass through each layer
    for (int i = 0; i < numLayers; i++)
    {
        const Layer *layer = &network->layers[i];
        const float *currentInput = network->activations[i];
        float *currentOutput = network->activations[i + 1];
        bool isLastLayer = (i == numLayers - 1);

        forwardLayer(layer, currentInput, currentOutput, batchSize, isLastLayer);
    }

    // Copy to output if requested
    if (output != nullptr)
    {
        float *finalOutput = network->activations[numLayers];
        int finalSize = network->layers[numLayers - 1].outputSize;
        checkCudaError(cudaMemcpy(output,
                                  finalOutput,
                                  batchSize * finalSize * sizeof(float),
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy final output");
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

__global__ void backwardLayerKernel(
    const float *inputActivations,
    const float *outputActivations,
    const float *outputGradients,
    float *d_weights,
    float *d_biases,
    float *inputGradients,
    const float *weights,
    int inputSize,
    int outputSize,
    int batchSize,
    bool isLastLayer)
{
    int outNeuron = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y * blockDim.y + threadIdx.y;

    if (outNeuron < outputSize && sample < batchSize)
    {
        float dZ;
        int outOffset = sample * outputSize + outNeuron;

        if (isLastLayer)
        {
            dZ = outputGradients[outOffset];
        }
        else
        {
            float dY = outputGradients[outOffset];
            float a = outputActivations[outOffset];
            dZ = (a > 0.0f) ? dY : 0.0f;
        }

        // Accumulate bias gradient
        atomicAdd(&d_biases[outNeuron], dZ);

        // Update weight gradients and compute input gradients
        for (int inNeuron = 0; inNeuron < inputSize; inNeuron++)
        {
            int inOffset = sample * inputSize + inNeuron;
            float x = inputActivations[inOffset];

            // Weight gradient
            atomicAdd(&d_weights[inNeuron * outputSize + outNeuron], x * dZ);

            // Input gradient (if not the first layer)
            if (inputGradients != nullptr)
            {
                float w = weights[inNeuron * outputSize + outNeuron];
                atomicAdd(&inputGradients[inOffset], w * dZ);
            }
        }
    }
}

void backwardLayer(Layer *layer,
                   const float *inputActivations,
                   const float *outputActivations,
                   const float *outputGradients,
                   float *inputGradients,
                   int batchSize,
                   bool isLastLayer)
{
    // Zero out gradients
    checkCudaError(cudaMemset(layer->d_weights, 0,
                              layer->inputSize * layer->outputSize * sizeof(float)),
                   "cudaMemset d_weights");
    checkCudaError(cudaMemset(layer->d_biases, 0,
                              layer->outputSize * sizeof(float)),
                   "cudaMemset d_biases");
    checkCudaError(cudaMemset(inputGradients, 0,
                              batchSize * layer->inputSize * sizeof(float)),
                   "cudaMemset inputGradients");

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((layer->outputSize + blockDim.x - 1) / blockDim.x,
                 (batchSize + blockDim.y - 1) / blockDim.y);

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
        batchSize,
        isLastLayer);

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
        bool isLastLayer = (i == L - 1);

        backwardLayer(layer,
                      network->activations[i],     // inputActivations
                      network->activations[i + 1], // outputActivations
                      gradBuffer[i + 1],           // outputGradients
                      gradBuffer[i],               // inputGradients
                      batchSize,
                      isLastLayer); // Add this parameter
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

__global__ void updateParametersKernel(
    float *params,
    float *momentum_buffer,
    const float *grads,
    const float lr,
    const float beta,
    const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        momentum_buffer[idx] = beta * momentum_buffer[idx] + (1.0f - beta) * grads[idx];
        params[idx] -= lr * momentum_buffer[idx];
    }
}

void updateLayerParameters(Layer *layer, float lr, float momentum_beta)
{
    // Update weights
    int weightSize = layer->inputSize * layer->outputSize;
    dim3 blockDim(256);
    dim3 gridDim((weightSize + blockDim.x - 1) / blockDim.x);

    updateParametersKernel<<<gridDim, blockDim>>>(
        layer->weights,
        layer->weight_momentum,
        layer->d_weights,
        lr,
        momentum_beta,
        weightSize);

    // Update biases
    int biasSize = layer->outputSize;
    gridDim.x = (biasSize + blockDim.x - 1) / blockDim.x;

    updateParametersKernel<<<gridDim, blockDim>>>(
        layer->biases,
        layer->bias_momentum,
        layer->d_biases,
        lr,
        momentum_beta,
        biasSize);
}

void updateNetworkParameters(NeuralNetwork *network)
{
    float momentum_beta = 0.9f;
    for (int i = 0; i < network->numLayers; i++)
    {
        updateLayerParameters(&network->layers[i], network->learningRate, momentum_beta);
    }
}

// __global__ void computeLossKernel(const float *pred,
//                                   const float *target,
//                                   float *loss,
//                                   int batchSize,
//                                   int outputSize)
// {
//     extern __shared__ float temp[]; // For parallel reduction

//     int tid = threadIdx.x;
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     // Initialize shared memory
//     temp[tid] = 0.0f;

//     // Each thread computes loss for one sample
//     if (idx < batchSize)
//     {
//         float sampleLoss = 0.0f;
//         int offset = idx * outputSize;

//         for (int i = 0; i < outputSize; i++)
//         {
//             float p = pred[offset + i];
//             // Clip predictions for numerical stability
//             p = fmaxf(fminf(p, 1.0f - 1e-7f), 1e-7f);
//             sampleLoss -= target[offset + i] * logf(p);
//         }
//         temp[tid] = sampleLoss;
//     }

//     __syncthreads();

//     // Parallel reduction in shared memory
//     for (int s = blockDim.x / 2; s > 0; s >>= 1)
//     {
//         if (tid < s)
//         {
//             temp[tid] += temp[tid + s];
//         }
//         __syncthreads();
//     }

//     // Write block result to global memory
//     if (tid == 0)
//     {
//         atomicAdd(loss, temp[0] / batchSize); // Normalize by batch size
//     }
// }

__global__ void computeBatchLossKernel(
    const float *predictions, // Shape: [batchSize, outputSize]
    const float *targets,     // Shape: [batchSize, outputSize]
    float *loss,              // Single float for accumulated loss
    int batchSize,
    int outputSize)
{
    extern __shared__ float temp[]; // Shared memory for reduction
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    temp[tid] = 0.0f;

    // Each thread handles one sample in the batch
    if (idx < batchSize)
    {
        float sampleLoss = 0.0f;
        int offset = idx * outputSize;

        // Compute cross-entropy loss for this sample
        for (int i = 0; i < outputSize; i++)
        {
            float pred = predictions[offset + i];
            // Clip predictions for numerical stability
            pred = fmaxf(fminf(pred, 1.0f - 1e-7f), 1e-7f);
            sampleLoss -= targets[offset + i] * logf(pred);
        }
        temp[tid] = sampleLoss;
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (tid == 0)
    {
        atomicAdd(loss, temp[0]);
    }
}

float computeBatchLoss(
    const float *predictions,
    const float *targets,
    int batchSize,
    int outputSize)
{
    float *d_loss;
    checkCudaError(cudaMalloc(&d_loss, sizeof(float)), "cudaMalloc loss");
    checkCudaError(cudaMemset(d_loss, 0, sizeof(float)), "cudaMemset loss");

    // Configure kernel launch parameters
    int blockSize = 256;
    int numBlocks = (batchSize + blockSize - 1) / blockSize;

    // Launch kernel with shared memory for reduction
    computeBatchLossKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        predictions,
        targets,
        d_loss,
        batchSize,
        outputSize);

    // Check for kernel errors
    checkCudaError(cudaGetLastError(), "computeBatchLossKernel launch");
    checkCudaError(cudaDeviceSynchronize(), "computeBatchLossKernel sync");

    // Get the result
    float totalLoss;
    checkCudaError(cudaMemcpy(&totalLoss, d_loss, sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy loss");

    // Free temporary memory
    cudaFree(d_loss);

    // Return average loss per sample
    return totalLoss / batchSize;
}

float trainNetwork(NeuralNetwork *network,
                   const float *d_trainImages,
                   const float *d_trainLabels,
                   int numSamples,
                   int batchSize)
{
    int stepsPerEpoch = (numSamples + batchSize - 1) / batchSize; // Ceiling division
    float totalLoss = 0.0f;

    for (int s = 0; s < stepsPerEpoch; s++)
    {
        int currentBatchSize = min(batchSize, numSamples - s * batchSize);
        const float *d_batchImages = d_trainImages + (s * batchSize * network->layers[0].inputSize);
        const float *d_batchLabels = d_trainLabels + (s * batchSize * network->layers[network->numLayers - 1].outputSize);

        // Forward pass
        forwardNetwork(network, d_batchImages, nullptr, currentBatchSize);

        // Compute loss (without batch normalization)
        float batchLoss = computeBatchLoss(
            network->activations[network->numLayers],
            d_batchLabels,
            currentBatchSize,
            network->layers[network->numLayers - 1].outputSize);
        totalLoss += batchLoss;

        // Backward pass and update (remove batch normalization from gradients)
        backwardNetwork(network, d_batchLabels, currentBatchSize);
        updateNetworkParameters(network);

        // Free intermediate activations
        for (int i = 1; i <= network->numLayers; i++)
        {
            if (network->activations[i])
            {
                cudaFree(network->activations[i]);
                network->activations[i] = nullptr;
            }
        }
    }

    return totalLoss / stepsPerEpoch;
}