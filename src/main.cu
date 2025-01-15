#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "include/neural_network.h" // your header for the network interface
#include "include/utils.h"          // assumed utilities (checkCudaError, etc.)
#include "include/data_loader.h"

/**
 * A small helper to print a float array from host memory
 */
void printArray(const float *arr, int size, const char *name)
{
    printf("%s = [", name);
    for (int i = 0; i < size; i++)
    {
        printf("%.4f", arr[i]);
        if (i < size - 1)
            printf(", ");
    }
    printf("]\n");
}

int main()
{
    // ------------------------------------------------
    // 1) Prepare some dummy data for XOR learning
    // ------------------------------------------------
    // XOR truth table (4 samples)
    //   Inputs (2D):
    //      (0,0) -> 0
    //      (0,1) -> 1
    //      (1,0) -> 1
    //      (1,1) -> 0
    float h_trainInputs[8] = {0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f};
    float h_trainLabels[4] = {0.f, 1.f, 1.f, 0.f};
    const int numSamples = 4;
    const int inputSize = 2;  // each sample has 2 inputs
    const int outputSize = 1; // we want 1 output (for XOR)

    // Copy them to device
    float *d_inputs = nullptr, *d_labels = nullptr;
    checkCudaError(cudaMalloc(&d_inputs, numSamples * inputSize * sizeof(float)), "cudaMalloc d_inputs");
    checkCudaError(cudaMalloc(&d_labels, numSamples * outputSize * sizeof(float)), "cudaMalloc d_labels");
    checkCudaError(cudaMemcpy(d_inputs, h_trainInputs, numSamples * inputSize * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy h_trainInputs -> d_inputs");
    checkCudaError(cudaMemcpy(d_labels, h_trainLabels, numSamples * outputSize * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy h_trainLabels -> d_labels");

    // ------------------------------------------------
    // 2) Define a small network architecture
    // ------------------------------------------------
    // For example, 2 -> 4 -> 1
    // layerSizes[0] = inputSize=2
    // layerSizes[1] = hiddenSize=4
    // layerSizes[2] = outputSize=1
    int layerSizes[] = {2, 4, 1};
    const int numLayers = 3; // This includes input layer, hidden layer(s), and output layer
    float learningRate = 0.1f;

    NeuralNetwork network;
    initNetwork(&network, layerSizes, numLayers, learningRate);

    // ------------------------------------------------
    // 3) Train the network
    // ------------------------------------------------
    // We'll train in a single batch of size=4 (entire XOR set).
    // For real tasks, you'd typically have many batches, etc.
    int batchSize = 4;
    int epochs = 1000;

    printf("\n[INFO] Training the network on XOR...\n");
    trainNetwork(&network, d_inputs, d_labels, numSamples, batchSize, epochs);

    // ------------------------------------------------
    // 4) Test the trained network (forward pass)
    // ------------------------------------------------
    // We re-run the same 4 inputs (d_inputs), see if we get close to the XOR truth
    //  (which is {0,1,1,0}).

    // Create a device buffer for the outputs
    float *d_output = nullptr;
    checkCudaError(cudaMalloc(&d_output, batchSize * outputSize * sizeof(float)),
                   "cudaMalloc d_output");

    // Forward pass
    forwardNetwork(&network, d_inputs, d_output, batchSize);

    // Copy the outputs back to host
    float h_output[4];
    checkCudaError(cudaMemcpy(h_output, d_output, batchSize * outputSize * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy d_output -> h_output");

    // Print the results
    printf("\nTrained Network Output (after %d epochs):\n", epochs);
    for (int i = 0; i < numSamples; i++)
    {
        printf("Input = (%.1f, %.1f)  ->  Pred = %.4f  (Target = %.1f)\n",
               h_trainInputs[2 * i], h_trainInputs[2 * i + 1],
               h_output[i],
               h_trainLabels[i]);
    }

    // ------------------------------------------------
    // 5) Cleanup
    // ------------------------------------------------
    cudaFree(d_inputs);
    cudaFree(d_labels);
    cudaFree(d_output);

    freeNetwork(&network);
    cudaDeviceReset();
    return 0;
}
