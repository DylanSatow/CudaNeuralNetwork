#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>

#include "include/neural_network.h" // Your NN interface
#include "include/data_loader.h"    // loadMNISTCSV
#include "include/utils.h"          // checkCudaError, etc.

// Optional helper: evaluate accuracy of a batch.
//   - For each sample, we do a forward pass, find the argmax output,
//     compare to integer label.
//   - Because we have a single batch for the entire dataset here, we can
//     do it all in one call. Otherwise you'd batch this as well.
float evaluateAccuracy(NeuralNetwork *network,
                       const float *d_images,
                       const float *d_labels,
                       int numSamples,
                       int batchSize)
{
    // Buffer on GPU for the final output of shape (batchSize, 10)
    float *d_output = nullptr;
    checkCudaError(cudaMalloc(&d_output, batchSize * network->layers[network->numLayers - 1].outputSize * sizeof(float)),
                   "cudaMalloc d_output");

    // Allocate host buffer dynamically instead of stack allocation
    float *h_output = (float *)malloc(batchSize * 10 * sizeof(float));
    if (!h_output)
    {
        cudaFree(d_output);
        throw std::runtime_error("Failed to allocate host output buffer");
    }

    int correctCount = 0;
    int totalBatches = (numSamples + batchSize - 1) / batchSize;

    for (int b = 0; b < totalBatches; b++)
    {
        int start = b * batchSize;
        int thisBatch = (start + batchSize <= numSamples)
                            ? batchSize
                            : (numSamples - start);

        if (thisBatch <= 0)
            break;

        const float *d_batchImages = d_images + (start * 784);
        const float *d_batchLabels = d_labels + (start * 10); // Note: labels are now one-hot encoded

        // Forward pass on this batch
        forwardNetwork(network, d_batchImages, d_output, thisBatch);

        // Copy back the results
        checkCudaError(cudaMemcpy(h_output,
                                  d_output,
                                  thisBatch * 10 * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy d_output->h_output for evaluateAccuracy");

        // Evaluate
        for (int i = 0; i < thisBatch; i++)
        {
            // find argmax of the 10 outputs
            int offset = i * 10;
            float maxVal = h_output[offset];
            int maxIdx = 0;
            for (int c = 1; c < 10; c++)
            {
                if (h_output[offset + c] > maxVal)
                {
                    maxVal = h_output[offset + c];
                    maxIdx = c;
                }
            }

            // Compare to the one-hot encoded label
            float label[10];
            checkCudaError(cudaMemcpy(label,
                                      d_batchLabels + (i * 10),
                                      10 * sizeof(float),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy label in evaluateAccuracy");

            // Find the position of 1 in the one-hot encoded label
            int labelIdx = 0;
            for (int j = 0; j < 10; j++)
            {
                if (label[j] > 0.5f)
                {
                    labelIdx = j;
                    break;
                }
            }

            if (maxIdx == labelIdx)
            {
                correctCount++;
            }
        }
    }

    // Clean up
    free(h_output);
    cudaFree(d_output);

    return 100.0f * (float)correctCount / (float)numSamples;
}

int main(int argc, char **argv)
{
    try
    {
        // ------------------------------------------------
        // 1) Load training data from a CSV file
        // ------------------------------------------------
        // Example usage:  ./mnist_train data/mnist_train.csv
        // Adjust path as needed.
        if (argc < 2)
        {
            fprintf(stderr, "Usage: %s <train_csv>\n", argv[0]);
            return 1;
        }
        std::string trainCSV = argv[1];

        float *d_trainImages = nullptr;
        float *d_trainLabels = nullptr;
        int trainSamples = 0;

        loadMNISTCSV(trainCSV, &d_trainImages, &d_trainLabels, trainSamples);
        printf("[main] Training set: %d samples loaded.\n", trainSamples);

        // ------------------------------------------------
        // 2) Define a simple 3-layer network
        //    (784 -> 128 -> 10)
        // ------------------------------------------------
        int layerSizes[] = {784, 128, 10};
        int numLayers = 3; // includes input, hidden, output
        float learningRate = 0.01f;

        NeuralNetwork network;
        initNetwork(&network, layerSizes, numLayers, learningRate);

        // ------------------------------------------------
        // 3) Train: pick a batch size, # epochs
        // ------------------------------------------------
        // For simplicity, let's do an entire epoch in
        // 'steps' of batchSize
        int batchSize = 128;
        int epochs = 30;

        printf("[main] Starting training for %d epochs, batchSize=%d...\n", epochs, batchSize);
        trainNetwork(&network,
                     d_trainImages,
                     d_trainLabels,
                     trainSamples,
                     batchSize,
                     epochs);

        // ------------------------------------------------
        // 4) Evaluate on the same training set (for demo)
        //    If you have a separate MNIST test CSV, call
        //    loadMNISTCSV(...) for test data and evaluate
        //    that as well.
        // ------------------------------------------------
        float trainAcc = evaluateAccuracy(&network,
                                          d_trainImages,
                                          d_trainLabels,
                                          trainSamples,
                                          batchSize);
        printf("[main] Training accuracy = %.2f%%\n", trainAcc);

        // ------------------------------------------------
        // 5) Cleanup
        // ------------------------------------------------
        cudaFree(d_trainImages);
        cudaFree(d_trainLabels);
        freeNetwork(&network);

        // Reset device
        cudaDeviceReset();
        return 0;
    }
    catch (const std::exception &ex)
    {
        fprintf(stderr, "Exception in main(): %s\n", ex.what());
        return 1;
    }
}
