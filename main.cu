#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>

#include "include/neural_network.h" // Your NN interface
#include "include/data_loader.h"    // loadMNISTCSV
#include "include/utils.h"          // checkCudaError, etc.

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
        // 1) Check command line arguments
        // ------------------------------------------------
        if (argc < 3)
        {
            fprintf(stderr, "Usage: %s <train_csv> <test_csv>\n", argv[0]);
            return 1;
        }
        std::string trainCSV = argv[1];
        std::string testCSV = argv[2];

        // ------------------------------------------------
        // 2) Load training data
        // ------------------------------------------------
        float *d_trainImages = nullptr;
        float *d_trainLabels = nullptr;
        int trainSamples = 0;

        loadMNISTCSV(trainCSV, &d_trainImages, &d_trainLabels, trainSamples);
        printf("[main] Training set: %d samples loaded.\n", trainSamples);

        // ------------------------------------------------
        // 3) Load test data
        // ------------------------------------------------
        float *d_testImages = nullptr;
        float *d_testLabels = nullptr;
        int testSamples = 0;

        loadMNISTCSV(testCSV, &d_testImages, &d_testLabels, testSamples);
        printf("[main] Test set: %d samples loaded.\n", testSamples);

        // ------------------------------------------------
        // 4) Define network architecture
        // ------------------------------------------------
        int layerSizes[] = {784, 512, 10};
        int numLayers = 3;
        float learningRate = 0.001f;

        NeuralNetwork network;
        initNetwork(&network, layerSizes, numLayers, learningRate);

        // Training parameters
        int batchSize = 32;
        int epochs = 20;
        float decayRate = 0.95f;
        int decaySteps = 5; // Decay learning rate every 5 epochs

        printf("[main] Starting training for %d epochs, batchSize=%d...\n",
               epochs, batchSize);

        // ------------------------------------------------
        // 5) Train network
        // ------------------------------------------------
        float bestTrainAccuracy = 0.0f;
        float bestTestAccuracy = 0.0f;

        for (int e = 0; e < epochs; e++)
        {
            // Apply learning rate decay
            if (e > 0 && e % decaySteps == 0)
            {
                network.learningRate *= decayRate;
                printf("[main] Learning rate decayed to %f\n", network.learningRate);
            }

            // Train one epoch
            float epochLoss = trainNetwork(&network,
                                           d_trainImages,
                                           d_trainLabels,
                                           trainSamples,
                                           batchSize);

            // Evaluate on training set
            float trainAcc = evaluateAccuracy(&network,
                                              d_trainImages,
                                              d_trainLabels,
                                              trainSamples,
                                              batchSize);

            // Evaluate on test set
            float testAcc = evaluateAccuracy(&network,
                                             d_testImages,
                                             d_testLabels,
                                             testSamples,
                                             batchSize);

            bestTrainAccuracy = fmaxf(bestTrainAccuracy, trainAcc);
            bestTestAccuracy = fmaxf(bestTestAccuracy, testAcc);

            printf("Epoch %d: Loss=%.4f, Train Acc=%.2f%% (Best=%.2f%%), Test Acc=%.2f%% (Best=%.2f%%)\n",
                   e, epochLoss, trainAcc, bestTrainAccuracy, testAcc, bestTestAccuracy);
        }

        // ------------------------------------------------
        // 6) Final evaluation
        // ------------------------------------------------
        printf("\nTraining completed. Final evaluation:\n");
        printf("----------------------------------------\n");

        float finalTrainAcc = evaluateAccuracy(&network,
                                               d_trainImages,
                                               d_trainLabels,
                                               trainSamples,
                                               batchSize);

        float finalTestAcc = evaluateAccuracy(&network,
                                              d_testImages,
                                              d_testLabels,
                                              testSamples,
                                              batchSize);

        printf("Final Training Accuracy: %.2f%%\n", finalTrainAcc);
        printf("Final Test Accuracy: %.2f%%\n", finalTestAcc);
        printf("Best Training Accuracy: %.2f%%\n", bestTrainAccuracy);
        printf("Best Test Accuracy: %.2f%%\n", bestTestAccuracy);

        // ------------------------------------------------
        // 7) Cleanup
        // ------------------------------------------------
        cudaFree(d_trainImages);
        cudaFree(d_trainLabels);
        cudaFree(d_testImages);
        cudaFree(d_testLabels);
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