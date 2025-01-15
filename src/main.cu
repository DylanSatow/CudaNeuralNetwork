// main.cu
#include <iostream>
#include "include/neural_network.h"
#include "include/utils.h"

extern void loadMNISTCSV(const std::string &csvPath,
                         float **d_images,
                         float **d_labels,
                         int &numSamples);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <mnist_csv_path>" << std::endl;
        return 1;
    }
    std::string csvPath = argv[1];

    // 1) Load MNIST CSV
    float *d_images = nullptr;
    float *d_labels = nullptr;
    int numSamples = 0;

    loadMNISTCSV(csvPath, &d_images, &d_labels, numSamples);

    std::cout << "MNIST loaded. numSamples = " << numSamples << std::endl;

    // 2) Build a small network, e.g. [784 -> 128 -> 10]
    int layerSizes[3] = {784, 128, 10};
    NeuralNetwork net;
    initNetwork(&net, layerSizes, 3, /*learningRate=*/0.01f);

    // 3) Forward pass for a small batch, say batchSize = 4 for quick debug
    int batchSize = 4;
    if (batchSize > numSamples)
        batchSize = numSamples; // safeguard

    // Allocate device memory for the output of shape (batchSize x 10)
    float *d_output = nullptr;
    cudaMalloc(&d_output, batchSize * layerSizes[2] * sizeof(float));

    // Forward pass: d_images -> net -> d_output
    forwardNetwork(&net, d_images, d_output, batchSize);

    // 4) Copy output to host & print first few logits
    float *h_output = (float *)malloc(batchSize * layerSizes[2] * sizeof(float));
    cudaMemcpy(h_output, d_output, batchSize * layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);

    // Print partial results
    for (int b = 0; b < batchSize; b++)
    {
        std::cout << "Sample #" << b << " output logits: ";
        for (int j = 0; j < layerSizes[2]; j++)
        {
            std::cout << h_output[b * layerSizes[2] + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    freeNetwork(&net);
    cudaFree(d_output);
    free(h_output);
    cudaFree(d_images);
    cudaFree(d_labels);

    return 0;
}
