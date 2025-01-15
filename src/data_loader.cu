#include "include/data_loader.h"

/**
 * \brief Loads MNIST data from a CSV file and copies it to GPU memory.
 *
 * \param csvPath       Path to the MNIST CSV file (e.g., "data/mnist_train.csv").
 * \param d_images      (output) Pointer to device memory for image pixels (size = numSamples * 784).
 * \param d_labels      (output) Pointer to device memory for labels (size = numSamples).
 * \param numSamples    (output) Number of rows (samples) read from the CSV file.
 *
 * \note  Expects the CSV format:
 *          label, pixel_0, pixel_1, ..., pixel_783
 *        Skips the first header line if present.
 */
void loadMNISTCSV(
    const std::string &csvPath,
    float **d_images,
    float **d_labels,
    int &numSamples)
{
    std::ifstream file(csvPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + csvPath);
    }

    std::string line;
    std::vector<float> host_images; // Will hold (numSamples * 784) floats
    std::vector<float> host_labels; // Will hold numSamples floats

    bool isHeader = true;
    numSamples = 0;

    while (std::getline(file, line))
    {
        // If the first line is a header, skip it
        if (isHeader)
        {
            isHeader = false;
            // Check if we want to confirm it is indeed a header.
            // For simplicity, we just skip the first line:
            continue;
        }

        // Parse CSV line: label, pixel_0, ..., pixel_783
        std::stringstream ss(line);
        std::string token;

        // 1) Parse label
        if (!std::getline(ss, token, ','))
        {
            // If we failed to read the label, skip line or throw error
            continue;
        }
        float labelValue = std::stof(token);
        int digit = static_cast<int>(labelValue);

        // expand to one-hot of length 10
        for (int c = 0; c < 10; c++)
        {
            host_labels.push_back((c == digit) ? 1.f : 0.f);
        }

        // 2) Parse 784 pixels
        for (int i = 0; i < 784; i++)
        {
            if (!std::getline(ss, token, ','))
            {
                throw std::runtime_error("Invalid CSV format: expected 784 pixel values, got fewer.");
            }
            float pixelValue = std::stof(token);
            // Normalize to range [0,1] - MNIST pixels are originally [0,255]
            pixelValue /= 255.0f;
            host_images.push_back(pixelValue);
        }

        numSamples++;
    }

    file.close();

    std::cout << "[loadMNISTCSV] Loaded " << numSamples
              << " samples from " << csvPath << std::endl;

    // Allocate device memory
    //   Images: numSamples * 784
    //   Labels: numSamples
    size_t imagesSize = numSamples * 784 * sizeof(float);
    size_t labelsSize = numSamples * 10 * sizeof(float); // NEW: 10 floats/label
    
    cudaError_t err;
    err = cudaMalloc((void **)d_images, imagesSize);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaMalloc failed for d_images");
    }

    err = cudaMalloc((void **)d_labels, labelsSize);
    if (err != cudaSuccess)
    {
        cudaFree(*d_images);
        throw std::runtime_error("cudaMalloc failed for d_labels");
    }

    // Copy data from host to device
    err = cudaMemcpy(*d_images, host_images.data(), imagesSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(*d_images);
        cudaFree(*d_labels);
        throw std::runtime_error("cudaMemcpy failed for d_images");
    }

    err = cudaMemcpy(*d_labels, host_labels.data(), labelsSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(*d_images);
        cudaFree(*d_labels);
        throw std::runtime_error("cudaMemcpy failed for d_labels");
    }

    std::cout << "[loadMNISTCSV] Copied data to GPU memory." << std::endl;
}
