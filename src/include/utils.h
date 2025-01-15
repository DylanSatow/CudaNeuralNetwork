#ifndef UTILS_H
#define UTILS_H
#endif

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

static void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

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
