#ifndef DATA_LOADER_H
#define DATA_LOADER_H
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

void loadMNISTCSV(
    const std::string &csvPath,
    float **d_images,
    float **d_labels,
    int &numSamples);