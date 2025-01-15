#!/usr/bin/env python3
import subprocess
import os

def main():
    # Paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(project_dir, "data")
    src_dir     = os.path.join(project_dir, "src")
    mnist_csv   = os.path.join(data_dir, "mnist_test.csv")  # or "mnist_train.csv"

    if not os.path.exists(mnist_csv):
        raise FileNotFoundError(
            f"Could not find MNIST CSV at {mnist_csv}. "
             "Did you run downloader.py first?"
        )

    # 1) Compile: neural_network.cu, data_loader.cu, main.cu
    compile_cmd = [
        "nvcc",
        os.path.join(src_dir, "neural_network.cu"),
        os.path.join(src_dir, "data_loader.cu"),
        os.path.join(src_dir, "main.cu"),
        "-o",
        os.path.join(project_dir, "my_net"),
        "-I", os.path.join(project_dir, "include"),  # so we find neural_network.h, utils.h
        "--std=c++11",
    ]
    print("Compiling with NVCC:\n", " ".join(compile_cmd))
    subprocess.check_call(compile_cmd)

    # 2) Run the executable with the CSV file
    run_cmd = [
        os.path.join(project_dir, "my_net"),
        mnist_csv
    ]
    print("\nRunning forward pass:\n", " ".join(run_cmd))
    subprocess.check_call(run_cmd)

if __name__ == "__main__":
    main()
