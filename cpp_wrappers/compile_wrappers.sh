#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd cpp_neighbors
python3 setup.py build_ext --inplace
cd ..

# Compile cpp preprocess
cd cpp_preprocess
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) propagation.cpp -o propagation$(python3-config --extension-suffix) -larmadillo -lmlpack -fopenmp