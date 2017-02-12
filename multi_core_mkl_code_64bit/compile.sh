#!/bin/bash

icc -O2 -mkl -openmp driver1.c rank_revealing_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver1

