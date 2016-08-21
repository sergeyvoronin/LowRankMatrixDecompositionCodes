#!/bin/bash


nvcc driver_mkl_and_cublas1.c rank_revealing_algorithms_mkl_and_cublas.c  matrix_vector_functions_mkl_and_cublas.c -o driver_mkl_and_cublas1 -Xcompiler -fopenmp -lmkl_rt -lcublas -lcudart -liomp5 


