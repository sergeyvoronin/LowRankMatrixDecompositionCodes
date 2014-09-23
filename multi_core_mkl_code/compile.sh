#!/bin/bash
#icc -mkl -openmp -fpic driver_multi_core_mkl.c low_rank_svd_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver_multi_core_mkl 
icc -mkl -openmp driver_multi_core_mkl.c low_rank_svd_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver_multi_core_mkl 
