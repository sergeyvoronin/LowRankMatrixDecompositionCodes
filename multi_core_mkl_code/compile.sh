#!/bin/bash

icc -O2 -mkl -openmp driver_multi_core_mkl1.c rank_revealing_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver_multi_core_mkl1
##
icc -O2 -mkl -openmp driver_multi_core_mkl2.c rank_revealing_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver_multi_core_mkl2
##
icc -O2 -mkl -openmp driver_multi_core_mkl3.c rank_revealing_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver_multi_core_mkl3
#
icc -O2 -mkl -openmp driver_multi_core_mkl4.c rank_revealing_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver_multi_core_mkl4

#icc -debug -O0 -mkl -openmp driver_multi_core_mkl3.c rank_revealing_algorithms_intel_mkl.c matrix_vector_functions_intel_mkl.c -o driver_multi_core_mkl3

