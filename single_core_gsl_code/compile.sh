#!/bin/bash

#gcc -O3 driver_single_core_gsl.c matrix_vector_functions_gsl.c low_rank_svd_algorithms_gsl.c  -o driver_single_core_gsl -lgsl -lgslcblas -lm -w
gcc -O3 driver_single_core_gsl.c matrix_vector_functions_gsl.c low_rank_svd_algorithms_gsl.c  -o driver_single_core_gsl -lgsl -lgslcblas -lm 

