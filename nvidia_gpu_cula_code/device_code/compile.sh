#!/bin/bash


nvcc rsvd_code_cula_device.c -o rsvd_code_cula_device -Xcompiler -fopenmp -lmkl_rt -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH

