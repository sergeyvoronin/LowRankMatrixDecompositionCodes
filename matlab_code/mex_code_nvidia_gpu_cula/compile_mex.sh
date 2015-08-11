#!/bin/bash
RSVD_INCLUDE="../../nvidia_gpu_cula_code/"

nvcc -Xcompiler -fPIC -Xcompiler -fopenmp -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH -shared -I "$RSVD_INCLUDE" -I "/usr/local/MATLAB/R2015a/extern/include/" rsvd_cula_mex1.c $RSVD_INCLUDE/rank_revealing_algorithms_nvidia_cula.c $RSVD_INCLUDE/matrix_vector_functions_nvidia_cula.c -L"/usr/local/MATLAB/R2015a/bin/glnxa64" -lmex -lmx -o rsvd_cula_mex1.mexa64

