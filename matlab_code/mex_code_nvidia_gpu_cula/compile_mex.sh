#!/bin/bash
RSVD_INCLUDE="../../nvidia_gpu_cula_code/"
MATLAB_INC="/opt/shared/Matlab/R2015a/extern/include/"
MATLAB_LIB="/opt/shared/Matlab/R2015a/bin/glnxa64/"
MKL_LIB="/opt/shared/intel/2013_sp1/mkl/lib/intel64/" 


nvcc -Xcompiler -fPIC -Xcompiler -fopenmp -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH -shared -I "$RSVD_INCLUDE" -I "$MATLAB_INC" low_rank_svd_rand_decomp_fixed_rank_cula_mex.c $RSVD_INCLUDE/rank_revealing_algorithms_nvidia_cula.c $RSVD_INCLUDE/matrix_vector_functions_nvidia_cula.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o low_rank_svd_rand_decomp_fixed_rank_cula_mex.mexa64


nvcc -Xcompiler -fPIC -Xcompiler -fopenmp -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH -shared -I "$RSVD_INCLUDE" -I "$MATLAB_INC" rsvd_cula_mex1.c $RSVD_INCLUDE/rank_revealing_algorithms_nvidia_cula.c $RSVD_INCLUDE/matrix_vector_functions_nvidia_cula.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o rsvd_cula_mex1.mexa64


nvcc -Xcompiler -fPIC -Xcompiler -fopenmp -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH -shared -I "$RSVD_INCLUDE" -I "$MATLAB_INC" rsvd_cula_mex2.c $RSVD_INCLUDE/rank_revealing_algorithms_nvidia_cula.c $RSVD_INCLUDE/matrix_vector_functions_nvidia_cula.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o rsvd_cula_mex2.mexa64

