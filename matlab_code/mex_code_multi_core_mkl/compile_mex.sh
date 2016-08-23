#!/bin/bash
RSVD_INCLUDE="../../multi_core_mkl_code/"
MATLAB_INC="/opt/shared/Matlab/R2015a/extern/include/"
MATLAB_LIB="/opt/shared/Matlab/R2015a/bin/glnxa64/"
MKL_LIB="/opt/shared/intel/2013_sp1/mkl/lib/intel64/" 

icc -mkl -fpic -shared -I "$MATLAB_INC" start_mkl_mex.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o start_mkl_mex.mexa64

icc -openmp -mkl -fpic -shared -I "$RSVD_INCLUDE" -I "$MATLAB_INC" low_rank_svd_rand_decomp_fixed_rank_mkl_mex.c $RSVD_INCLUDE/rank_revealing_algorithms_intel_mkl.c $RSVD_INCLUDE/matrix_vector_functions_intel_mkl.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o low_rank_svd_rand_decomp_fixed_rank_mkl_mex.mexa64



icc -openmp -mkl -fpic -shared -I "$RSVD_INCLUDE" -I "$MATLAB_INC" rsvd_mkl_mex1.c $RSVD_INCLUDE/rank_revealing_algorithms_intel_mkl.c $RSVD_INCLUDE/matrix_vector_functions_intel_mkl.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o rsvd_mkl_mex1.mexa64

icc -openmp -mkl -fpic -shared -I "$RSVD_INCLUDE" -I "$MATLAB_INC" rsvd_mkl_mex2.c $RSVD_INCLUDE/rank_revealing_algorithms_intel_mkl.c $RSVD_INCLUDE/matrix_vector_functions_intel_mkl.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o rsvd_mkl_mex2.mexa64


icc -openmp -mkl -fpic -shared -I "$RSVD_INCLUDE" -I "$MATLAB_INC" id_mkl_mex1.c $RSVD_INCLUDE/rank_revealing_algorithms_intel_mkl.c $RSVD_INCLUDE/matrix_vector_functions_intel_mkl.c -L"$MATLAB_LIB" -L"$MKL_LIB" -lmkl_rt -lmex -lmx -o id_mkl_mex1.mexa64

