#!/bin/bash
RSVD_INCLUDE="../../multi_core_mkl_code/"

icc -mkl -fpic -shared -I "/usr/local/MATLAB/R2015a/extern/include/" start_mkl_mex.c -L"/usr/local/MATLAB/R2015a/bin/glnxa64" -L"/opt/intel/mkl/lib/intel64" -lmkl_rt -lmex -lmx -o start_mkl_mex.mexa64

icc -openmp -mkl -fpic -shared -I "$RSVD_INCLUDE" -I "/usr/local/MATLAB/R2015a/extern/include/" rsvd_mkl_mex1.c $RSVD_INCLUDE/rank_revealing_algorithms_intel_mkl.c $RSVD_INCLUDE/matrix_vector_functions_intel_mkl.c -L"/usr/local/MATLAB/R2015a/bin/glnxa64" -L"/opt/intel/mkl/lib/intel64" -lmkl_rt -lmex -lmx -o rsvd_mkl_mex1.mexa64

