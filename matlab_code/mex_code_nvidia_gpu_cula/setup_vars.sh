#!/bin/bash
KMP_DUPLICATE_LIB_OK=TRUE

export PATH=$PATH:/usr/local/cuda-6.0/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-6.0/lib64/

export CULA_ROOT="/usr/local/cula"
export CULA_INC_PATH="$CULA_ROOT/include"
export CULA_LIB_PATH_32="$CULA_ROOT/lib"
export CULA_LIB_PATH_64="$CULA_ROOT/lib64"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CULA_LIB_PATH_64

#export BLAS_VERSION=${MKLROOT}/lib/intel64/libmkl_rt.so
#export LAPACK_VERSION=${MKLROOT}/lib/intel64/libmkl_rt.so
