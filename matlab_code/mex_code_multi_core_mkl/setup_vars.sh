#!/bin/bash
KMP_DUPLICATE_LIB_OK=TRUE
source /opt/intel/bin/compilervars.sh intel64
export BLAS_VERSION=${MKLROOT}/lib/intel64/libmkl_rt.so
export LAPACK_VERSION=${MKLROOT}/lib/intel64/libmkl_rt.so
