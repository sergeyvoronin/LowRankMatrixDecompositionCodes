#!/bin/bash
KMP_DUPLICATE_LIB_OK=TRUE
source /opt/shared/intel/2013_sp1/bin/compilervars.sh intel64
export BLAS_VERSION=${MKLROOT}/lib/intel64/libmkl_rt.so
export LAPACK_VERSION=${MKLROOT}/lib/intel64/libmkl_rt.so
