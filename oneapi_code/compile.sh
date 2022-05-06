#!/bin/bash
export MKLROOT=/opt/intel/oneapi/mkl/2022.0.2/

icx -m64 -mkl -openmp -I"${MKLROOT}/include" driver1.c matrix_vector_functions_intel_mkl.c rank_revealing_algorithms_intel_mkl.c -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -o run1
