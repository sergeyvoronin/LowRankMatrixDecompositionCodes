#!/bin/bash
export MKLROOT=/opt/intel/oneapi/mkl/2022.0.2/

icx -m64 -mkl -openmp -I"${MKLROOT}/include" -I"${MKLROOT}/examples/c/blas/source/" driver1.c matrix_vector_functions_one_api.c rank_revealing_algorithms_one_api.c -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -o rund1

icx -m64 -mkl -openmp -I"${MKLROOT}/include" -I"${MKLROOT}/examples/c/blas/source/" driver2.c matrix_vector_functions_one_api.c rank_revealing_algorithms_one_api.c -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -o rund2

icx -m64 -mkl -openmp -I"${MKLROOT}/include" -I"${MKLROOT}/examples/c/blas/source/" driver3.c matrix_vector_functions_one_api.c rank_revealing_algorithms_one_api.c -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -o rund3
