#!/bin/bash

#nvcc driver_gpu_nvidia_cula.c low_rank_svd_algorithms_nvidia_cula.c matrix_vector_functions_nvidia_cula.c -o driver_gpu_nvidia_cula -Xcompiler -fopenmp -fgomp -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH

nvcc driver_gpu_nvidia_cula1.c rank_revealing_algorithms_nvidia_cula.c matrix_vector_functions_nvidia_cula.c -o driver_gpu_nvidia_cula1 -Xcompiler -fopenmp -lmkl_rt -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH

nvcc driver_gpu_nvidia_cula2.c rank_revealing_algorithms_nvidia_cula.c matrix_vector_functions_nvidia_cula.c -o driver_gpu_nvidia_cula2 -Xcompiler -fopenmp -lmkl_rt -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH

nvcc driver_gpu_nvidia_cula3.c rank_revealing_algorithms_nvidia_cula.c matrix_vector_functions_nvidia_cula.c -o driver_gpu_nvidia_cula3 -Xcompiler -fopenmp -lmkl_rt -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH

nvcc driver_gpu_nvidia_cula4.c rank_revealing_algorithms_nvidia_cula.c matrix_vector_functions_nvidia_cula.c -o driver_gpu_nvidia_cula4 -Xcompiler -fopenmp -lmkl_rt -lcula_lapack -lcublas -lcudart -liomp5 -L$CULA_LIB_PATH_64 -I$CULA_INC_PATH
