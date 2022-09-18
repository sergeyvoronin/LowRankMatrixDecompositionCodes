#!/bin/bash
export MKLROOT=/opt/intel/oneapi/mkl/2022.0.2/
export PATH=$PATH:/opt/intel/oneapi/compiler/2022.0.2/linux/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/mkl/2022.0.2/lib/intel64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64/
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/opt/intel/oneapi/mkl/2022.0.2/include/
