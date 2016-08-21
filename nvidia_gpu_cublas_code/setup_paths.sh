#!/bin/bash

export PATH=$PATH:/usr/local/cuda/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

#export CULA_ROOT="/cluster/home/svoron01/apps/cula/"
##export CULA_ROOT="/opt/shared/cula/dense-r15/"
#export CULA_INC_PATH="$CULA_ROOT/include"
#export CULA_LIB_PATH_32="$CULA_ROOT/lib"
#export CULA_LIB_PATH_64="$CULA_ROOT/lib64"
#
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CULA_LIB_PATH_64
#
#source /opt/intel/bin/iccvars.sh intel64
source /opt/shared/intel/2013_sp1/bin/iccvars.sh intel64
