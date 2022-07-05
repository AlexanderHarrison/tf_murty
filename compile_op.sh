#!/bin/bash

export CUDA_HOME="/opt/cuda/"
export CPATH="/opt/cuda/targets/x86_64-linux/include/" 
export LIBRARY_PATH="-L/opt/cuda/targets/x86_64-linux/lib/" 
export LD_LIBRARY_PATH="-L/opt/cuda/targets/x86_64-linux/lib/" 

export TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
export TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

if [ "$1" != "skip_fut" ]; then
    futhark c --library murty.fut
    mv murty.c murtycpu.c
    mv murty.h murtycpu.h
    futhark cuda --library murty.fut
    mv murty.c murtygpu.c
    mv murty.h murtygpu.h
fi

rm libmurtyop.so
gcc -O2 -Wl,--copy-dt-needed-entries,--no-as-needed -lcuda -lcudart -lnvrtc -shared -fPIC -o "libmurtyop.so" -I/usr/include/tensorflow/ -L/opt/cuda/targets/x86_64-linux/lib/ ./manual_op/kernels/murtycpu.cc ./manual_op/kernels/murtygpu.cc ./manual_op/kernels/murty.cc murtycpu.c ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
