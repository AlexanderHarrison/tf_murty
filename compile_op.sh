#!/bin/bash

export CUDA_HOME="/opt/cuda/"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include/" 
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

    # Hacks to prevent name collision
    sed -i 's/futhark/gpu_futhark/g' murtygpu.c 
    sed -i 's/futhark/gpu_futhark/g' murtygpu.h
    sed -i 's/futhark/cpu_futhark/g' murtycpu.c
    sed -i 's/futhark/cpu_futhark/g' murtycpu.h
fi

rm libmurtyop.so

gcc -O3 -lcuda -lcudart -lnvrtc -shared -fPIC -flto -o "libmurtyop.so" -I/usr/include/tensorflow/ -L/opt/cuda/targets/x86_64-linux/lib/ murtygpu.c murtycpu.c murtyop.cc ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
