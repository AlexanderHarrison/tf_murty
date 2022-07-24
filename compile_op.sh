#!/bin/bash

export CUDA_HOME="/opt/cuda/"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include/" 
export LIBRARY_PATH="-L/opt/cuda/targets/x86_64-linux/lib/" 
export LD_LIBRARY_PATH="-L/opt/cuda/targets/x86_64-linux/lib/" 

export TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
export TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

if [ "$1" != "skip_fut" ]; then
    ~/futhark/bin/futhark cuda --library murty.fut
    mv murty.c murtygpu.c
    mv murty.h murtygpu.h

    # Hacks to prevent name collision
    sed -i 's/futhark/gpu_futhark/g' murtygpu.c 
    sed -i 's/futhark/gpu_futhark/g' murtygpu.h
fi

cd fastmurty
gcc -D NDEBUG -c -O2 -fPIC subproblem.c queue.c sspDense.c sspSparse.c murtysplitDense.c murtysplitSparse.c da.c
cd ..

rm libmurtyop.so

export FASTMURTY_FILES="fastmurty/da.o fastmurty/queue.o fastmurty/sspDense.o fastmurty/sspSparse.o fastmurty/murtysplitDense.o fastmurty/murtysplitSparse.o fastmurty/subproblem.o "

gcc -I/usr/local/cuda-11.3/targets/x86_64-linux/include/ -L/usr/local/cuda-11.3/targets/x86_64-linux/lib/ -c -fPIC -O3 murtygpu.c
g++ -Wl,--no-undefined -I/usr/include/tensorflow/ -I/usr/local/cuda-11.3/targets/x86_64-linux/include/ -L/usr/local/cuda-11.3/targets/x86_64-linux/lib/ -O3 -D NDEBUG -shared -fPIC -flto -o "libmurtyop.so" murtyop.cc murtygpu.o $FASTMURTY_FILES -lcuda -lcudart -lnvrtc ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} 
