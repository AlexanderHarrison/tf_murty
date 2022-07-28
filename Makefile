SHELL := bash
TF_INCLUDE_HOME = /usr/include/tensorflow/

all: cpu_and_gpu

cpu_and_gpu: tf_flags murty.o fastmurty
	@ if [ "${CUDA_HOME}" = "" ]; then \
		echo "error: CUDA_HOME is not set. (/opt/cuda/, /usr/local/cuda/, etc. depending on your system)"; \
        exit 1; \
    fi
	g++ -D GPU -Wl,--no-undefined -I$(TF_INCLUDE_HOME) -I$(CUDA_HOME)/targets/x86_64-linux/include/ -L$(CUDA_HOME)/targets/x86_64-linux/lib/ -O3 -D NDEBUG -shared -fPIC -flto -o "libmurtyop.so" murtyop.cc murty.o -lcuda -lcudart -lnvrtc da.o queue.o sspDense.o sspSparse.o murtysplitDense.o murtysplitSparse.o subproblem.o @tf_flags

cpu: tf_flags fastmurty
	g++ -Wl,--no-undefined -I$(TF_INCLUDE_HOME) -O3 -D NDEBUG -shared -fPIC -flto -o "libmurtyop.so" murtyop.cc da.o queue.o sspDense.o sspSparse.o murtysplitDense.o murtysplitSparse.o subproblem.o @tf_flags

tf_flags: 
	@ python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()+tf.sysconfig.get_link_flags()))' > tf_flags;
	if ! [ -s tf_flags ]; then \
		echo "Error getting Tensorflow compilation flags"; rm tf_flags; exit 1; \
	fi

murty.o: murty.fut radix_sort.fut
	@ if [ "${CUDA_HOME}" = "" ]; then \
        echo "error CUDA_HOME is not set. (/opt/cuda/, /usr/local/cuda/, etc. depending on your system"; \
        exit 1; \
    fi
	futhark cuda --library murty.fut
	gcc -I$(CUDA_HOME)/targets/x86_64-linux/include/ -L$(CUDA_HOME)/targets/x86_64-linux/lib/ -c -fPIC -O3 murty.c

fastmurty: fastmurty/*.c
	gcc -D NDEBUG -c -O3 -fPIC fastmurty/subproblem.c fastmurty/queue.c fastmurty/sspDense.c fastmurty/sspSparse.c fastmurty/murtysplitDense.c fastmurty/murtysplitSparse.c fastmurty/da.c

clean:
	rm -f libmurtyop.so *.o murty.h murty.c tf_flags

.PHONY: clean fastmurty cpu_and_gpu cpu all
