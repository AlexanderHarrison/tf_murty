#ifndef MURTY_GPU
#define MURTY_GPU

#include "murty.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"

namespace tensorflow {

namespace functor {

template<>
struct MurtyFunctor<Eigen::GpuDevice> {
  void operator()(const Eigen::GpuDevice& d, const int64_t k, const int64_t n, const int64_t m, const float* in, float* out);
};

}
}

#endif
