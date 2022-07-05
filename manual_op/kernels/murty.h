#ifndef Murty_H_
#define Murty_H_

#include <unsupported/Eigen/CXX11/Tensor>

namespace tensorflow {

namespace functor {

template <typename Device>
struct MurtyFunctor {
  void operator()(const Device& d, int64_t k, int64_t n, int64_t m, const float* in, float* out);
};

}

}

#endif
