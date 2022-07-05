#ifndef MURTY_CPU
#define MURTY_CPU

#include "murty.h"

namespace tensorflow {

namespace functor {

template<>
struct MurtyFunctor<Eigen::ThreadPoolDevice> {
  void operator()(const Eigen::ThreadPoolDevice& d, const int64_t k, const int64_t n, const int64_t m, const float* in, float* out);
};

}
}

#endif
