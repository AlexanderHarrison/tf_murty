#ifndef Murtyop_H_
#define Murtyop_H_

#include <unsupported/Eigen/CXX11/Tensor>

namespace tensorflow {

namespace functor {

template <typename Device>
struct MurtyFunctor {
  void operator()(
          const Device& d,
          const int64_t k,
          const int64_t n,
          const int64_t m,
          double* in,
          int* out_asgns, 
          double* out_costs
  );
};

}

}

#endif
