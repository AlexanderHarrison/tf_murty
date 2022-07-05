#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "murty.h"
#include "../../murtygpu.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"

namespace tensorflow {

namespace functor {

template<>
struct MurtyFunctor<Eigen::GpuDevice> {
  void operator()(const Eigen::GpuDevice& d, const int64_t k, const int64_t n, const int64_t m, const float* in, float* out) {
    static auto ctx = [](){
        struct futhark_context_config* cfg = futhark_context_config_new();
        return futhark_context_new(cfg);
    } ();
    struct futhark_f32_2d* fut_in = futhark_new_f32_2d(ctx, in, n, m); 
    struct futhark_f32_1d* fut_out = futhark_new_f32_1d(ctx, out, k); 
    futhark_entry_main(ctx, &fut_out, fut_in, k);
  }
};

}
}
