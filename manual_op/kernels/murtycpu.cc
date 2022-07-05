#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "murty.h"
#include "../../murtycpu.h"

namespace tensorflow {

namespace functor {

template<>
struct MurtyFunctor<Eigen::ThreadPoolDevice> {
  void operator()(const Eigen::ThreadPoolDevice& d, const int64_t k, const int64_t n, const int64_t m, const float* in, float* out) {
    auto cfg = futhark_context_config_new();
    auto ctx = futhark_context_new(cfg);
    struct futhark_f32_2d* fut_in = futhark_new_f32_2d(ctx, in, m, n); 
    struct futhark_f32_1d* fut_out = futhark_new_f32_1d(ctx, out, k); 
    futhark_entry_main(ctx, &fut_out, fut_in, k);
    futhark_context_sync(ctx);
    futhark_values_f32_1d(ctx, fut_out, out);
    futhark_free_f32_2d(ctx, fut_in);
    futhark_free_f32_1d(ctx, fut_out);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
  }
};

}
}
