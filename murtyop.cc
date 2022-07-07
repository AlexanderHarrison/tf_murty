#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"

#include "../../murtygpu.h"
#include "../../murtycpu.h"

#include "murtyop.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

namespace tensorflow {

namespace functor {

template<>
struct MurtyFunctor<Eigen::ThreadPoolDevice> {
  void operator()(const Eigen::ThreadPoolDevice& d, const int64_t k, const int64_t n, const int64_t m, const float* in, float* out) {
    auto cfg = cpu_futhark_context_config_new();
    auto ctx = cpu_futhark_context_new(cfg);
    struct cpu_futhark_f32_2d* fut_in = cpu_futhark_new_f32_2d(ctx, in, m, n); 
    struct cpu_futhark_f32_1d* fut_out = cpu_futhark_new_f32_1d(ctx, out, k); 
    cpu_futhark_entry_main(ctx, &fut_out, fut_in, k);
    cpu_futhark_context_sync(ctx);
    cpu_futhark_values_f32_1d(ctx, fut_out, out);
    cpu_futhark_free_f32_2d(ctx, fut_in);
    cpu_futhark_free_f32_1d(ctx, fut_out);
    cpu_futhark_context_free(ctx);
    cpu_futhark_context_config_free(cfg);
  }
};

template<>
struct MurtyFunctor<Eigen::GpuDevice> {
  void operator()(const Eigen::GpuDevice& d, const int64_t k, const int64_t n, const int64_t m, const float* in, float* out) {
    auto cfg = gpu_futhark_context_config_new();
    auto ctx = gpu_futhark_context_new(cfg);
    struct gpu_futhark_f32_2d* fut_in = gpu_futhark_new_f32_2d(ctx, in, m, n); 
    struct gpu_futhark_f32_1d* fut_out = gpu_futhark_new_f32_1d(ctx, out, k); 
    gpu_futhark_entry_main(ctx, &fut_out, fut_in, k);
    gpu_futhark_context_sync(ctx);
    gpu_futhark_values_f32_1d(ctx, fut_out, out);
    gpu_futhark_free_f32_2d(ctx, fut_in);
    gpu_futhark_free_f32_1d(ctx, fut_out);
    gpu_futhark_context_free(ctx);
    gpu_futhark_context_config_free(cfg);
  }
};

template <typename Device>
class MurtyOp : public OpKernel {
 public:
  explicit MurtyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    int k = static_cast<int*>(context->input(1).data())[0];
    
    TensorShape in_shape = input_tensor.shape();
    int64_t m = in_shape.dim_size(0);
    int64_t n = in_shape.dim_size(1);

    TensorShape out_shape = input_tensor.shape();
    out_shape.RemoveLastDims(2);
    out_shape.AddDim(k); // EWWWWWW
    
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &output_tensor));

    MurtyFunctor<Device>()(
        context->eigen_device<Device>(),
        k,
        n,
        m,
        static_cast<float*>(input_tensor.data()),
        static_cast<float*>(output_tensor->data())
    );
  }
};

REGISTER_KERNEL_BUILDER( Name("Murty").Device(DEVICE_CPU), MurtyOp<Eigen::ThreadPoolDevice>); 
REGISTER_KERNEL_BUILDER( Name("Murty").Device(DEVICE_GPU), MurtyOp<Eigen::GpuDevice>);
}
}

REGISTER_OP("Murty")
    .Input("costs: float32")
    .Input("k: int32")
    .Output("mincost: float32");
