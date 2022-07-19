#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"

#include "murtygpu.h"
#include "murtycpu.h"

#include "murtyop.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

namespace tensorflow {

namespace functor {

template<>
struct MurtyFunctor<Eigen::ThreadPoolDevice> {
  void operator()(
          const Eigen::ThreadPoolDevice& d,
          const int64_t k,
          const int64_t n,
          const int64_t m,
          const float* in,
          int64_t* out_asgns, 
          float* out_costs
  ) {
    auto cfg = cpu_futhark_context_config_new();
    auto ctx = cpu_futhark_context_new(cfg);
    struct cpu_futhark_f32_2d* cost_mat = cpu_futhark_new_f32_2d(ctx, in, m, n); 
    struct cpu_futhark_i64_2d* least_asgns = cpu_futhark_new_i64_2d(ctx, out_asgns, m, k); 
    struct cpu_futhark_f32_1d* least_costs = cpu_futhark_new_f32_1d(ctx, out_costs, k); 

    cpu_futhark_entry_main(ctx, &least_asgns, &least_costs, cost_mat, k);
    cpu_futhark_context_sync(ctx);

    cpu_futhark_values_i64_2d(ctx, least_asgns, out_asgns);
    cpu_futhark_values_f32_1d(ctx, least_costs, out_costs);

    cpu_futhark_free_f32_2d(ctx, cost_mat);
    cpu_futhark_free_i64_2d(ctx, least_asgns);
    cpu_futhark_free_f32_1d(ctx, least_costs);

    cpu_futhark_context_free(ctx);
    cpu_futhark_context_config_free(cfg);
  }
};

template<>
struct MurtyFunctor<Eigen::GpuDevice> {
  void operator()(
          const Eigen::GpuDevice& d,
          const int64_t k,
          const int64_t n,
          const int64_t m,
          const float* in,
          int64_t* out_asgns, 
          float* out_costs
  ) {
    static auto cfg = gpu_futhark_context_config_new();
    static auto ctx = gpu_futhark_context_new(cfg);
    struct gpu_futhark_f32_2d* cost_mat = gpu_futhark_new_f32_2d(ctx, in, m, n); 
    struct gpu_futhark_i64_2d* least_asgns = gpu_futhark_new_i64_2d(ctx, out_asgns, m, k); 
    struct gpu_futhark_f32_1d* least_costs = gpu_futhark_new_f32_1d(ctx, out_costs, k); 

    gpu_futhark_entry_main(ctx, &least_asgns, &least_costs, cost_mat, k);
    gpu_futhark_context_sync(ctx);

    gpu_futhark_values_i64_2d(ctx, least_asgns, out_asgns);
    gpu_futhark_values_f32_1d(ctx, least_costs, out_costs);

    gpu_futhark_free_f32_2d(ctx, cost_mat);
    gpu_futhark_free_i64_2d(ctx, least_asgns);
    gpu_futhark_free_f32_1d(ctx, least_costs);
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

    TensorShape asgns_shape;
    asgns_shape.AddDim(k);
    asgns_shape.AddDim(m);

    TensorShape costs_shape;
    costs_shape.AddDim(k);
    
    Tensor* asgns_tensor = NULL;
    Tensor* costs_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, asgns_shape, &asgns_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, costs_shape, &costs_tensor));

    MurtyFunctor<Device>()(
        context->eigen_device<Device>(),
        k,
        n,
        m,
        static_cast<float*>(input_tensor.data()),
        static_cast<int64_t*>(asgns_tensor->data()),
        static_cast<float*>(costs_tensor->data())
    );
  }
};

REGISTER_KERNEL_BUILDER( Name("Murty").Device(DEVICE_CPU).HostMemory("k"), MurtyOp<Eigen::ThreadPoolDevice>); 
REGISTER_KERNEL_BUILDER( Name("Murty").Device(DEVICE_GPU).HostMemory("k"), MurtyOp<Eigen::GpuDevice>);
}
}

REGISTER_OP("Murty")
    .Input("costs: float32")
    .Input("k: int32")
    .Output("asgns: int64")
    .Output("asgn_costs: float32");
