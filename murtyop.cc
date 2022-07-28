#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"

#ifdef GPU
#include "murty.h"
#endif

#include "murtyop.h"

extern "C" {
#include "fastmurty/da.h"
}

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

namespace tensorflow {

namespace functor {

template <> 
struct MurtyFunctor<Eigen::ThreadPoolDevice> {
    void operator()(const Eigen::ThreadPoolDevice &d, const int64_t k,
                    const int64_t n, const int64_t m, double *in,
                    int *out_asgns, double *out_costs) {
	thread_local WorkvarsforDA workvars =
            allocateWorkvarsforDA(m, n, k);
	if (workvars.m != m || workvars.n != n) {
            deallocateWorkvarsforDA(workvars);
	    workvars = allocateWorkvarsforDA(m, n, k);
	}

        static bool* priors = [](){
            bool* arr = new bool[256];
            for (int i = 0; i < 256; ++i)
                arr[i] = true;
            return arr;
        }();

        static double prior_weights = 0;

        da(in, 1, priors, &prior_weights,
            1, priors, &prior_weights, k,
            out_asgns, out_costs, &workvars);
    }
};

#ifdef GPU 
template <> struct MurtyFunctor<Eigen::GpuDevice> {
    void operator()(const Eigen::GpuDevice &d, const int64_t k, const int64_t n,
                    const int64_t m, const double *in, int *out_asgns,
                    double *out_costs) {
        static auto cfg = futhark_context_config_new();
        static auto ctx = futhark_context_new(cfg);
        struct futhark_f64_2d *cost_mat =
            futhark_new_f64_2d(ctx, in, m, n);
        struct futhark_i32_2d *least_asgns =
            futhark_new_i32_2d(ctx, out_asgns, m, k);
        struct futhark_f64_1d *least_costs =
            futhark_new_f64_1d(ctx, out_costs, k);

        futhark_entry_main(ctx, &least_asgns, &least_costs, cost_mat, k);
        futhark_context_sync(ctx);

        futhark_values_i32_2d(ctx, least_asgns, out_asgns);
        futhark_values_f64_1d(ctx, least_costs, out_costs);

        futhark_free_f64_2d(ctx, cost_mat);
        futhark_free_i32_2d(ctx, least_asgns);
        futhark_free_f64_1d(ctx, least_costs);
    }
};
#endif

template <typename Device> class MurtyOp : public OpKernel {
  public:
    explicit MurtyOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        const Tensor &input_tensor = context->input(0);

        int k = static_cast<int *>(context->input(1).data())[0];

        TensorShape in_shape = input_tensor.shape();
        int64_t m = in_shape.dim_size(0);
        int64_t n = in_shape.dim_size(1);

        TensorShape asgns_shape;
        asgns_shape.AddDim(k);
        asgns_shape.AddDim(m);

        TensorShape costs_shape;
        costs_shape.AddDim(k);

        Tensor *asgns_tensor = NULL;
        Tensor *costs_tensor = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, asgns_shape, &asgns_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, costs_shape, &costs_tensor));

        MurtyFunctor<Device>()(context->eigen_device<Device>(), k, n, m,
                               static_cast<double *>(input_tensor.data()),
                               static_cast<int *>(asgns_tensor->data()),
                               static_cast<double *>(costs_tensor->data()));
    }
};

REGISTER_KERNEL_BUILDER(Name("Murty").Device(DEVICE_CPU),
                        MurtyOp<Eigen::ThreadPoolDevice>);
#ifdef GPU
REGISTER_KERNEL_BUILDER(Name("Murty").Device(DEVICE_GPU).HostMemory("k"),
                        MurtyOp<Eigen::GpuDevice>);
#endif
} // namespace functor
} // namespace tensorflow

REGISTER_OP("Murty")
    .Input("costs: float64")
    .Input("k: int32")
    .Output("asgns: int32")
    .Output("asgn_costs: float64");
