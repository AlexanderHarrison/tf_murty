#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"

#include "murtygpu.h"
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

        thread_local std::vector<int> out_assocs {};

        static bool* priors = [](){
            bool* arr = new bool[256];
            for (int i = 0; i < 256; ++i)
                arr[i] = true;
            return arr;
        }();

        double prior_weights = 0;

        out_assocs.resize(k * (m + n) * 2);
        for (int i = 0; i < k * (m + n) * 2; ++i)
            out_assocs[i] = 0;

        //for (int i = 0; i < k; ++i)
        //    out_costs[i] = 0;

        bool error = da(in, 1, priors, &prior_weights,
                        1, priors, &prior_weights, k,
                        &out_assocs[0], out_costs, &workvars);

        //FILE* f = fopen("out.txt", "w+");

        // Fastmurty has the most obnoxious return format that I have ever seen.
        // 
        // <rant>
        //
        // Normally you'd expect the assignments to be returned in two parts - the row assignments,
        // and the column assignments. Both lists of indices into the passed costs.
        // Or just one of them is returned, and has the user calculate the other if needed.
        // That's what I did with the GPU implementation - just return the row assignments.
        //
        // For some reason the author decided that returning both
        // the row assignments and columns assignments IN A SINGLE ARRAY was a good idea.
        // I have no idea how they came to this conclusion, but fine.
        // That's weird, but fine. Reduce an allocation, only one return ptr, there's reasons.
        // But that's not all. They don't return row indices and then column indices like any
        // reasonable person, they return them MIXED, in (row, col) pairs.
        // This means the array is twice the size than straight row assignment indices.
        // So I have to alloc a 2x array just for you, you special snowflake.
        // And this buffer is written to, so it isn't thread safe, 
        // so I need a while new buffer for each thread, which tensorflow uses many of.
        // I also need a buffer containing only 1s for each row and column.
        // That is RO and can be static, but still, really?
        // 
        // But that's not all. Since returning assignment pairs necessarily has a dynamic length,
        // and the author returns a fixed length (n+m)*2*k buffer,
        // a portion of the returned pairs are zeroed.
        // This means there are a bunch of redundant (0, 0) pairs sitting at the end, waiting for a
        // helpless dev to assign row 0 to index 0.
        // They didn't even have the tact to return redundant (-1, -1) pairs, no,
        // that would have been too logical.
        //
        // And of course none of this is documented.
        // I had to figure this out through trial and error and the source code of a fork.
        // I would fork this myself if I had the time.
        //
        // Why did they do it this way? I have no clue.
        // Just alloc two buffers for the indices like I explained,
        // and update assigned indices as necessary. It's so simple.
        // They would be fixed length, with no redundant data, easier to parse, 
        // easier to transform, and half the size than what they did here.
        //
        // </rant>
        for (int solution_idx = 0; solution_idx < k; ++solution_idx) {
            int out_assocs_idx = solution_idx * (n+m) * 2;
            int *association = &out_assocs[out_assocs_idx];
            
            //for (int j=0; j<association_len;++j) {
            //    fprintf(f, "%d, ", association[j]);
            //}
            //fprintf(f, "\n");

            int rows_to_asgn = m;
            for (int i = 0; rows_to_asgn != 0; i+=2) {
                int row = association[i];
                int row_asgn = association[i+1];

                if (row != -1) {
                    out_asgns[solution_idx*m + row] = row_asgn;
                    rows_to_asgn -= 1;
                }
            }
        }
        //fclose(f);
	//delete[] priors;
        //deallocateWorkvarsforDA(workvars);
    }
};

template <> struct MurtyFunctor<Eigen::GpuDevice> {
    void operator()(const Eigen::GpuDevice &d, const int64_t k, const int64_t n,
                    const int64_t m, const double *in, int *out_asgns,
                    double *out_costs) {
        static auto cfg = gpu_futhark_context_config_new();
        static auto ctx = gpu_futhark_context_new(cfg);
        struct gpu_futhark_f64_2d *cost_mat =
            gpu_futhark_new_f64_2d(ctx, in, m, n);
        struct gpu_futhark_i32_2d *least_asgns =
            gpu_futhark_new_i32_2d(ctx, out_asgns, m, k);
        struct gpu_futhark_f64_1d *least_costs =
            gpu_futhark_new_f64_1d(ctx, out_costs, k);

        gpu_futhark_entry_main(ctx, &least_asgns, &least_costs, cost_mat, k);
        gpu_futhark_context_sync(ctx);

        gpu_futhark_values_i32_2d(ctx, least_asgns, out_asgns);
        gpu_futhark_values_f64_1d(ctx, least_costs, out_costs);

        gpu_futhark_free_f64_2d(ctx, cost_mat);
        gpu_futhark_free_i32_2d(ctx, least_asgns);
        gpu_futhark_free_f64_1d(ctx, least_costs);
    }
};

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
REGISTER_KERNEL_BUILDER(Name("Murty").Device(DEVICE_GPU).HostMemory("k"),
                        MurtyOp<Eigen::GpuDevice>);
} // namespace functor
} // namespace tensorflow

REGISTER_OP("Murty")
    .Input("costs: float64")
    .Input("k: int32")
    .Output("asgns: int32")
    .Output("asgn_costs: float64");
