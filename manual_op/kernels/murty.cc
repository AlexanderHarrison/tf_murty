#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"

//#include "murtygpu.h"
#include "murtycpu.cc"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

namespace tensorflow {

namespace functor {

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
//REGISTER_KERNEL_BUILDER( Name("Murty").Device(DEVICE_GPU), MurtyOp<Eigen::GpuDevice>);
}
}

REGISTER_OP("Murty")
    .Input("costs: float32")
    .Input("k: int32")
    .Output("mincost: float32");
