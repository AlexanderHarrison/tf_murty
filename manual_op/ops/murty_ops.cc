#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Murty")
    .Attr("K: int")
    .Attr("N: int")
    .Attr("M: int")
    .Input("costs: float")
    .Output("mincost: float");
