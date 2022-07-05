# Murty Algorithm on GPU for Tensorflow
Compile with `compile_op.sh`. 
Requires Futhark, a linux machine, and determination.
After compilation, load shared lib in tensorflow as in `op_test.py`.

## TODO:
- Finish gpu wrapper + test
- Figure out how to prevent conflicting namespace stuff when tf wrapping both cpu and gpu
- Optimizations
    - Switch to augmenting deassigned row rather than raw jv each murty step.
    - Sorting each murty step is unnecessary
    - Reduce/save allocations in tf wrapper
    - Make sure tf fuses gpu wrapper
    - Profile gpu version -> micro optimizations
