# Murty Algorithm on GPU for Tensorflow
Compile with `compile_op.sh`. 
Requires Futhark, a linux machine, and determination.
After compilation, load shared lib in tensorflow as in `op_test.py`.
Usage requires python, tensorflow, numpy, etc.
Nvidia GPU not required (contains cpu backup).

## TODO:
- Optimizations
    - Switch to augmenting deassigned row rather than raw jv each murty step.
    - Sorting each murty step is unnecessary
    - Reduce/save allocations in tf wrapper
    - Use built-in tensorflow allocators rather than futhark's default malloc calls
    - Make sure tf fuses gpu wrapper
    - Profile gpu version -> micro optimizations
    - Use PGO for PTX and/or futhark compilation 
    - (?) Allow multiple matrices to be solved in parallel
