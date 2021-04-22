MPI Problem on Summit
=====================

This repo is a minimal working example to demonstrate a potential memory leak on
Summit. The program simply does an `MPI_Sendrecv` many times in a fixed pattern
using Cuda-aware MPI, reminiscent of typical simulation codes where domain
communication is invoked at each time step. Use the following command to
compile:

    mkdir build
    cd build
    source ../env
    CC=gcc CXX=g++ cmake ..
    make

To trigger the problem, run it either in a batch job or an interactive job with either of the following:

    jsrun --smpiargs="-gpu" -n 1 -r 1 -a 6 -c 6 -g 6 ./test

or

    jsrun --smpiargs="-gpu" -n 6 -r 6 -a 1 -c 1 -g 1 ./test

One can see that in the output, the available memory on the GPU goes down
gradually, simply by invoking `MPI_Sendrecv` on a device pointer.

The problem goes away if one exports `PAMI_DISABLE_IPC=1` before launching the
job. However, this noticeably tanks the MPI performance (especially intra-node
communication). It would be awesome if there is a combination of `--smpiargs`
and environment variables that can solve this issue without sacrificing the
performance.
