# CUDA C
Practical assignment of the Large Scale Computation course of the Masters in Informatics Engineering of the University of Aveiro.

## Problem
The circular cross-correlation is an important tool to detect similarities between two signals. For two signals with N samples, x(k) and y(k), with 0 ≤ k < N, the circular cross-correlation xy (i), with 0 ≤ i < N, is defined by the formula:<br>
</br><b> xy (i) = Σ[k=0, n−1] x (k )⋅ y [(i+k ) mod n] </b><br><br>

## Implementation
The aim is to develop a <b>CUDA program</b> to be run in a <b>GPU</b> under Linux.</br>

The kernel should compute one cross-correlation point.
Two approaches should be tried:
- The threads in a warp compute successive cross-correlation points
- The threads in a warp compute cross-correlation points separated by a fixed distance.

In both approaches, the best running configuration should be sought, the execution times should be compared with running similar kernels in the CPU and the following question should be answered "Is it worthwhile to use the GPU to run this kind of problems?".
