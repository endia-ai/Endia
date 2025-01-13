# Fast Fourier Transform 

From forgotten handwritings by Gauss in the early 1800s, through a rediscovery in the 1960s (for [detecting atomic tests](https://www.youtube.com/watch?v=nmgFG7PUHfo&t=1s) 💣), to its ubiquitous presence in modern technology, the **Fast Fourier Transform (FFT)** is one of the most impactful and elegant algorithms ever. Endia's new FFT module offers a suite of optimized FFT implementations - integrated into Endia's AutoGrad engine, mirroring the functionality of popular frameworks like PyTorch, while benefiting from being written in pure Mojo.

#### 

<div align="center">
  <img src="../../../assets/fft_title_image.jpeg" alt="FFT Title Image" style="max-width: 1000px;"/> <!-- style="max-width: 800px;" -->
</div>

#### 

> **🧪 Experimental:** The FFT module is currently available on Endia's nightly branch, requiring the [Mojo nightly build](https://docs.modular.com/max/install). While it offers exciting new features, please be aware of potential instability. Good news: All current changes are scheduled to be pushed to Endia's main branch with the next Mojo release, expected in a couple of weeks.

## Overview

This module provides a suite of optimized, non-recursive Fast Fourier Transform (FFT) implementations based on the [Cooley-Tukey algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm), featuring parallel processing and efficient multi-dimensional operations. 🏛️

- [utils.mojo](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/utils.mojo)
  - Functions for data buffer manipulation
  - Bit reversal operations
  - Encoding/decoding of runtime parameters
  - Trait for differentiable FFT operations

- [fft_cooley_tukey.mojo](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/fft_cooley_tukey.mojo)
  - Three iterative versions of Cooley-Tukey FFT algorithm for 1-dimensional inputs:
    1. **cooley_tukey_non_recursive:** Mimics divide-and-conquer logic without recursion creating independent parallelizable subtasks
    2. **fft_cooley_tukey_inplace_bit_reversal:** Iterative Bit-reversal implementation of Cooley-Tukey
    3. **fft_cooley_tukey_parallel:** A hybrid approach:
       - Creates independent tasks through iterative divide-and-conquer
       - Applies parallel bit-reversal Cooley-Tukey algorithm on subtasks
  - Enables efficient parallelization and improved performance

- [fftn.mojo](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/fftn_op.mojo) and [ifftn.mojo](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/ifftn_op.mojo)
  - Multi-dimensional FFT and Inverse FFT operations
  - Use parallelizable **fft_cooley_tukey_parallel** algorithm
  - IFFT conjugates input/output and scales output
  - Efficient multi-dimensional processing:
    - **Avoids slicing operations**
    - Uses axis swapping and reshaping
    - Requires k calls of parallel FFT function for k dimensions of n-dimensional array

- Other FFT operations
  - [fft](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/fft_op.mojo), [ifft](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/ifft_op.mojo), [fft2](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/fft2_op.mojo), [ifft2](https://github.com/endia-ai/Endia/blob/nightly/endia/functional/fft_ops/ifft2_op.mojo)
  - Specialized cases of fftn and ifftn operations


## Benchmarks 🔥

The plot below illustrates speed comparisons of 1-dimensional FFTs across various input sizes, ranging from `2**2` to `2**22`. (Measured on an Apple M3).

<div align="center">
  <img src="../../../assets/Endia_vs_PyTorch_FFT_Benchmark.png" alt="Endia_vs_PyTorch_FFT_Benchmark Image" style="max-width: 800px;"/> 
</div>

#### 

Endia's FFT implementation, despite its compactness, delivers performance **not far behind established frameworks**. Further optimizations and algorithmic refinements could push Endia's performance to fully match or even exceed existing solutions.

**If you have ideas on how to further enhance Endia's FFT performance or functionality, feel free to submit a pull request to the nightly branch. Let's push the boundaries of science together!**

## Current Limitations and Roadmap 🚧

1. **Input Size Flexibility**: 
   - Current: Axis dimensions must be a power of 2
   - Coming Soon: Support for arbitrary input sizes (via chirp Z-transform)

2. **GPU Acceleration**: 
   - Current: CPU-only implementation
   - Planned: Full GPU support (top priority once MAX Engine enables GPU capabilities)

3. **Robustness**:
   - Ongoing: Comprehensive edge-case testing for differentiable FFT operations