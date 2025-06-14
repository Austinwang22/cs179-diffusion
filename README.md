Austin Wang, Bobby Wang

For our CS 179 project, we are implementing diffusion models (sampling with the GPU). We are following the framework
presented in [this paper](https://arxiv.org/pdf/2206.00364).

For our CPU implementation, check out the pyref directory. We implemented the training and inference pipelines using PyTorch.
We also implemented a UNet with PyTorch. Our code can be run both with CPU or GPU (simply set the device argument when running a file). 

To train a diffusion model on MNIST, run the following:

```bash
python3 pyref/train.py
```

To generate new samples, run:

```bash
python3 pyref/sample.py --device cpu
```

Here are some generated samples:

![Alt text](figs/sample2.png)
![Alt text](figs/sample3.png)
![Alt text](figs/sample4.png)

For our CUDA implementation, check out the src/ folder. We implemented all of the neural network architecture 
and kernels in the src/diffusion folder. This includes kernels for activation functions, convolution, transposed 
convolution, tensor arithmetic, etc. 

To verify the accuracy of our implementation, we wrote some unit tests for the neural 
network-related kernels, and also wrote a file src/main.cu to check the output of the UNet's 
forward pass on a fixed data point. We wrote pyref/debug.py to run our reference Python model 
on the same input and verified that the CUDA implementation and the Python implementation of 
the UNet matched on this specific input. 

We implemented sampling with the Euler solver as well. Our implementation, while not as fast as the PyTorch 
version using GPU, is currently comparable in speed to the highly optimized PyTorch implementation on CPU. 
We hypothesize that if we were to naively implement all the operations (i.e. convolution, linear layers, etc.) 
in C++, our GPU implementation would be much faster; however, for highly optimized PyTorch code on the CPU and 
on a relatively small model and image resolutions, the PyTorch implementation is comparable in speed to our 
CUDA implementation. Unforunately, our current implementation of the Euler sampling has some bugs that we didn't 
have the time to fix, but we can at least run our forward call of the UNet smoothly and run the entire sampling 
process.

## CUDA Implementation Details

### Codebase Structure
```
src/
├── diffusion/
│   ├── DiffusionConfig.h        # Configuration constants
│   ├── DiffusionEDMPrecond.cuh  # EDM preconditioning wrapper
│   ├── DiffusionHelper.cuh      # Helper functions and kernels
│   ├── DiffusionKernels.cuh     # Core CUDA kernels
│   ├── DiffusionLayers.cuh      # Neural network layers
│   ├── DiffusionLoader.cu       # Model weight loading
│   └── DiffusionUNet.cuh        # UNet architecture
├── CudaBuffer.cuh               # CUDA memory management
├── ErrorCheck.h                 # CUDA error handling
├── HostBuffer.h                 # Host memory management
└── vendor/                      # Third-party dependencies
```

### Key Components

1. **UNet Architecture** (`DiffusionUNet.cuh`)
   - Encoder blocks with downsampling
   - Bottleneck block
   - Decoder blocks with skip connections
   - All operations in BF16 precision

2. **EDM Preconditioning** (`DiffusionEDMPrecond.cuh`)
   - Implements the EDM denoiser: D_θ(x;σ) = c_skip·x + c_out·F_θ(c_in·x, log σ)
   - Handles coefficient computation and blending
   - Manages memory for intermediate results

3. **Core Operations** (`DiffusionLayers.cuh`, `DiffusionKernels.cuh`)
   - Convolution and transposed convolution
   - Activation functions (ReLU)
   - Time embedding
   - Tensor arithmetic operations

4. **Memory Management** (`CudaBuffer.cuh`)
   - Efficient CUDA memory allocation and deallocation
   - Automatic memory management with RAII
   - Support for BF16 and FP32 data types

### Building and Running

1. Build the project:
```bash
mkdir build && cd build
cmake ..
make
```

2. Run sampling:
```bash
./sample
```

3. Run tests:
```bash
./tests
```

### Implementation Notes

- All neural network operations use BF16 precision for better performance
- Memory is managed through the `CudaBuffer` class to prevent leaks
- CUDA streams are used for asynchronous operations
- The sampling process uses Euler integration 

### Performance Optimizations

1. **Memory Management**
   - Reuse of temporary buffers
   - Efficient memory layout for tensor operations
   - Minimal host-device transfers

2. **CUDA Optimizations**
   - Coalesced memory access patterns
   - Efficient kernel launch configurations
   - Stream-based asynchronous execution

We also did some profiling. It seems that our GPU implementation was not the most efficient, 
as we had a small grid size that limited throughput and low achieved occupancy, according to the profiling
screenshots. If we had more time, we would focus on first fixing bugs in the code, then ensuring that
we could focus on the optimization of our CUDA kernels.