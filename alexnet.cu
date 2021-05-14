/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**

This example shows how to run convolution kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Ampere GPU.

Writing a single high performance convolution kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of implicit gemm kernel. When used properly, the kernels can hit peak performance
of GPU easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how tensors are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set tensors will be used to compute
output of convolution.

First, we setup the data types of the input tensor A, weights' tensor B and output tensor C along
with alpha, beta as the equation for convolution is C = alpha * Conv2dFprop(A, B) + beta * C. In CUTLASS,
the kernels first compute Conv2dFprop(A, B) and leave the rest of the computation to end of the kernel as
alpha * X + beta * C is a simple element-wise operation on X (Conv2dFprop(A, B)) and C. We call this as 
epilogue of kernel. Hence, we setup data types for alpha and beta to be equal to 
ElementComputeEpilogue = float. We use the data type for elements in input tensor A and B as 
cutlass::half_t. We convey this to CUTLASS kernel by initializing template variables ElementAccumulator (float),
ElementComputeEpilogue (float), ElementInputA (cutlass::half_t), ElementInputB (cutlass::half_t),
ElementOutput (float). Communicating just the data type is not enough. As the data is laid out 
linearly in memory, we have to convey the layout of tensors. We do that by initializing template
variables LayoutInputA, LayoutInputB and LayoutOutput to TensorNHWC cutlass variable. Next, we setup
rules to comptue alpha * X + beta * C which is called epilogue of the kernel. We initialize template
variable EpilogueOp, which takes the data type of output ElementOutput (float), the number of
elements per vector memory access (8), data type of accumulator (float) and data type of
computation of linear combination (alpha * X + beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x64,
64x64x64, 16x8x16 (MxNxK) respectively. When passed to instantiate CUTLASS Implicit GEMM kernel, it
internally deduces the amount of threads needed per thread-block, amount of shared memory, storing
data in bank-conflict free manner, and ton of other variables required to compose, intialize and
launch a high performance Implicit GEMM kernel. This is the beauty of CUTLASS, it relieves developer
from understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma multistage pipeline.
(see include/cutlass/conv/threadblock/implicit_gemm_multistage.h)

tensor in global memory --cp_async--> tile in shared memory --smem loads--> registers 
--mma--> registers --global stores--> output to global memory

NVIDIA Ampere uses `cp_async` to build multistage software pipeline to better hide latencies.


There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS Implicit GEMM
kernel using cutlass::conv::device::ImplicitGemm template.

The next step is to intialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare tensors as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the tensors are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (N = 1, H = 64, W = 64, C = 128), filter size (K = 64,
R = 3, S = 3, C = 128 ), padding, strides, dilation, tensors, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to intialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference convolution kernel (from CUTLASS utilities) to
compare if the output from CUTLASS kernel is same as the reference implicit GEMM kernel.
*/

#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include "gemm.cuh"

#define batch_size 8

// #define BIT_WIDTH 16
// #define BIT_WIDTH 8
#define BIT_WIDTH 4
// #define BIT_WIDTH 1

#if BIT_WIDTH == 32
using ElementInputA           = float;
using ElementInputB           = float;
using ElementOutput           = float;
using ElementAccumulator      = float;
using ElementCompute          = float;
using ElementComputeEpilogue = ElementAccumulator;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, 
    cutlass::layout::TensorNHWC,
    ElementInputB, 
    cutlass::layout::TensorNHWC,
    ElementOutput, 
    cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;
#endif 


#if BIT_WIDTH == 16
using ElementInputA           = cutlass::half_t;
using ElementInputB           = cutlass::half_t;
using ElementOutput           = float;
using ElementAccumulator      = float;
using ElementCompute          = float;
using ElementComputeEpilogue = ElementAccumulator;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;  // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                    128 / cutlass::sizeof_bits<ElementOutput>::value, float, float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
#endif

#if BIT_WIDTH == 8

using ElementInputA           = int8_t;
using ElementInputB           = int8_t;
using ElementOutput           = int32_t;
using ElementAccumulator = int32_t;                   // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;    // Data type of epilogue computation (alpha, beta)

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;  // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;          // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;    // TensorCore instruction shape
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                                                64 / cutlass::sizeof_bits<ElementOutput>::value,
                                                int32_t,
                                                float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
#endif // END if BIT_WIDTH == 8

#if BIT_WIDTH == 4
using ElementInputA = cutlass::int4b_t;              // Data type of elements in input tensor
using ElementInputB = cutlass::int4b_t;              // Data type of elements in input tensor
using ElementOutput = cutlass::int4b_t;              // Data type of elements in output tensor
using ElementAccumulator = int32_t;                   // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;    // Data type of epilogue computation (alpha, beta)

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;    // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;             // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;       // TensorCore instruction shape
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                                                64 / cutlass::sizeof_bits<ElementOutput>::value,
                                                int32_t,
                                                float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAddSaturate,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;
#endif  // END if BIT_WIDTH == 4


#if BIT_WIDTH == 1
using ElementInputA = cutlass::uint1b_t;              // Data type of elements in input tensor
using ElementInputB = cutlass::uint1b_t;              // Data type of elements in input tensor
using ElementOutput = int32_t;                        // Data type of elements in output tensor
using ElementAccumulator = int32_t;                   // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;    // Data type of epilogue computation (alpha, beta)

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 1024>;  // Threadblock tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 1024>;         // Warp tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 256>;    // TensorCore instruction shape
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 2;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<ElementOutput,
                                                128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                int32_t,
                                                float>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;
#endif  // END if BIT_WIDTH == 1

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(false),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 8;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord input_size,
    cutlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c(); 

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "22_ampere_tensorop_conv2dfprop example\n\n"
      << "  This example uses Ampere's Tensor Core operators on F16 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n <int>            Input tensor extent N\n"
      << "  --h <int>            Input tensor extent H\n"
      << "  --w <int>            Input tensor extent W\n"
      << "  --c <int>            Input tensor extent C (input channel)\n"
      << "  --k <int>            Filter extent K (output channel)\n"
      << "  --r <int>            Filter extent R (filter height)\n"
      << "  --s <int>            Filter extent S (filter width)\n\n"
      << "  --alpha <float>      Epilogue scalar alpha\n"
      << "  --beta <float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations <int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag <string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/22_ampere_tensorop_conv2dfprop/22_ampere_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/22_ampere_tensorop_conv2dfprop/22_ampere_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Precision,Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out << "BIT_WIDTH-" << BIT_WIDTH << ","
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_c(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill tensor C on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_c.host_view());

  // Fill tensor C for reference on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_c.host_view());

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_ref_c.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  typename ImplicitGemm::Arguments arguments{
    {
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices 
    },
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_c.device_ref(),
    {options.alpha, options.beta},

    
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm_op();

  CUTLASS_CHECK(result.status);

  //
  // Optional reference check
  //
  
  if (options.reference_check) {
    std::cout << "Verification on host...\n";

    cutlass::conv::Conv2dProblemSize problem_size(
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      mode
    );

    // Compute with reference implementation
    cutlass::reference::host::Conv2dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
    >(
      problem_size,
      tensor_a.host_ref(),
      tensor_b.host_ref(),
      tensor_c.host_ref(),
      tensor_ref_c.host_ref(),
      options.alpha,
      options.beta
    );

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_c.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(
      tensor_c.host_view(),
      tensor_ref_c.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    }
    else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  }
  else {
    result.reference_check = cutlass::Status::kInvalid;
  }

  if (options.save_workspace) {

    std::stringstream ss;

    ss << "22_ampere_workspace_conv2dfprop_"
      << options.input_size.n() << "x" << options.input_size.h() << "x" << options.input_size.w() << "x" << options.input_size.c() 
      << "_"
      << options.filter_size.n() << "x" << options.filter_size.h() << "x" << options.filter_size.w() << "x" << options.filter_size.c() 
      << ".dat";

    std::ofstream output_workspace(ss.str());

    output_workspace 
      << "Input = \n" << tensor_a.host_view() << "\n\n"
      << "Filters = \n" << tensor_b.host_view() << "\n\n";

    if (options.reference_check) {
      output_workspace << "Reference = \n" << tensor_ref_c.host_view() << "\n\n";
    }

    output_workspace << "Computed = \n" << tensor_c.host_view() << std::endl;

    std::cout << "Results written to '" << ss.str() << "'." << std::endl;
  }
  
  //
  // Performance measurement
  //

  if (options.measure_performance) {

    cudaEvent_t events[2];
    
    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }
    }

    // Record an event at the start of a series of convolution operations.
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Launch a sequence of implicit GEMM operations on the device
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result.status = implicit_gemm_op();
      CUTLASS_CHECK(result.status);
    }

    // Record an event when the convolutions have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Print average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
  //
  // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major > 8 || (props.major == 8 && props.minor >= 0))) {
    std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  struct Benchmark {
    int h, w, c, k, r, s;
  } layers[] = {
    // AlexNet (imageNet)
    {224,224,32,64,11,11},
    {28,28,64,192,5,5},
    {14,14,192,384,3,3},
    {14,14,384,256,3,3},
    {14,14,256,256,3,3},
  };

  Result::print_header(std::cout, options) << std::endl;

  // run convolution layers
  int idx = 1;
  for (auto const &layer : layers) {

      options.update({batch_size, layer.h, layer.w, layer.c}, {layer.k, layer.r, layer.s, layer.c});

      Result result = profile_convolution(options);
      result.print(std::cout, idx, options) << std::endl;
      ++idx;
  }

  std::vector<std::vector<int>> MLP_layers_config = 
    {
     {256*6*6,  4096},
     {4096, 4096},
     {4096, 1000},
    };

  auto out = MLP_input_layer<cutlass::int4b_t, cutlass::layout::RowMajor>(batch_size, PAD32(MLP_layers_config[0][1]), PAD32(MLP_layers_config[0][0]));
  for (int i = 1; i < MLP_layers_config.size(); i++){
      out = MLP_hidden_layer<cutlass::int4b_t , cutlass::layout::RowMajor>(batch_size, PAD32(MLP_layers_config[i][1]), PAD32(MLP_layers_config[i][0]), out);
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////