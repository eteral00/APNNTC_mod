#ifndef layer_h
#define layer_h

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include <cutlass/numeric_types.h>

#include <ctime>
#include <cudnn.h>
#include "helper.h"
#include "gemm.cuh"
#include "config.h"

class CONV{
public:
    CONV(int batch_size, int input_height, int input_width, 
        int in_channels, int out_channels,
        int filter_height, int filter_width, int stride){
        
        _batch_size = batch_size;
        _input_height = input_height;
        _input_width = input_width;     
        
        _in_channels = in_channels;
        _out_channels = out_channels;

        _filter_height = filter_height;
        _filter_width = filter_width;

        _stride = stride;

        // compute the output shape.
        _output_height = (_input_height + 2*padding_h - filter_height)/_stride + 1;
        _output_width = (_input_width + 2*padding_w - filter_width)/_stride + 1;
        
        init();

        printf("Init CONV (n,c,h,w): %d,%d,%d,%d\n", _batch_size, _in_channels, _input_height, _input_width);
        printf("CONV output (h, w): %d,%d\n", _output_height, _output_width);
    }

    ~CONV(){
        cudaFree(filter);
        cudaFree(output);
    }

    void init()
    {
        mode = cutlass::conv::Mode::kCrossCorrelation;
        // problem_size(_batch_size, _in_channels, _out_channels);
        // allocate memory for weight.
        cudaMalloc(&filter, _in_channels*_out_channels*_filter_height*_filter_width*sizeof(ElementInputB)); 
        // allocate memory for output.
        cudaMalloc(&output, _batch_size*_out_channels*_output_height*_output_width*sizeof(ElementOutput)); 
        // update with tensor reference.
        filter_tensor_ref = cutlass::TensorRef<ElementInputB,LayoutInputB>(filter); 
        output_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput>(output); 
        // Initialize alpha and beta for dot product computation.
        alpha = ElementComputeEpilogue(1);
        beta = ElementComputeEpilogue(0);

        input_size = cutlass::Tensor4DCoord(_batch_size, _in_channels, _input_height, _input_width);
        filter_size = cutlass::Tensor4DCoord(_out_channels, _in_channels, _filter_height, _filter_width);
        padding = cutlass::Tensor4DCoord(1,1,1,1);
        conv_stride = cutlass::MatrixCoord(_stride, _stride);
        dilation = cutlass::MatrixCoord(1,1);
        output_size = cutlass::Tensor4DCoord(_batch_size, _out_channels, _output_height, _output_width);
    }

    ElementInputA* forward(ElementInputA* input){

        input_tensor_ref = cutlass::TensorRef<ElementInputA, LayoutInputA>(input); 

        // runnking kernel.
        typename ImplicitGemm::Arguments arguments{
            {
              input_size,
              filter_size,
              padding,
              conv_stride,
              dilation,
              output_size,
              mode,
              split_k_slices 
            },
            input_tensor_ref,
            filter_tensor_ref,
            output_tensor_ref,
            output_tensor_ref,
            {alpha, beta},    
          };
        //
        // Initialize CUTLASS Convolution
        //
        // implicit_gemm_op;
        workspace_size = implicit_gemm_op.get_workspace_size(arguments);

        // Allocate workspace memory
        workspace = cutlass::device_memory::allocation<uint8_t>(workspace_size);
        CUTLASS_CHECK(implicit_gemm_op.initialize(arguments, workspace.get()));
        //
        // Launch initialized CUTLASS kernel
        //
        std::clock_t c_start = std::clock();

        CUTLASS_CHECK(implicit_gemm_op());

        std::clock_t c_end = std::clock();
        float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;

        printf("Forward CONV (ms): %.3f\n", time_elapsed_ms);

        return output;
    }


private:
    int _in_channels;
    int _out_channels;
    int _input_height;
    int _input_width;
    int _filter_height;
    int _filter_width;
    int _stride;
    int _batch_size;

    int _output_height;
    int _output_width;

    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord output_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;

    ImplicitGemm implicit_gemm_op;
    size_t workspace_size;
    cutlass::device_memory::allocation<uint8_t> workspace;
    cutlass::conv::Mode mode; 
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_tensor_ref;
    cutlass::TensorRef<ElementInputB, LayoutInputB> filter_tensor_ref;
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_tensor_ref;
    ElementComputeEpilogue alpha, beta;

    int split_k_slices = 1;     //-> Split K dimension into 1 partitions
    int padding_w = 1;
    int padding_h = 1;

    ElementInputA* input;
    ElementInputB* filter;
    ElementOutput* output;
};

class FC{
public:
    FC(int batch_size, int in_channels, int out_channels){

        _batch_size = batch_size;
        _in_channels = in_channels;
        _out_channels = out_channels;

        init();

        printf("Init FC (n,in,out): %d,%d,%d\n", _batch_size, _in_channels, _out_channels);
    }

    ~FC(){
        cudaFree(weight);
        cudaFree(output);
    }

    void init()
    {   
        problem_size = cutlass::gemm::GemmCoord(_batch_size, _in_channels, _out_channels);
        // allocate memory for weight.
        cudaMalloc(&weight, _in_channels*_out_channels*sizeof(ElementInputB)); 
        // allocate memory for output.
        cudaMalloc(&output, _batch_size*_out_channels*sizeof(ElementOutput)); 
        // update with tensor reference.
        weight_tensor_ref = cutlass::TensorRef<ElementInputB,LayoutInputB_gemm>(weight); 
        output_tensor_ref = cutlass::TensorRef<ElementOutput,LayoutOutput_gemm>(output); 
        // Initialize alpha and beta for dot product computation.
        alpha = ElementComputeEpilogue_gemm(1);
        beta = ElementComputeEpilogue_gemm(0);
    }


    ElementInputA* forward(ElementInputA* input){

        input_tensor_ref = cutlass::TensorRef<ElementInputA, LayoutInputA_gemm>(input); 
        // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
        // instantiated CUTLASS kernel
        typename Gemm::Arguments arguments{
            problem_size,  // <- problem size of matrix multiplication
            input_tensor_ref,
            weight_tensor_ref,
            output_tensor_ref,
            output_tensor_ref,
            {alpha, beta},          // <- tuple of alpha and beta
            split_k_slices};        // <- k-dimension split factor

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        workspace_size = Gemm::get_workspace_size(arguments);
        // Allocate workspace memory
        workspace = cutlass::device_memory::allocation<uint8_t>(workspace_size);
        // Instantiate CUTLASS kernel depending on templates
        // cutlass::Status status = 
        CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

        std::clock_t c_start = std::clock();

        CUTLASS_CHECK(gemm_op());
        
        std::clock_t c_end = std::clock();
        float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;

        printf("Forward FC (ms): %.3f\n", time_elapsed_ms);

        return output;
    }

private:
    int _batch_size;
    int _in_channels;
    int _out_channels;
    
    int split_k_slices = 1;         // <-- Split K dimension into 1 partitions

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementInputA, LayoutInputA_gemm> input_tensor_ref;
    cutlass::TensorRef<ElementInputB, LayoutInputB_gemm> weight_tensor_ref;
    cutlass::TensorRef<ElementOutput, LayoutOutput_gemm> output_tensor_ref;
    ElementComputeEpilogue_gemm alpha;
    ElementComputeEpilogue_gemm beta;
    cutlass::device_memory::allocation<uint8_t> workspace;
    size_t workspace_size;
    Gemm gemm_op;

    ElementInputA* input;
    ElementInputB* weight;
    ElementOutput* output;
};


//
// Pooling Layer kernel_size = 3, stride = 2.
//
class POOL{

    public:
        POOL(int batch_size, int in_channels, int input_height, int input_width, cudnnHandle_t* cuDNN_handler)
    {

        cudnn = cuDNN_handler;
        _batch_size = batch_size;
        _in_channels = in_channels;
        _input_height = input_height;
        _input_width = input_width;     

        _output_height = (_input_height - 3)/2+1;
        _output_width = (_input_width - 3)/2+1;

        output_bytes = _batch_size*_in_channels*_output_height*_output_width*sizeof(cuDNNtype);

        _init();

        printf("Init POOL (n,c,h,w): %d,%d,%d,%d\n",_batch_size, _in_channels, _input_height, _input_width);
        printf("POOL output (h, w): %d,%d\n", _output_height, _output_width);
    }

    ~POOL(){
        cudaFree(output);
        cudaFree(filter);

    }

    void _init() {

        checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));        
        checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                                CUDNN_POOLING_MAX,       //mode - max pooling
                                                CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                                3,                       //window height
                                                3,                       //window width
                                                0,                       //vertical padding
                                                0,                       //horizontal padding
                                                2,                       //vertical stride
                                                2));                     //horizontal stride
                                                
        checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));

        checkCUDNN(cudnnSetTensor4dDescriptor(
                    input_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));            
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _output_height,                
                    _output_width));  

        cudaMalloc(&output, output_bytes);
    }

    cuDNNtype* forward(cuDNNtype* input){

        std::clock_t c_start = std::clock();

        checkCUDNN(cudnnPoolingForward(*cudnn,         
                                        pooling_desc,  
                                        &alpha,       
                                        input_desc,
                                        input,
                                        &beta,        
                                        output_desc,   
                                        output));   

        std::clock_t c_end = std::clock();
        float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
        printf("Forward Pooling (ms): %.3f\n", time_elapsed_ms);

        return output;
    }

    int get_output_height(){ return _output_height;}
    int get_output_width(){ return _output_width; }

private:
    cudnnHandle_t* cudnn;
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnTensorDescriptor_t input_desc, output_desc;

    float alpha = 1.0f;
    float beta = 0.0f;

    int _batch_size;
    int _input_height, _input_width;
    int _output_height, _output_width;
    int _in_channels, _out_channels;

    int output_bytes;

    cuDNNtype* input;
    cuDNNtype* filter;
    cuDNNtype* output;
};

//
// RELU Layer
//
class RELU{
public:
    RELU(int batch_size, int in_channels, int input_height, int input_width, cudnnHandle_t* cuDNN_handler)
    {

        cudnn = cuDNN_handler;
        _batch_size = batch_size;
        _in_channels = in_channels;
        _input_height = input_height;
        _input_width = input_width;     

        output_bytes = _batch_size*_in_channels*_input_height*_input_width*sizeof(cuDNNtype);

        _init();
        printf("Init ReLU (n,c,h,w): %d,%d,%d,%d\n",_batch_size, _in_channels, _input_height, _input_width);
    }

    ~RELU(){
        cudaFree(output);
    }

    void _init() {

        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc));
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                    CUDNN_ACTIVATION_RELU,
                    CUDNN_PROPAGATE_NAN,
                    0.0) );
        
        checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));

        checkCUDNN(cudnnSetTensor4dDescriptor(
                    input_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));            
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    output_desc,             
                    CUDNN_TENSOR_NCHW,       
                    CUDNN_DTYPE,            
                    _batch_size,                       
                    _in_channels,                      
                    _input_height,                
                    _input_width));  

        cudaMalloc(&output, output_bytes);
    }

    cuDNNtype* forward(cuDNNtype* input){

        std::clock_t c_start = std::clock();

        checkCUDNN( cudnnActivationForward(
                    *cudnn,
                    activDesc,
                    &alpha,
                    input_desc,
                    input,
                    &beta,
                    output_desc,
                    output) );   
        std::clock_t c_end = std::clock();
        float time_elapsed_ms = 1000.0f * (c_end-c_start) / CLOCKS_PER_SEC;
        printf("Forward ReLU (ms): %.3f\n", time_elapsed_ms);
        
        return output;
    }


private:
    cudnnHandle_t* cudnn;

    cudnnActivationDescriptor_t  activDesc;
    cudnnTensorDescriptor_t input_desc, output_desc;

    float alpha = 1.0f;
    float beta = 0.0f;

    int _batch_size;

    int _input_height;
    int _input_width;

    int _in_channels;
    int _out_channels;

    int output_bytes;

    cuDNNtype* input;
    cuDNNtype* output;

};

#endif