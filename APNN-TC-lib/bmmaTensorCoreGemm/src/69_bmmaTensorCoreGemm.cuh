/*
  1-bit BMMA code.
  Runs at 500TOPS for matrix size of 4096x4096x8192.
  Borrows largely from CUDA-SDK.

  By Boyuan
*/

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#define CHUNK_K 4
#define SKEW 1
#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32
#define THREADS_PER_BLOCK WARP_SIZE * WARPS_PER_BLOCK
#define CHUNK_LINE_BYTES CHUNK_K * sizeof(int4)
#define WARP_COPY_BYTES WARP_SIZE * sizeof(int4)
#define CHUNK_COPY_LINES_PER_WARP WARP_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES WARP_SIZE / CHUNK_COPY_LINES_PER_WARP
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES WARP_ROW_TILES * BLOCK_ROW_WARPS
#define BLOCK_COL_TILES WARP_COL_TILES * BLOCK_COL_WARPS
#define M 8
#define N 8
#define K 128

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

typedef union {
  int4 vec;
  int a[4];
} U4;

// Assume that Kernel size is 3x3.
// Assume CIN is 128.
// __global__
__device__ __inline__ 
void APConv_w2a2_pack_pool(const int4 *W, const int4 *X, int *Output, 
                                      int Height, int Width, int CIN, int COUT) {
  // GEMM Configuration
  int X_bit_offset = (Height+2) * (Width+2) * CIN/128;
  int W_bix_offset = 9*CIN*COUT/128;
  int BIT = 2;
  int X_ROW_BIT = (Width+2)*CIN/128;
  int W_ROW_BIT = 9*(CIN/128);

  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   // for(int i = 0; i<Height*Width*CIN/32*BIT; i++) {
  //   //   printf("X[%d]: %x\n", i, *((int*)X+i));
  //   // }  
  //   for(int i = 0; i<COUT*9*CIN/32; i++) {
  //     printf("W[%d]: %x\n", i, *((int*)W+i));
  //   }  
  // }

  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.
  wmma::fragment<wmma::accumulator, 8, 8, 128, int> c[WARP_COL_TILES]
    [WARP_ROW_TILES];
  wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
  wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];


  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos/(COUT/32)) / (Width/8) * 4;
    const unsigned int block_j = (block_pos/(COUT/32)) % (Width/8) * 8;
    const unsigned int block_z = block_pos % (COUT/32) * 32;

    if (block_i >= Height) {
      break;
    }

    int image_starting_idx = block_i * (Width+2) * CIN/128 + block_j * CIN/128;

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j=0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);
    
    // On the K dimension, there are 9*CIN/128 element to solve.
    // This for loop computes [0,1,2,...,int(9*CIN/128/CHUNK_K)*CHUNK_K-1]. Next for loop computes [int(9*CIN/128/CHUNK_K)*CHUNK_K, ..., 9*CIN/128-1]
    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k+CHUNK_K < 9*CIN/128; tile_k += CHUNK_K) {

      int SHMEM_i = threadIdx.x/4;
      int bit_flag = SHMEM_i / (64/BIT); // bit_flag = 0/1, indicates 
      int SHMEM_offset = SHMEM_i % (64/BIT);
      int row = SHMEM_offset / 8;
      int col = SHMEM_offset % 8;
      int t = threadIdx.x % 4;

      int sub_row = (tile_k+t)/(3*CIN/128);
      int sub_col = (tile_k+t)%(3*CIN/128);


      int GL_idx = image_starting_idx + bit_flag*X_bit_offset + row*X_ROW_BIT + col*CIN/128 + sub_row*X_ROW_BIT + sub_col;

      // if (block_pos == 0 && tile_k ==0 && SHMEM_i == 1) {
      //   printf("tile_k: %d, block_i: %d, block_j: %d, row: %d, col: %d, sub_row: %d, sub_col: %d, GL_idx: %d\n", tile_k, block_i, block_j, row, col, sub_row, sub_col, GL_idx);
      //   printf("X[17]: %x %x %x %x\n", *((int*)X+ 4*17), *((int*)X+ 4*17+1), *((int*)X+ 4*17+2), *((int*)X+ 4*17+3));
      // }
  

      shmem[SHMEM_i][t] = X[GL_idx];

      SHMEM_i += 64;
      int weight_load_idx = bit_flag * W_bix_offset + (block_z + SHMEM_offset) * W_ROW_BIT + tile_k + t;
      shmem[SHMEM_i][t] = W[weight_load_idx];

      __syncthreads();

      // if (block_pos == 0 && warpId == 0 && laneId == 0) {
      //   for(int i = 64; i < 65; i++) {
      //     for(int j = 0; j < 16; j++) {
      //       int *tile_ptr = (int*)&shmem[0][0] + i*20 + j;
      //       printf("tile_k: %d, i: %d, j: %d, val: %x\n", tile_k, i, j, *tile_ptr);
      //     }
      //   }
      // }
  


      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
          const int4 *tile_ptr = &shmem[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);
          
        // if (block_pos == 0 && warpId == 4 && laneId == 0) {
        //   printf("tile_k: %d, k_step: %d, shmem_idx_a: %d\n", tile_k, k_step, shmem_idx_a);
        //   for(int t = 0; t<a[i].num_elements; t++) {
        //       printf("tile_k: %d, k_step: %d, a[%d].x[%d]: %x\n", tile_k, k_step, i, t, a[i].x[t]);
        //   }
        // }

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = 64 +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const int4 *tile_ptr = &shmem[shmem_idx_b][k_step * (K/128)];

              wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
            }
            // printf("ckpt4\n");

            // if (block_pos == 0 && warpId == 0 && laneId == 0 && tile_k == 0) {
            //   for(int t = 0; t<b[j].num_elements; t++) {
            //       printf("b[%d].x[%d]: %x\n", j, t, b[j].x[t]);
            //   }
            // }
            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
          }
        }
      }
      __syncthreads();
    }


#pragma unroll
    for (int tile_k = int(9*CIN/128/CHUNK_K)*CHUNK_K; tile_k < 9*CIN/128; tile_k++) {
      int SHMEM_i = threadIdx.x/4;
      int bit_flag = SHMEM_i / (64/BIT);
      int SHMEM_offset = SHMEM_i % (64/BIT);
      int row = SHMEM_offset / 8;
      int col = SHMEM_offset % 8;
      int t = threadIdx.x % 4;

      int sub_row = (tile_k)/(3*CIN/128);
      int sub_col = (tile_k)%(3*CIN/128);

      int GL_idx = image_starting_idx + bit_flag*X_bit_offset + row*X_ROW_BIT + col*CIN/128 + sub_row*X_ROW_BIT + sub_col;
      *((int*)&shmem[SHMEM_i][0] + t) = *((int*)&X[GL_idx] + t);

      SHMEM_i += 64;
      int weight_load_idx = bit_flag * W_bix_offset + (block_z + SHMEM_offset) * W_ROW_BIT + tile_k;
      *((int*)&shmem[SHMEM_i][0] + t) = *((int*)&W[weight_load_idx] + t);

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.

#pragma unroll
      for (int i = 0; i < WARP_COL_TILES; i++) {
        size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
        const int4 *tile_ptr = &shmem[shmem_idx_a][0];

        wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

#pragma unroll
        for (int j = 0; j < WARP_ROW_TILES; j++) {
          if (i == 0) {
            // Load the B matrix fragment once, because it is going to be
            // reused against the other A matrix fragments.
            size_t shmem_idx_b = 64 +
                                  (WARP_ROW_TILES * N) * (warpId % 2) +
                                  (j * N);
            const int4 *tile_ptr = &shmem[shmem_idx_b][0];

            wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
          }
          // printf("ckpt4\n");

          wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
        }
      }
      __syncthreads();
    }
    // if (block_pos == 0 && warpId == 4 && laneId == 0) {
    //   for(int t = 0; t<c[0][0].num_elements; t++) {
    //       printf("c[0][0].x[%d]: %d\n", t, c[0][0].x[t]);
    //   }
    // }
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int*)&shmem[0][0] +
                              (warpId / 2) * 64 * 8 * 2 +
                              (warpId % 2) * 32; // Will be used only when writing back D. May be moved outside the for loop. TODO.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        int *tile_ptr = shmem_warp_tile_ptr + i*64*8 + j*8;
        wmma::store_matrix_sync(tile_ptr, c[i][j], 64,  wmma::mem_row_major);
      }
    }

    __syncthreads();

    // if (block_pos == 0 && warpId == 0 && laneId == 0) {
    //   for(int i = 31; i < 34; i++) {
    //     for(int j = 0; j < 64; j++) {
    //       int *tile_ptr = (int*)&shmem[0][0] + i*64 + j;
    //       printf("i: %d, j: %d, val: %d\n", i, j, *tile_ptr);
    //     }
    //   }
    // }

    int val[4];
    int final_val;
    int *shmem_warp_stream_ptr = (int*)&shmem[0][0] + (warpId/4)*16*64 + (warpId%4)*2*64 + laneId;
    int mask = 1;
    int bit0, bit1;
    unsigned r0, r1;
    int tmp0, tmp1, tmp2, tmp3;

    int Output_row = warpId/4;
    int Output_col = warpId%4;
    int *dst_gmem_warp_stream_ptr;

    tmp0 = *shmem_warp_stream_ptr;
    tmp1 = *(shmem_warp_stream_ptr+32);
    tmp2 = *(shmem_warp_stream_ptr+32*64);
    tmp3 = *(shmem_warp_stream_ptr+32+32*64);
    val[0] = tmp0 + 2*tmp1 + 2*tmp2 + 4*tmp3;

    tmp0 = *(shmem_warp_stream_ptr+64);
    tmp1 = *(shmem_warp_stream_ptr+64+32);
    tmp2 = *(shmem_warp_stream_ptr+64+32*64);
    tmp3 = *(shmem_warp_stream_ptr+64+32+32*64);
    val[1] = tmp0 + 2*tmp1 + 2*tmp2 + 4*tmp3;

    tmp0 = *(shmem_warp_stream_ptr+8*64);
    tmp1 = *(shmem_warp_stream_ptr+8*64+32);
    tmp2 = *(shmem_warp_stream_ptr+8*64+32*64);
    tmp3 = *(shmem_warp_stream_ptr+8*64+32+32*64);
    val[2] = tmp0 + 2*tmp1 + 2*tmp2 + 4*tmp3;

    tmp0 = *(shmem_warp_stream_ptr+9*64);
    tmp1 = *(shmem_warp_stream_ptr+9*64+32);
    tmp2 = *(shmem_warp_stream_ptr+9*64+32*64);
    tmp3 = *(shmem_warp_stream_ptr+9*64+32+32*64);
    val[3] = tmp0 + 2*tmp1 + 2*tmp2 + 4*tmp3;

    final_val = (val[0]+val[1]+val[2]+val[3])/4;

    bit0 = final_val & (mask << 0);
    bit1 = (final_val & (mask << 1)) >> 1;
    r0 = __ballot_sync(0xFFFFFFFF, bit0);
    r1 = __ballot_sync(0xFFFFFFFF, bit1);
    if (laneId == 0) {
      // printf("r0: %x, r1: %x\n", r0, r1);
      dst_gmem_warp_stream_ptr = Output + block_i/2 * Width/2 * COUT/32 + block_j/2*COUT/32 + block_z/32 
            + Output_row*Width/2*COUT/32 + Output_col*COUT/32;
      *dst_gmem_warp_stream_ptr = __brev(r0);
      *(dst_gmem_warp_stream_ptr+Width/2*Height/2*COUT/32) = __brev(r1);
    }

    __syncthreads();
  }
}

void init_matrices(int4 *X, int4 *W, int Height, int Width, int CIN, int COUT, int X_BIT, int W_BIT){
  int *X_int = (int*) X;
  int *W_int = (int*) W;
  for(int b = 0; b<X_BIT; b++) {
    for(int i=0; i < Height+2; i++) {
      for(int j=0; j < Width+2; j++) {
        for(int k = 0; k < CIN/32; k++) {
          // X_int[b*(Height+2)*(Width+2)*CIN/32 + i*(Width+2)*CIN/32 + j*CIN/32 + k] = 0xFFFFFFFF;
          // X_int[b*(Height+2)*(Width+2)*CIN/32 + i*(Width+2)*CIN/32 + j*CIN/32 + k] = i;
          // X_int[b*(Height+2)*(Width+2)*CIN/32 + i*(Width+2)*CIN/32 + j*CIN/32 + k] = j;
          X_int[b*(Height+2)*(Width+2)*CIN/32 + i*(Width+2)*CIN/32 + j*CIN/32 + k] = rand();
        }      
      }
    }  
  }

  for(int b=0; b<W_BIT; b++) {
    for(int i = 0; i < COUT; i++) {
      for(int j = 0; j < 9*CIN/32; j++) {
        // W_int[b*COUT*9*CIN/32+i*9*CIN/32+j] = 0xFFFFFFFF;
        W_int[b*COUT*9*CIN/32+i*9*CIN/32+j] = rand();
        // W_int[b*COUT*9*CIN/32+i*9*CIN/32+j] = i;
      }
    }
  }
}

// int popcnt(int i) {
//      // Java: use int, and use >>> instead of >>
//      // C or C++: use int
//      i = i - ((i >> 1) & 0x55555555);
//      i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
//      return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
// }

int int_pow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp % 2)
           result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}



void compute_ref(int4 *X, int4 *W, int *ref_C, int Height, int Width, int CIN, int COUT, int X_BIT, int W_BIT) {
  int *X_int = (int*) X;
  int *W_int = (int*) W;

  for (int co=0; co<COUT; co++) {
    for (int m = 0; m < Height; m++) {
      for (int n = 0; n < Width; n++) {
      int tmp = 0;
      for(int xb=0; xb<X_BIT; xb++) {
        int X_Multiplier = int_pow(2,xb);
        for(int wb=0; wb<W_BIT; wb++) {
          int W_Multiplier = int_pow(2,wb);
          for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
              for(int k_tile=0; k_tile<CIN/32; k_tile++) {
                  int x_int = X_int[xb*(Height+2)*(Width+2)*CIN/32 + (m+i)*(Width+2)*CIN/32 + (n+j)*CIN/32 + k_tile];
                  int w_int = W_int[wb*COUT*9*CIN/32 + co*9*CIN/32 + i*3*CIN/32 + j*CIN/32 + k_tile];
                  for(int k=0; k<32; k++) {
                    int mask = 1;
                    int x_val = ((mask << k) & x_int) >> k;
                    int w_val = ((mask << k) & w_int) >> k;
                    tmp += X_Multiplier * W_Multiplier * x_val * w_val;
                  }
                  // if(m==0 && n==1 && co == 0) {
                  //   printf("xb: %d, i: %d, j: %d, k_tile: %d, x_int: %x, w_int: %x, tmp: %d, idx: %d\n", xb, i, j, k_tile, x_int, w_int, tmp, xb*Height*Width*CIN/32 + (m+i)*Width*CIN/32 + (n+j)*CIN/32 + k_tile);
                  // }
                }
              }
            }
          }
        }
        ref_C[m*Width*COUT + n*COUT + co]= tmp;
      }
    }  
  }
}

void compute_ref_pack(int4 *W, int4 *X, int *ref_C, int Height, int Width, int CIN, int COUT, int W_BIT, int X_BIT, int OUT_BIT) {
  int *W_int = (int*) W;
  int *X_int = (int*) X;
  int C_ref_before_decompose[Height*Width*COUT];

  for (int co=0; co<COUT; co++) {
    for (int m = 0; m < Height; m++) {
      for (int n = 0; n < Width; n++) {
      int tmp = 0;
      for(int xb=0; xb<X_BIT; xb++) {
        int X_Multiplier = int_pow(2,xb);
        for(int wb=0; wb<W_BIT; wb++) {
          int W_Multiplier = int_pow(2,wb);
          for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
              for(int k_tile=0; k_tile<CIN/32; k_tile++) {
                  int x_int = X_int[xb*(Height+2)*(Width+2)*CIN/32 + (m+i)*(Width+2)*CIN/32 + (n+j)*CIN/32 + k_tile];
                  int w_int = W_int[wb*COUT*9*CIN/32 + co*9*CIN/32 + i*3*CIN/32 + j*CIN/32 + k_tile];
                  for(int k=0; k<32; k++) {
                    int mask = 1;
                    int x_val = ((mask << k) & x_int) >> k;
                    int w_val = ((mask << k) & w_int) >> k;
                    tmp += X_Multiplier * W_Multiplier * x_val * w_val;
                  }
                  // if(m==0 && n==1 && co == 0) {
                  //   printf("xb: %d, i: %d, j: %d, k_tile: %d, x_int: %x, w_int: %x, tmp: %d, idx: %d\n", xb, i, j, k_tile, x_int, w_int, tmp, xb*Height*Width*CIN/32 + (m+i)*Width*CIN/32 + (n+j)*CIN/32 + k_tile);
                  // }
                }
              }
            }
          }
        }
        C_ref_before_decompose[m*Width*COUT + n*COUT + co]= tmp;
      }
    }  
  }

  for(int m=0; m<Height; m++) {
    for(int n=0; n<Width; n++) {
      int val[OUT_BIT];
      for(int b=0; b<OUT_BIT; b++) {
        val[b] = 0;
      }
      for(int co_tile = 0; co_tile<COUT/32; co_tile++) {
        for(int co=0; co<32; co++) {
          int tmp = C_ref_before_decompose[m*Width*COUT + n*COUT + co_tile*32+co];
          tmp = (tmp - 0);  // Can be modified for other quantized parameters.
          for(int b=0; b<OUT_BIT; b++) {
            int mask = 1;
            val[b] = val[b] << 1;
            val[b] = val[b] | (((mask<<b) & tmp) >> b);
          }
        }
        for(int b=0; b<OUT_BIT; b++) {
          ref_C[b*Height*Width*COUT/32+m*Width*COUT/32+n*COUT/32 + co_tile] = val[b];
        }
      }
    }
  }
}


void compute_ref_pack_pool(int4 *W, int4 *X, int *ref_C, int Height, int Width, int CIN, int COUT, int W_BIT, int X_BIT, int OUT_BIT) {
  int *W_int = (int*) W;
  int *X_int = (int*) X;
  int C_ref_before_decompose[Height*Width*COUT];

  for (int co=0; co<COUT; co++) {
    for (int m = 0; m < Height; m++) {
      for (int n = 0; n < Width; n++) {
      int tmp = 0;
      for(int xb=0; xb<X_BIT; xb++) {
        int X_Multiplier = int_pow(2,xb);
        for(int wb=0; wb<W_BIT; wb++) {
          int W_Multiplier = int_pow(2,wb);
          for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
              for(int k_tile=0; k_tile<CIN/32; k_tile++) {
                  int x_int = X_int[xb*(Height+2)*(Width+2)*CIN/32 + (m+i)*(Width+2)*CIN/32 + (n+j)*CIN/32 + k_tile];
                  int w_int = W_int[wb*COUT*9*CIN/32 + co*9*CIN/32 + i*3*CIN/32 + j*CIN/32 + k_tile];
                  for(int k=0; k<32; k++) {
                    int mask = 1;
                    int x_val = ((mask << k) & x_int) >> k;
                    int w_val = ((mask << k) & w_int) >> k;
                    tmp += X_Multiplier * W_Multiplier * x_val * w_val;
                  }
                  // if(m==0 && n==1 && co == 0) {
                  //   printf("xb: %d, i: %d, j: %d, k_tile: %d, x_int: %x, w_int: %x, tmp: %d, idx: %d\n", xb, i, j, k_tile, x_int, w_int, tmp, xb*Height*Width*CIN/32 + (m+i)*Width*CIN/32 + (n+j)*CIN/32 + k_tile);
                  // }
                }
              }
            }
          }
        }
        C_ref_before_decompose[m*Width*COUT + n*COUT + co]= tmp;
      }
    }  
  }

  int size_after_pool = (int)((float)Height /2 * (float)Width/2 * COUT);
  // printf("Height: %d, Width: %d, COUT: %d, size_after_pool: %d\n", Height, Width, COUT, (int)size_after_pool);

  int half_width = Width/2;
  int half_height = Height/2;

  int C_ref_after_pool[size_after_pool];
  for(int m=0; m<half_height; m++) {
    for(int n=0; n<half_width; n++) {
      for(int co=0; co<COUT; co++) {
        int val1 = C_ref_before_decompose[2*m*Width*COUT+2*n*COUT+co];
        int val2 = C_ref_before_decompose[2*m*Width*COUT+(2*n+1)*COUT+co];
        int val3 = C_ref_before_decompose[(2*m+1)*Width*COUT+2*n*COUT+co];
        int val4 = C_ref_before_decompose[(2*m+1)*Width*COUT+(2*n+1)*COUT+co];
        
        C_ref_after_pool[m*half_width*COUT+n*COUT+co] = (val1+val2+val3+val4)/4;
      }
    }
  }

  // for(int co=32; co<64; co++) {
  //   int val1 = C_ref_before_decompose[12*Width*COUT+ 8*COUT+co];
  //   int val2 = C_ref_before_decompose[12*Width*COUT+ 9*COUT+co];
  //   int val3 = C_ref_before_decompose[13*Width*COUT+ 8*COUT+co];
  //   int val4 = C_ref_before_decompose[13*Width*COUT+ 9*COUT+co];
  //   int val = C_ref_after_pool[6*half_width*COUT+4*COUT+co];
  //   printf("co: %d, val1: %d, val2: %d, val3: %d, val4: %d, val: %x\n", co, val1, val2, val3, val4, val);
  // }




  for(int m=0; m<half_height; m++) {
    for(int n=0; n<half_width; n++) {
      int val[OUT_BIT];
      for(int b=0; b<OUT_BIT; b++) {
        val[b] = 0;
      }
      for(int co_tile = 0; co_tile<COUT/32; co_tile++) {
        for(int co=0; co<32; co++) {
          int tmp = C_ref_after_pool[m*half_width*COUT + n*COUT + co_tile*32+co];
          tmp = (tmp - 0);  // Can be modified for other quantized parameters.
          for(int b=0; b<OUT_BIT; b++) {
            int mask = 1;
            val[b] = val[b] << 1;
            val[b] = val[b] | (((mask<<b) & tmp) >> b);
          }
        }
        for(int b=0; b<OUT_BIT; b++) {
          ref_C[b*half_height*half_width*COUT/32+m*half_width*COUT/32+n*COUT/32 + co_tile] = val[b];
        }
      }
    }
  }
}

void validate_results(int *C, int* ref_C, int Height, int Width, int COUT) {
  printf("Checking computed result for correctness: \n");
  bool correct = true;
  double eps = 1.e-6;  // machine zero

  for(int i = 0; i < Height; i++) {
    for(int j = 0; j < Width; j++) {
      for(int co=0; co<COUT; co++) {
        int idx = i*Width*COUT+j*COUT+co;
        double dst = fabs(C[idx] - ref_C[idx]);
        double abs = fabs(C[idx]) * fabs(ref_C[idx]);
        double ref_err = dst / abs;
        if (ref_err > eps) {
          // printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",, eps);
          printf("i: %d, j: %d, co: %d, C: %d, ref_C: %d\n", i, j, co, C[idx], ref_C[idx]);
          // printf("non equal\n");
          correct = false;
        }
      }
    }
  }
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}



void validate_results_pack(int *C, int* ref_C, int Height, int Width, int COUT, int OUT_BIT) {
  printf("Checking computed result for correctness: \n");
  bool correct = true;
  double eps = 1.e-6;  // machine zero

  for(int i = 0; i<Height; i++) {
    for(int j = 0; j<Width; j++) {
      for(int co=0; co<COUT/32; co++) {
        for(int b=0; b<OUT_BIT; b++) {
          int idx = b*Height*Width*COUT/32 + i*Width*COUT/32+j*COUT/32+co;
          double dst = fabs(C[idx] - ref_C[idx]);
          double abs = fabs(C[idx]) * fabs(ref_C[idx]);
          double ref_err = dst / abs;
          if (ref_err > eps) {
            // printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",, eps);
            printf("xb: %d, i: %d, j: %d, co: %d, C: %x, ref_C: %x\n", b, i, j, co, C[idx], ref_C[idx]);
            // printf("non equal\n");
            correct = false;
          }
        }
      }
    }
  }
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

// #define verify_output

#ifdef main
int main(int argc, char **argv) {
  printf("Initializing...\n");

  int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  int Height = 32;
  int Width = 32;
  int X_BIT = 2;
  int W_BIT = 2;

  // for(int CIN = 128; CIN <= 2048; CIN+=128) {
    int CIN = 256;
    int COUT = CIN;
    int4 *X = NULL;
    int4 *W = NULL;
    int *Output = NULL;
  
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&X), sizeof(int4) * (Height+2) * (Width+2) * (CIN/128) * X_BIT));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&W), sizeof(int4) * 9 * (CIN/128) * COUT * W_BIT));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&Output), sizeof(int4) * Height * Width * COUT ));
  
#ifdef verify_output
  printf("Preparing validation data for GPU...\n");
  int4 *W_h = NULL;
  int4 *X_h = NULL;
  int *Output_h = NULL;
  
  X_h = (int4 *)malloc(sizeof(int4) * (Height+2) * (Width+2) * (CIN/128) * X_BIT);
  W_h = (int4 *)malloc(sizeof(int4) * 9 * (CIN/128) * COUT * W_BIT);
  Output_h = (int *)malloc(sizeof(int) * (Height+2) * (Width+2) * COUT);
  init_matrices(X_h, W_h, Height, Width, CIN, COUT, X_BIT, W_BIT);
  checkCudaErrors(cudaMemcpy(X, X_h, sizeof(int4) * (Height+2) * (Width+2) * (CIN/128) * X_BIT, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(W, W_h, sizeof(int4) * 9 * (CIN/128) * COUT * W_BIT, cudaMemcpyHostToDevice));
#endif
  
    int SHMEM_SZ = 65536;
    checkCudaErrors(cudaFuncSetAttribute(
      APConv_w2a2_pack_pool, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SHMEM_SZ));
  
    // Run ours NUM_PROFILES times and record time.
    float bmma_ms_avg = 0.0f;
    int NUM_PROFILES = 1;
    for(int iter=0; iter<NUM_PROFILES; ++iter){
            float bmma_ms = 0.0f;
            cudaEvent_t bmma_start;
            cudaEvent_t bmma_end;
            cudaEventCreate(&bmma_start);
            cudaEventCreate(&bmma_end);
            cudaEventRecord(bmma_start);
            checkKernelErrors(
              (APConv_w2a2_pack_pool<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                    SHMEM_SZ>>>(W, X, Output, Height, Width, CIN, COUT)));
            cudaEventRecord(bmma_end);
            cudaEventSynchronize(bmma_end);
            cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
            cudaEventDestroy(bmma_start);
            cudaEventDestroy(bmma_end);
            bmma_ms_avg += bmma_ms;
    }
  
    bmma_ms_avg = bmma_ms_avg/(double)NUM_PROFILES;
    printf("H: %d, W: %d, CIN: %d, COUT: %d, W_BIT: %d, X_BIT: %d\n", Height, Width, CIN, COUT, W_BIT, X_BIT);
    printf("Time: %f ms\n", bmma_ms_avg);
  
    printf("TOPS: %.2f\n\n", (((double)9 * CIN * Height * Width * COUT * 2)/(bmma_ms_avg/1000.)) / 1e12);
  
#ifdef verify_output
    printf("Validating results...\n");
    checkCudaErrors(cudaMemcpy(Output_h, Output, sizeof(int) * Height * Width * COUT, cudaMemcpyDeviceToHost));

    int *C_ref = (int *)malloc(sizeof(int) * Height * Width * COUT * X_BIT);

    /* Copmpute reference matrix on CPU */
    compute_ref_pack_pool(W_h, X_h, C_ref, Height, Width, CIN, COUT, W_BIT, X_BIT, X_BIT);

    /* validation results */
    validate_results_pack(Output_h, C_ref, Height/2, Width/2, COUT, X_BIT);
    free(C_ref);
    free(X_h);
    free(W_h);
    free(Output_h);
#endif
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(W)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(X)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(Output)));

  // }

  return EXIT_SUCCESS;
}
#endif