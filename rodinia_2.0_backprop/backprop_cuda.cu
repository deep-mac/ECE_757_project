

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include <backprop_cuda_kernel.cu>
#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

extern "C"
const char* goldfile;
double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  printf("Num_blocks = %d\n", num_blocks);
  dim3  threads(16 , 16);

  float **approx_shmem;
  struct shmemTableEntry **shmemTable;
  
  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
 
  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
	  input_weights_one_dim[m] = net->input_weights[k][j];
	  input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }
 
  //FIXME figure out a way to allocate all pointers in the cuda memory itself. Looks like this way, GPU array is pointing towards host addresses.
  float **h_approx_shmem = (float**)malloc(num_blocks*sizeof(float*));
  struct shmemTableEntry **h_shmemTable = (struct shmemTableEntry**)malloc(num_blocks*sizeof(struct shmemTableEntry*));
  printf("Malloc 1\n");
  for (int blks = 0; blks < num_blocks; blks++) {
      cudaMalloc((void**)&h_approx_shmem[blks], (SHMEM_NUM_BANKS*SHMEM_ELEMENTS_PER_BANK) * sizeof(float));
      cudaMalloc((void**)&h_shmemTable[blks], (SHMEM_NUM_BANKS*SHMEM_TABLE_NUM_ENTRIES) * sizeof(struct shmemTableEntry));
  }
  printf("Malloc 2\n");

  cudaMalloc((void***)&approx_shmem, num_blocks*sizeof(float*));
  cudaMalloc((void***)&shmemTable, num_blocks*sizeof(struct shmemTableEntry*));
  cudaMemcpy(approx_shmem, h_approx_shmem, num_blocks*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(shmemTable, h_shmemTable, num_blocks*sizeof(struct shmemTableEntry*), cudaMemcpyHostToDevice);
  printf("Malloc 3\n");

  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  
  
#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  
  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  /*  for (int blk = 0; blk < num_blocks; blk++){
        for (int b = 0; b < SHMEM_NUM_BANKS; b++) {
            for (int c = 0; c <  SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; c++) {
                shmemTable[blk][b][c].paddr = c;
                shmemTable[blk][b][c].vaddrQueue = NULL;
            }
       }
   }*/
  
  
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid, 
                                              approx_shmem, 
                                              shmemTable);
 
  cudaThreadSynchronize();
  
  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
  
  cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += partial_sum[k * hid + j-1] ;
    }
	sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  #endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);


  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
												);

  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  FILE* ofile = fopen("result.txt", "w");
  unsigned long long int checksum = 0; 
  for (int x = 0; x < (in + 1) * (hid + 1); x++) {
    //printf("input_weights[%d] = %g\n", x, input_weights_one_dim[x]);  // uncomment for detail debug 
	fprintf(ofile, "input_weights[%d] = %.3f\n", x, input_weights_one_dim[x]);
    checksum += ((unsigned int *)input_weights_one_dim)[x]; 
  }
  fclose(ofile);
  printf("checksum = %#llx\n", checksum);
   
	printf("Result stored in result.txt\n");
	
	if(goldfile){
		FILE *gold = fopen(goldfile, "r");
		FILE *result = fopen("result.txt", "r");
		int result_error=0;
		while(!feof(gold)&&!feof(result)){
			if (fgetc(gold)!=fgetc(result)) {
				result_error = 1;
				break;
			}
		}
		if((feof(gold)^feof(result)) | result_error) {
			printf("\nFAILED\n");
		} else {
			printf("\nPASSED\n");
		}

		fclose(gold);
		fclose(result);
	}

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

#endif   
  
  
  

}
