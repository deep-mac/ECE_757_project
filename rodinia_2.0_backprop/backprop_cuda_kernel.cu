

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

//typedef struct shmemTableEntry;

struct queueNode{
    int vaddr;
    queueNode* next;
};

__device__ int get_smid(void) {
     int ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

__device__ void pushQueue(struct queueNode **q, int vaddr){
        queueNode *newNode = (struct queueNode*)malloc(sizeof(struct queueNode));
        newNode->vaddr = vaddr;
        newNode->next = NULL;
        struct queueNode *ptr = *q; 
        if(*q == NULL)	*q = newNode;
        else
        {
        	while(ptr->next != NULL) ptr = ptr->next;
        	ptr->next = newNode;
        }
}

__device__ int findInQueue(struct queueNode **q, int vaddr){
	struct queueNode *ptr = *q;
	int index = 0;
	if(*q == NULL) return -1;
	while(ptr != NULL)
	{
		if(ptr->vaddr == vaddr) 
			return index;
		ptr = ptr->next;
		index++;
	}
	printf("You idiot, you moron, good for nothing dumbo! Can't even write a simple line of code. Why did you even come for MS?!\n");
	return -1;
}

__device__ void delFromQueue(struct queueNode **q, int index)
{
	struct queueNode *ptr = *q;
	struct queueNode *prev = NULL;
	int pos = 0;
	if(index == 0)
	{
		*q = (*q)->next;
		free(ptr);
		return;
	}
	while(ptr!= NULL)
	{
		if(pos == index)
		{
			prev->next = ptr->next;
			free(ptr);
			return;
		}
		prev = ptr;
		ptr = ptr->next;
		pos++;
	}
	return;
}

__device__ int sizeQueue(struct queueNode **q)
{
    struct queueNode *ptr = *q;
    int count = 0;
    if(ptr == NULL)
        return 0;
    while(ptr != NULL)
    {
        ptr = ptr->next;
        count++;
    }
    return count;
}

__device__ void delQueue(struct queueNode **q)
{
    struct queueNode *ptr = *q;
    if(ptr == NULL)
        return;
    while (*q != NULL)
    {
        *q = (*q)->next;
        free(ptr);
        ptr = *q;
    }

}

__device__ int getValueQueue(struct queueNode **q, int index)
{
    struct queueNode *ptr = *q;
    if (ptr == NULL)
        return -1;
    for(int i = 0; i < index; i++)
    {
        ptr = ptr->next;
    }
    return (ptr->vaddr);
}

struct shmemTableEntry {
    int valid;
    struct queueNode *vaddrQueue;
    int paddr;
    float hash[3]; //0 = min, 1 = max, 2 = avg
    int status; //Empty = 0, partial = 1 to 3, full = 4
} ;


__device__ float getApproxShmem(float *input_vaddr, float approx_shmem[][SHMEM_ELEMENTS_PER_BANK], struct shmemTableEntry shmemTable[][SHMEM_TABLE_NUM_ENTRIES]) {
    int b;
    long int vaddr = (long int)input_vaddr;
    b = vaddr % 32;
    vaddr = vaddr << 5;
    float value;
    int match = 0;
    
    for (int i = 0; i < SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
        if (shmemTable[b][i].valid) {
            int idx = findInQueue(&shmemTable[b][i].vaddrQueue, vaddr >> 2);
            if (idx != -1) {  //There is a match so check for status
                //there is match
                value = approx_shmem[b][shmemTable[b][i].paddr << 2 + (vaddr & 3)];
                match = 1;
                break;
            }
        }
    }
    if (match)
        return value;
    else 
        return 0;
}

__device__ void approxShmem(float weight[][WIDTH], float approx_shmem[][SHMEM_ELEMENTS_PER_BANK], struct shmemTableEntry shmemTable[][SHMEM_TABLE_NUM_ENTRIES], int bx, int by, int tx, int ty, float *input_vaddr){
   
    int b; //b = bank
    long int vaddr = (long int)input_vaddr;
    int hash_diverged = 0;
    int diverge_index = -1;
    int foundMatch = 0;
    float temp_hash;
    printf("INSIDE THE FUnCTION\n");
    //if write
        //check the block in which this falls is getting full
        //Check if there is an entry for the this block of shmem - check all partial blocks
        //If none exist then allocate new entry and make that entry partial
        //If getting full
            //Calculate hash
            //Check with any other existing hashes 
                //if it is same then discard all data
                //Else allocate new entry in the table and store the hash


    //Looping shmemTable -
        //If entry valid or not
        //If valid && the vaddr is matching with the entry
            //Set foundMatch = 1;
            //If partial - just write the data, update status
                //If it is becoming full - case 1
                    //Calculate hash 
                    //Check with all other entries in table
                        //If matches - compress it
                        //If not, do nothing
            //If full
                //Recalculate hash - but don't update table yet
                //If this was already compressed and previous hash differs from new hash - case 2
                    //Allocate new entry - set a hash_diverged flag for now
                //If not already compressed, do nothing just write the hash in table - case 3
        //If not valid
            //Do nothing
        //foundMatch == 0 || hash_diverged == 1, then nothing happened in the above loop that means there was nothing matching
            //So just allocate a new entry
        //If hash_diverged == 0, At the end of all this, check whether any two hashes are now matching 
            //If matching - invalid that entry and add its vaddr to the matching entry
    hash_diverged = 0;
    diverge_index = -1;
    foundMatch = 0;
    temp_hash = 0;
    b = vaddr % 32;
    vaddr = vaddr >> 5;
    
    for (int i = 0; i< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
        if(shmemTable[b][i].valid == 1) //&& shmemTable[i].status == < 4) //Check whether there is anything valid and partial
        {
            int idx = findInQueue(&shmemTable[b][i].vaddrQueue, vaddr >> 2);
            if (idx != -1) {  //There is a match so check for status
                foundMatch = 1;
                if (shmemTable[b][i].status < SHMEM_CHUNK_SIZE) {
                    approx_shmem[b][shmemTable[b][i].paddr<<2+ (vaddr & 3)] = weight[ty][tx]; //Append two least significant bits in the paddr
                    shmemTable[b][i].status++;
                    if (shmemTable[b][i].status == SHMEM_CHUNK_SIZE) {
                       //calculate hash
                       temp_hash = approx_shmem[b][shmemTable[b][i].paddr<<2 + 0] +
                                   approx_shmem[b][shmemTable[b][i].paddr<<2 + 1] +
                                   approx_shmem[b][shmemTable[b][i].paddr<<2 + 2] +
                                   approx_shmem[b][shmemTable[b][i].paddr<<2 + 3];
                       shmemTable[b][i].hash[0] = temp_hash;
                    }
                }
                else {
                    //re-calculate hash in temp variable
                   temp_hash = approx_shmem[b][shmemTable[b][i].paddr<<2 + 0] +
                               approx_shmem[b][shmemTable[b][i].paddr<<2 + 1] +
                               approx_shmem[b][shmemTable[b][i].paddr<<2 + 2] +
                               approx_shmem[b][shmemTable[b][i].paddr<<2 + 3];
                    if (sizeQueue(&shmemTable[b][i].vaddrQueue) > 1) {//This means already compressed
                        if (shmemTable[b][i].hash[0] != temp_hash) { //this needs to be some sort of similarity check not an exact equal to
                           hash_diverged = 1; 
                           diverge_index = i;
                        }
                        else {
                            //Do nothing and just write data
                            approx_shmem[b][shmemTable[b][i].paddr <<2 + (vaddr & 3)] = weight[ty][tx]; //Append two least significant bits in the paddr
                        }
                    }
                    else {
                        approx_shmem[b][shmemTable[b][i].paddr <<2 + (vaddr & 3)] = weight[ty][tx];  //Append two least significant bits in the paddr

                        //Just write the temp hash into the table
                        shmemTable[b][i].hash[0] = temp_hash;
                    }
                }
            }
        }
    }
    if (foundMatch == 0 || hash_diverged == 1) {
        for (int i = 0; i< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
            if(shmemTable[b][i].valid == 0) {
                delQueue(&shmemTable[b][i].vaddrQueue);
                pushQueue(&shmemTable[b][i].vaddrQueue, vaddr >> 2);
                approx_shmem[b][shmemTable[b][i].paddr << 2 + (vaddr & 3)] = weight[ty][tx];  //Append two least significant bits in the paddr
                shmemTable[b][i].status = 1;
                shmemTable[b][i].valid = 1;
                printf("Allocating an entry here %d, %d, %d, %d\n", b, i, ty, tx);
                //FIXME PLEASEEEEEEEEEEEEEEEEEEEEE Delete older vddr after hash divergece
                if (hash_diverged == 1) {
                    approx_shmem[b][shmemTable[b][i].paddr<<2 + 0] =  approx_shmem[b][shmemTable[b][diverge_index].paddr<<2 + 0];  // Append two least significant bits in the paddr<<2 + 
                    approx_shmem[b][shmemTable[b][i].paddr<<2 + 1] =  approx_shmem[b][shmemTable[b][diverge_index].paddr<<2 + 1];  // Append two least significant bits in the paddr<<2 + 
                    approx_shmem[b][shmemTable[b][i].paddr<<2 + 2] =  approx_shmem[b][shmemTable[b][diverge_index].paddr<<2 + 2];  // Append two least significant bits in the paddr<<2 + 
                    approx_shmem[b][shmemTable[b][i].paddr<<2 + 3] =  approx_shmem[b][shmemTable[b][diverge_index].paddr<<2 + 3];  // Append two least significant bits in the paddr<<2 + 
                    approx_shmem[b][shmemTable[b][i].paddr<<2 + (vaddr & 3)] =  weight[ty][tx];  // Append two least significant bits in the paddr
                    shmemTable[b][i].hash[0] = temp_hash;
                }
                break;
            }
        }
    }
    if( hash_diverged == 0) {
        for (int i = 0; i< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
            if (shmemTable[b][i].valid == 0)
                continue;
            for (int j = 0; j< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; j++){
                if (shmemTable[b][j].valid == 0)
                    continue;
                if (shmemTable[b][i].hash[0] == shmemTable[b][j].hash[0]) {
                    for (int k = 0; k < sizeQueue(&shmemTable[b][j].vaddrQueue); k++)
                        pushQueue(&shmemTable[b][i].vaddrQueue, getValueQueue(&shmemTable[b][j].vaddrQueue, k));
                        shmemTable[b][j].valid = 0;
                        delQueue(&shmemTable[b][i].vaddrQueue);
                }
            }
        }
    }
    
}

__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid) 
{
   int by = blockIdx.y;
   int bx = blockIdx.x;
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int smID = get_smid();

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];
   float approx_weight_matrix[HEIGHT][WIDTH];
   float approx_shmem[SHMEM_NUM_BANKS][SHMEMSIZE/4/SHMEM_NUM_BANKS]; //32 banks and 384 elements per bank
   struct shmemTableEntry shmemTable[SHMEM_NUM_BANKS][SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE]; //for all 32 banks, entries per chunk

    //FIXME do a shmem table init
    //Set paddr based on the loop index - a constant paddr per entry for simple allocation
    //Set all valid to 0, 
    for (int b = 0; b < SHMEM_NUM_BANKS; b++) {
        for (int i = 0; i <  SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++) {
            shmemTable[b][i].paddr = i;
        }
    }
    
   if ( tx == 0 )
       input_node[ty] = input_cuda[index_in];

   //cout << "inputNode for ty= " << ty << "is " input_node[ty] << "\n";
  // printf("inputNode for ty= %d is %d\n", ty, input_node[ty]);
   
   __syncthreads();

   approx_weight_matrix[ty][tx] = input_hidden_cuda[index];
   //weight_matrix[ty][tx] = input_hidden_cuda[index];

   //APPROXIMATION - (approx_weight_matrix - input/output, bx, by, tx, ty, smID as input)
   approxShmem(approx_weight_matrix, approx_shmem, shmemTable, bx, by, tx, ty, &weight_matrix[ty][tx]); 

   weight_matrix[ty][tx] = approx_weight_matrix[ty][tx];
   //printf("Weight matrix for ty= %d and tx = %d is %d\n", ty, tx, weight_matrix[ty][tx]);
   if (smID == 0)
       printf("WRITE bx,by,tx,ty,smID=%d,%d,%d,%d,%d weight_matrx=%f,weigh_matrix_address=%x\n", bx, by, tx, ty, smID, weight_matrix[ty][tx], &weight_matrix[ty][tx]);

   __syncthreads();

   approx_weight_matrix[ty][tx] = getApproxShmem(&weight_matrix[ty][tx], approx_shmem, shmemTable);
   approx_weight_matrix[ty][tx] = approx_weight_matrix[ty][tx] * input_node[ty];
   weight_matrix[ty][tx] = approx_weight_matrix[ty][tx];
   //printf("READWRITE bx,by,tx,ty,smID=%d,%d,%d,%d,%d weight_matrx=%f\n", bx, by, tx, ty, smID, weight_matrix[ty][tx]);

   __syncthreads();   
   
   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   approx_weight_matrix[ty][tx] = getApproxShmem(&weight_matrix[ty][tx], approx_shmem, shmemTable) + getApproxShmem(&weight_matrix[ty + power_two/2][tx], approx_shmem, shmemTable);
       approxShmem(approx_weight_matrix, approx_shmem, shmemTable, bx, by, tx, ty, &weight_matrix[ty][tx]); 
       weight_matrix[ty][tx] = approx_weight_matrix[ty][tx];

	   __syncthreads();

   }
   
   //__syncthreads();

   input_hidden_cuda[index] = getApproxShmem(&weight_matrix[ty][tx], approx_shmem, shmemTable);
   
/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
 
	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = getApproxShmem(&weight_matrix[tx][ty], approx_shmem, shmemTable);
   }

}


__global__ void bpnn_adjust_weights_cuda(float * delta,   
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw)  									
{
  
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}
#endif 
