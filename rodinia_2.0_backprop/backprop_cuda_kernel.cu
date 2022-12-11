

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

struct queueNode{
    long unsigned int vaddr;
    struct queueNode* next;
};

__device__ int get_smid(void) {
     int ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

__device__ void pushQueue(struct queueNode **q, long unsigned int vaddr){
    struct queueNode *newNode = (struct queueNode*)malloc(sizeof(struct queueNode));
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

__device__ int findInQueue(struct queueNode **q, long unsigned int vaddr, int print, int tx, int ty, int i){
	struct queueNode *ptr = *q;
	int index = 0;
	if(*q == NULL) {
        if (print)
            printf("Return -1 because NULL\n");
        return -1;
    }
	while(ptr != NULL)
	{
        if (print)
            printf("In findInQueue, index = %d, i = %d, tx = %d, ty= %d, ptr->vaddr = %lx, vaddr = %lx\n", index, i, tx, ty, ptr->vaddr,  vaddr);
		if(ptr->vaddr == vaddr) 
			return index;
		ptr = ptr->next;
		index++;
	}
    if (print)
        printf("Return -1 because no match, i = %d, idx = %d, tx = %d, ty = %d\n", i, index, tx, ty);
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

__device__ void clearQueue(struct queueNode **q)
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

__device__ void printQueue(struct queueNode *q, int i, int tx, int ty)
{
    struct queueNode *ptr = q;
    //printf("Printing queue i = %d, tx = %d, ty = %d", i, tx, ty);
    while(ptr!=NULL) {
        printf("%lx,", ptr->vaddr);
        ptr = ptr->next;
    }
    //printf("\n");
}

__device__ void printShmemTable(struct shmemTableEntry **shmemTable , int bank, int by, int tx, int ty)
{
    for (int b = 0; b < SHMEM_NUM_BANKS; b++) {
        bank = b;
        for (int i= 0; i<256/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
            printf("shmemTable by = %d, tx = %d, ty = %d, bank = %d, valid = %d, vaddr = ",by, tx, ty, bank, shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].valid);
            printQueue(shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, 0, tx, ty);
            printf(" paddr = %d, hash[0] = %f, status = %d\n", shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].paddr, shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].hash[0], shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].status);
        }
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

__device__ int getMinValueIndexQueue(struct queueNode **q)
{
    struct queueNode *ptr = *q;
    if (ptr == NULL)
        return -1;
    int min = getValueQueue(q, 0);
    int minIdx = 0;
    for(int i=0 ; ptr != NULL; ptr=ptr->next, i++)
        if(ptr->vaddr < min)    
        {
            min = ptr->vaddr;
            minIdx = i;
        }
    return minIdx;
}


__device__ float getApproxShmem(int input_vaddr, int b, float **approx_shmem, struct shmemTableEntry **shmemTable, int by, int tx, int ty) {
    
    //b = bank
    float value = 0;
    int chunk = input_vaddr/4;
    b = (ty*16+tx)%32;
    input_vaddr = (int)(ty/2);
    //int by = blockDim.y;
    
    for (int i = 0; i < 256/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
        if (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].valid) {
            //printf("In getApproxShmem, Found valid entry\n");
            //printf("Vaddr = %x\n", vaddr>>2);
            int idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk, 0, tx, ty, 0);
            if (idx != -1) {  //There is a match so check for status
                //there is match
                //printf("In getApproxShmem, Found entry in queue\n");
                value = approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+ ((shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].paddr << 2) + (input_vaddr & 3))];
                break;
            }
        }
    }
    //printf("get approxShmem, value = %f\n", value);
    return value;
}

__device__ void approxShmem(float weight[][WIDTH], float **approx_shmem, struct shmemTableEntry **shmemTable, int bx, int by, int tx, int ty, int bank, float *input_vaddr){
   
    int b; //b = bank
    long unsigned int vaddr = (long int)input_vaddr;
    int hash_diverged = 0;
    int diverge_index = -1;
    int foundMatch = 0;
    float temp_hash;
    long unsigned int chunk_vaddr;
    hash_diverged = 0;
    diverge_index = -1;
    foundMatch = 0;
    temp_hash = 0;
    b = bank;
    chunk_vaddr = vaddr >> 2;
    if (b == 0 && bx == 0 && by == 0)  {
        printf("Start of function\n");
    }
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
    //FIXME - the vaddr&3 will need to change based on the chunk size. It will become SHMEM_CHUNK_SIZE-1
    //FIXME chunk_vaddr = vaddr >> 2 will change when we change chunk size

    //FIXME - make this exact number of elements in weight matrix, rather than all
    //FIXME - status = 15 will change based on chunk size, even hash function will change
    //Start of time 
        // Just write everything
        //Calculate Hash and set hash
        //Perform conservative check
    //Next write 
    //For all chunks in shmem
        //Check for all valid entries
            //Need to find vaddr in the vaddrQueue for all valid chunks
            //If matching vaddr 
                //Set foundMatch = 1;
                //Recalculate hash - but don't update table yet
                //Check if already compressed
                    //if hash not matching 
                        //Allocate new entry, delete current vaddr from this and put in the new entry. Calculate new hash and store. 
                    //If hash matching 
                        //Do nothing just write data and maybe compute hash
                //If not compressed
                    //do nothing just write data and update hash the hash in table - case 3

        //foundMatch == 0 then nothing happened in the above loop that means there was nothing matching 
            //So just allocate a new entry
        //At the end of all this, check whether any two hashes are now matching 
            //If matching - invalid that entry and add its vaddr to the matching entry
    //FIXME - there is a problem with reads. If the hash diverges,  the 0th entry with vaddr zero can move to 1st entry with vaddr 0. It points to paddr 1. However in actual shmem, the entry is still at location 0. Unless there is second copy of the shared memory which reflects data exactly as it is in the shmem_Table, there will be an issue in reading approximate data. 
    //FIXME - the hash looks like is being calculated incorrectly. In the third iteration of writes, all the hashes for which there is compression are exactly same. Need to check. Could be related to the problem above
    for (int chunk = 0; chunk < 256/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; chunk++){
        if (b == 6 && bx == 0 && by == 0) 
            printf("Starting new iteration b = %d, chunk = %d, tx = %d, ty = %d, bx = %d, by = %d\n", b, chunk, tx, ty, bx, by);
        foundMatch = 0;
        hash_diverged = 0;
        diverge_index = -1;
        for (int i = 0; i< 256/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
            if(shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].valid == 1){ //Check whether there is anything valid and partial
                int idx;
                if (b == 6 && bx == 0 && by == 0) 
                    printf("Trying to find match b = %d, i = %d, tx = %d, ty = %d, bx = %d, by = %d\n", b, i, tx, ty, bx, by);
                idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk, 0, tx, ty, i);
                if (idx != -1) { 
                    foundMatch = 1;
                    if (b == 6 && bx == 0 && by == 0) 
                        printf("Found match b = %d, i = %d, tx = %d, ty = %d, bx = %d, by = %d\n", b, i, tx, ty, bx, by);
                    temp_hash = approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+0)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+1)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+2)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+3)];

                    if (sizeQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue) > 1) {//This means already compressed
                        if ((shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - temp_hash) <= 0.1 && (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - temp_hash) >= -0.1){  
                            //Set the new hash even though it just slightly off
                            shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] = temp_hash;
                        }
                        else {
                            //Allocate
                            int new_entry = 0;
                            int allocated = 0;
                            hash_diverged = 1;
                            diverge_index = i;
                            
                            /*while (allocated == 0 && new_entry < 256/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE){
                                if (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+new_entry].valid == 0) {
                                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+new_entry].valid = 1;
                                    temp_hash = approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+0)] +
                                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+1)] +
                                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+2)] +
                                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+3)];
                                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+new_entry].hash[0] = temp_hash;
                                    clearQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+new_entry].vaddrQueue);
                                    pushQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+new_entry].vaddrQueue, chunk);
                                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+new_entry].status = 15;
                                    allocated = 1;
                                }
                                new_entry++;
                            }
                            if (allocated == 0) {
                                printf("Couldn't find free entry after hash_diverged - something is wrong, bank = %d, tx = %d, ty= %d\n", b, tx, ty);
                            }*/
                        }
                    }
                    else {
                        //Do nothing just update the hash. No need to check if hash matched or not, just update it regardless
                        shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] = temp_hash;
                    }
                }
            }
        }
        if (b == 6 && bx == 0 && by == 0) 
            printf("FoundMatch = %d, hash_diverged = %d\n", foundMatch, hash_diverged);
        if(hash_diverged == 1)
        {
            int minIdx = getMinValueIndexQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue);
            int idx;
            if (b == 6 && bx == 0 && by == 0) 
                idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk, 1, tx, ty, 0);
            else
                idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk, 0, tx, ty, 0);
            
            
            if (idx == -1) {
                if (b == 6 && bx == 0 && by == 0) 
                    printf("Something is wrong since idx cannot be -1 here, bank = %d, tx = %d, ty = %d, i = %d, chunk = %d\n", b, tx, ty, 0, chunk);
            }
            else
            {
                if(minIdx == idx)
                {
                    //case 1: minIdx == idx: shift other elements to new min; 
                    int size = sizeQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue);

                    //get values in the addr queue of the chunk that diverged
                    struct queueNode *tempQueue = NULL;
                    for(int j=0; j<size; j++)
                        if(j != minIdx)
                            pushQueue(&tempQueue, getValueQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, j));

                    // find new min value in the queue (essentially, the second min value in the original queue) 
                    int newMinIdx = getMinValueIndexQueue(&tempQueue);
                    int shmemEntryIndex = getValueQueue(&tempQueue, newMinIdx);
                    // clear the queue at the existing index and reassign to temp queue
                    clearQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].vaddrQueue);
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].vaddrQueue = tempQueue;
                    //update hash
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].hash[0] = shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].hash[0];
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].valid = 1;
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].status = 15;
                    

                    // update the queue of the chunk that diverged: clear existing queue and push chunk addr to queue
                    clearQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue);
                    pushQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk);
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].hash[0] = temp_hash;
                }
                else
                {
                    //case 2: minIdx != idx: move idx to original location (chunk) & remove idx from vaddrQueue
                    clearQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+chunk].vaddrQueue);
                    pushQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+chunk].vaddrQueue, chunk);
                    delFromQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk);

                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+chunk].valid = 1;
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+chunk].status = 15;
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+chunk].hash[0] = temp_hash;
                }
            }
        }
        
        if (foundMatch == 0) {
            int i = 0;
            int allocated = 0;
            while (allocated == 0 && i < 256/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE){
                if (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].valid == 0) {
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].valid = 1;
                    temp_hash = approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+0)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+1)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+2)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+3)];
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] = temp_hash;
                    clearQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue);
                    pushQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk);
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].status = 15;
                    allocated = 1;
                    if (b == 6 && bx == 0 && by == 0) 
                        printf("Allocated b = %d, i = %d, tx = %d, ty = %d, bx = %d, by = %d, chunk = %d\n", b, i, tx, ty, bx, by, chunk);
                }
                i++;
            }
            if (allocated == 0) {
                printf("Couldn't find free entry for first time - something is wrong, bank = %d, tx = %d, ty= %d\n", b, tx, ty);
            }
        }
        for (int i = 0; i< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
            if (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].valid == 0)
                continue;
            for (int j = i+1; j< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; j++){
                if (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].valid == 0)
                    continue;
                //FIXME check if status is full as well before comparing hashes
                if ((shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].hash[0]) <= 0.1 && (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].hash[0]) >= -0.1){  
                    if (b == 6 && bx == 0 && by == 0) 
                        printf("Hash matched in the conservative check so some magic will happen\n");
                    for (int k = 0; k < sizeQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue); k++)
                        pushQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, getValueQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue, k));
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].valid = 0;
                    clearQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue);
                }
            }
        }
    }
    if (b == 6 && bx == 0 && by == 0)  {
        printf("End of function\n");
    }
    
}

__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
        float *output_hidden_cuda,
        float *input_hidden_cuda,
        float *hidden_partial_sum,
        int in,
        int hid,
        float **approx_shmem,
        struct shmemTableEntry **shmemTable) 
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
    //float approx_shmem[SHMEM_NUM_BANKS][SHMEMSIZE/4/SHMEM_NUM_BANKS]; //32 banks and 384 elements per bank
    //struct shmemTableEntry shmemTable[SHMEM_NUM_BANKS][SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE]; //for all 32 banks, entries per chunk
    int APPROX = 1;
    long unsigned int vaddr, vaddr_orig;
    int bank, offset;

    //FIXME do a shmem table init now by using all threads
    //Set paddr based on the loop index - a constant paddr per entry for simple allocation
    //Set all valid to 0, 
    //FIXME Confirm if this is correct

    if(ty < 2){
        for(int i = 0; i < SHMEM_TABLE_NUM_ENTRIES; i++) {
            shmemTable[by][(ty*16+tx)*SHMEM_TABLE_NUM_ENTRIES + i].paddr = i;
            shmemTable[by][(ty*16+tx)*SHMEM_TABLE_NUM_ENTRIES + i].vaddrQueue = NULL;
        }
    }

    /*if (tx ==0 && ty == 0) {
        vaddr = (long unsigned int)&weight_matrix[ty][tx];
        vaddr_orig = (long unsigned int)&weight_matrix[ty][tx];
        printf("Single by = %d, tx = %d, ty = %d, bank = %d, vaddr_original = %lx, vaddr = %lx, offset = %lx\n", by, tx, ty, bank, &weight_matrix[ty][tx], vaddr ,offset);
        vaddr = vaddr >> 2;
        printf("Single by = %d, tx = %d, ty = %d, bank = %d, vaddr_original = %lx, vaddr = %lx, offset = %lx\n", by, tx, ty, bank, &weight_matrix[ty][tx], vaddr ,offset);
        bank = vaddr % 32;
        printf("Single by = %d, tx = %d, ty = %d, bank = %d, vaddr_original = %lx, vaddr = %lx, offset = %lx\n", by, tx, ty, bank, &weight_matrix[ty][tx], vaddr ,offset);
        vaddr = vaddr >> 5;
        printf("Single by = %d, tx = %d, ty = %d, bank = %d, vaddr_original = %lx, vaddr = %lx, offset = %lx\n", by, tx, ty, bank, &weight_matrix[ty][tx], vaddr ,offset);
        offset = (long unsigned)(vaddr & 0x1ff);
        printf("Single bank = %d, vaddr_original = %lx, vaddr = %lx, offset = %lx, SHMEM_ELEMENTS = %d \n", bank, vaddr_orig, vaddr ,offset, SHMEM_ELEMENTS_PER_BANK);
    }
    //vaddr = (long unsigned int)&weight_matrix[ty][tx];
    //vaddr = vaddr >> 2;
    //bank = vaddr % 32;
    //vaddr = vaddr >> 5;
    //offset = (long unsigned)(vaddr & 0x1ff);
    //printf("by = %d, tx = %d, ty = %d, bank = %d, vaddr_original = %lx, vaddr = %lx, offset = %lx, SHMEM_ELEMENTS = %d \n", by, tx, ty, bank, vaddr_orig, vaddr ,offset, SHMEM_ELEMENTS_PER_BANK);
    */
    //tx - 0 to 15 & ty = 0, banks 0 to 15 and offset 0
    //tx - 0 to 15 & ty = 1, banks 16 to 31 and offset 0
    //tx - 0 to 15 & ty = 2, banks 0 to 15 and offset 1
    //tx - 0 to 15 & ty = 3, banks 16 to 31 and offset 1
    //tx - 0 to 15 & ty = 4, banks 0 to 15 and offset 2
    //tx - 0 to 15 & ty = 5, banks 16 to 31 and offset 2
    //tx - 0 to 15 & ty = 6, banks 0 to 15 and offset 3
    //tx - 0 to 15 & ty = 7, banks 16 to 31 and offset 3
    //tx - 0 to 15 & ty = 8, banks 0 to 15 and offset 4
    //tx - 0 to 15 & ty = 9, banks 16 to 31 and offset 4
    //tx - 0 to 15 & ty = 10, banks 0 to 15 and offset 5
    //tx - 0 to 15 & ty = 11, banks 16 to 31 and offset 5
    bank = (ty*16+tx)%32;
    offset = (int)(ty/2);
    //printf("by = %d, tx = %d, ty = %d, bank = %d, offset = %lx \n", by, tx, ty, bank, offset);

    __syncthreads();

    if ( tx == 0 )
        input_node[ty] = input_cuda[index_in];

    //cout << "inputNode for ty= " << ty << "is " input_node[ty] << "\n";
    // printf("inputNode for ty= %d is %d\n", ty, input_node[ty]);
    __syncthreads();

    if (APPROX) {
        weight_matrix[ty][tx] = input_hidden_cuda[index];
        approx_shmem[by][bank*SHMEM_ELEMENTS_PER_BANK+offset] = weight_matrix[ty][tx];
    }
    else {
        weight_matrix[ty][tx] = input_hidden_cuda[index];
        approx_shmem[by][bank*SHMEM_ELEMENTS_PER_BANK+offset] = weight_matrix[ty][tx];
    }

    //Call sync threads and compress. Then call sync threads again
    __syncthreads();
    if ( ty < 2) {
        approxShmem(approx_weight_matrix, approx_shmem, shmemTable, bx, by, tx, ty, bank, &weight_matrix[ty][tx]); 
    }

    //printf("Weight matrix for ty= %d and tx = %d is %d\n", ty, tx, weight_matrix[ty][tx]);
    //if (smID == 0)
    //    printf("WRITE bx,by,tx,ty,smID=%d,%d,%d,%d,%d weight_matrx=%f,weigh_matrix_address=%x\n", bx, by, tx, ty, smID, weight_matrix[ty][tx], &weight_matrix[ty][tx]);

    __syncthreads();
    if (bank == 0 && by == 0 && bx == 0 && ty < 2)
        printShmemTable(shmemTable, bank, by, tx, ty);

    if (APPROX) {
        weight_matrix[ty][tx] = getApproxShmem(offset, bank, approx_shmem, shmemTable, by, tx, ty) * input_node[ty];
        approx_shmem[by][bank*SHMEM_ELEMENTS_PER_BANK+offset] = weight_matrix[ty][tx];
    }
    else {
        weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
        approx_shmem[by][bank*SHMEM_ELEMENTS_PER_BANK+offset] = weight_matrix[ty][tx];
    }

    __syncthreads();   
    if ( ty < 2) {
        approxShmem(approx_weight_matrix, approx_shmem, shmemTable, bx, by, tx, ty, bank, &weight_matrix[ty][tx]); 
    }
    __syncthreads();   
    if (bank == 0 && by == 0 && bx == 0 && ty < 2)
        printShmemTable(shmemTable, bank, by, tx, ty);

    for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){

        int power_two = __powf(2, i);

        if( ty % power_two == 0 ) {
            if (APPROX) {
                weight_matrix[ty][tx] = getApproxShmem(offset, bank, approx_shmem, shmemTable, by, tx, ty) + getApproxShmem(offset, bank, approx_shmem, shmemTable, by, tx, ty+power_two/2);
                approx_shmem[by][bank*SHMEM_ELEMENTS_PER_BANK+offset] = weight_matrix[ty][tx];
            }
            else {
                weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
                approx_shmem[by][bank*SHMEM_ELEMENTS_PER_BANK+offset] = weight_matrix[ty][tx];
            }
        }
        __syncthreads();

    }

    if ( ty < 2) {
        approxShmem(approx_weight_matrix, approx_shmem, shmemTable, bx, by, tx, ty, bank, &weight_matrix[ty][tx]); 
    }

    __syncthreads();   
    if (bank == 0 && by == 0 && bx == 0 && ty < 2)
        printShmemTable(shmemTable, bank, by, tx, ty);

    //__syncthreads();

    if (APPROX) {
        input_hidden_cuda[index] = getApproxShmem(offset, bank, approx_shmem, shmemTable, by, tx, ty);
    }
    else {
        input_hidden_cuda[index] = weight_matrix[ty][tx];
    }

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
        if (APPROX) {
            hidden_partial_sum[by * hid + ty] = getApproxShmem(offset, bank, approx_shmem, shmemTable, by, tx, ty);
        }
        else {
            hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
        }
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
