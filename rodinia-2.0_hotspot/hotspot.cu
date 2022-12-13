#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "timer.h"
#define BLOCK_SIZE 32
#define STR_SIZE 256
#define SHARED_VARIABLES_IN_KERNEL 3
#include "hotspot.h"
#define HASH_ERROR 0.1

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)


__device__ int get_smid(void) {
     int ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

__device__ int isOneThreadPerBank(int ty){
    if (BLOCK_SIZE==32){
       if (ty == 0)
            return 1;
       else
            return 0;
    }
    else if (BLOCK_SIZE==16){
       if (ty < 2)
            return 1;
       else
            return 0;
    }
    else
        return 0;
    return 0;
}

__device__ int get_bank(int ty, int tx){
    if (BLOCK_SIZE==32){
        return (tx);
    }
    else if (BLOCK_SIZE==16){
        return ((ty*16+tx)%32);
    }
    else
        return 0;
}

__device__ int get_offset(int ty, int tx, int n){
    if (BLOCK_SIZE==32){
        return (ty+n*APP_SHMEM_ELEMENTS_PER_BANK/SHARED_VARIABLES_IN_KERNEL);
    }
    else if (BLOCK_SIZE==16){
        return ((ty/2)+n*APP_SHMEM_ELEMENTS_PER_BANK/SHARED_VARIABLES_IN_KERNEL);
    }
    else
        return 0;
}

/*__device__ float get_hash(float **approx_shmem, int blk, int bank, int chunk)
{
    float hash = 0;
    for (int i = 0; i < SHMEM_CHUNK_SIZE; i++){
        hash+=approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+(chunk*SHMEM_CHUNK_SIZE+i)];
    }
    return hash;
}*/

__device__ int get_fp8(float num){
    
    int fp8 = 0;
    int *numptr =(int*)&num;
    int inum = *numptr;
    //int bit_array[8] = {32, 31, 26, 25, 24, 23, 22, 21};
    int bit_array[8] = {20, 21, 22, 23, 24, 25, 30, 31};
    for (int i = 0; i < 8; i++){
       int bit = inum & (1 << bit_array[i]);
       if (bit){
           fp8 = fp8 | (1 << (i)); 
       }
    }
    //printf("num = %f, inum = %x fp8 = %x\n", num, inum, fp8);
    return fp8;
}

__device__ float get_hash(float **approx_shmem, int blk, int bank, int chunk)
{
    int hash[SHMEM_CHUNK_SIZE];
    int hash_concat = 0;
    float *hashptr;
    for (int i = 0; i < SHMEM_CHUNK_SIZE; i++){
        hash[i] = get_fp8(approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+(chunk*SHMEM_CHUNK_SIZE+i)]);
        //printf("i = %d, val = %f, hash_i = %d\n", i, approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+(chunk*SHMEM_CHUNK_SIZE+i)],hash[i]);
    }
    for (int i = 0; i < SHMEM_CHUNK_SIZE; i++){
        hash_concat = hash_concat | hash[i] << (i*8);
    }
    hashptr = (float*)&hash_concat;
    //printf("hash_concat = %d, hash_float = %f\n", hash_concat, *hashptr);
    return *hashptr;
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
            printf("In findInQueue, index = %d, i = %d, tx = %d, ty= %d, ptr->vaddr = %ld, vaddr = %ld\n", index, i, tx, ty, ptr->vaddr,  vaddr);
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
		//free(ptr);
		return;
	}
	while(ptr!= NULL)
	{
		if(pos == index)
		{
			prev->next = ptr->next;
			//free(ptr);
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
        //free(ptr);
        ptr = *q;
    }

}
__device__ void printQueue(struct queueNode *q, int i, int tx, int ty)
{
    struct queueNode *ptr = q;
    //printf("Printing queue i = %d, tx = %d, ty = %d", i, tx, ty);
    while(ptr!=NULL) {
        printf("%ld,", ptr->vaddr);
        ptr = ptr->next;
    }
    //printf("\n");
}

__device__ void printShmemTable(struct shmemTableEntry **shmemTable , int bank, int by, int tx, int ty, float **approx_shmem)
{
    printf("----------------------------------SHMEM TABLE START--------------------------------------\n");
    for (int b = 0; b < SHMEM_NUM_BANKS; b++) {
        bank = b;
        for (int i= 0; i<APP_SHMEM_TABLE_ENTRIES_PER_BANK; i++){
            printf("shmemTable by = %d, tx = %d, ty = %d, bank = %d, valid = %d, vaddr = ",by, tx, ty, bank, shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].valid);
            printQueue(shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, 0, tx, ty);
            printf(" paddr = %d, hash[0] = %f, status = %d\n", shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].paddr, shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].hash[0], shmemTable[by][bank*SHMEM_TABLE_NUM_ENTRIES+i].status);
        }
    }
    printf("----------------------------------SHMEM TABLE END--------------------------------------\n");

    /*printf("----------------------------------SHMEM START--------------------------------------\n");
    for(int b = 0; b < SHMEM_NUM_BANKS; b++){
        printf("Shem[%d][%d] = ", by, b);
        for (int i = 0; i<APP_SHMEM_ELEMENTS_PER_BANK; i++){
            printf("%f, ", approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+i]);
        }
        printf("\n");
    }
    printf("----------------------------------SHMEM END--------------------------------------\n");
    */
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


__device__ float getApproxShmem(int input_vaddr, int b, float **approx_shmem, struct shmemTableEntry **shmemTable, int blk, int tx, int ty) {
    
    //b = bank
    float value = 0;
    int chunk = input_vaddr/SHMEM_CHUNK_SIZE;
    int match = 0;
    
    for (int i = 0; i < APP_SHMEM_TABLE_ENTRIES_PER_BANK; i++){
        if (shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].valid) {
            //printf("In getApproxShmem, Found valid entry\n");
            //printf("Vaddr = %x\n", vaddr>>2);
            int idx = findInQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk, 0, tx, ty, i);
            if (idx != -1) {  //There is a match so check for status
                //there is match
                //printf("In getApproxShmem, Found entry in queue\n");
                value = approx_shmem[blk][b*SHMEM_ELEMENTS_PER_BANK+ ((shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].paddr << SHMEM_CHUNK_BITS) + (input_vaddr & (SHMEM_CHUNK_SIZE-1)))];
                //if (b == 7)
                //    printf("Returning for read, chunk = %d, b = %d, tx = %d, ty = %d, input_vaddr = %d, blk = %d, value = %f\n", chunk, b, tx, ty, input_vaddr, blk, value);
                match = 1;
                break;
            }
        }
    }
    if (match == 0) {
        printf("ERROR: There is no match in the vaddrQueue for reads, chunk = %d, b = %d, tx = %d, ty = %d, input_vaddr = %d, blk = %d\n", chunk, b, tx, ty, input_vaddr, blk);
    }
    //printf("get approxShmem, value = %f\n", value);
    return value;
}

__device__ void approxShmem(float **approx_shmem, struct shmemTableEntry **shmemTable, int blk, int tx, int ty, int bank, int print, int totalCompress[], int totalDiverge[], int currentCompress[]){
   
    int b; //b = bank
    int hash_diverged = 0;
    int diverge_index = -1;
    int foundMatch = 0;
    float temp_hash;
    hash_diverged = 0;
    diverge_index = -1;
    foundMatch = 0;
    temp_hash = 0;
    b = bank;
    if (b == 20 && blk == 0 && print)  {
        printf("Start of function\n");
    }
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
    //FIXED - there is a problem with reads. If the hash diverges,  the 0th entry with vaddr zero can move to 1st entry with vaddr 0. It points to paddr 1. However in actual shmem, the entry is still at location 0. Unless there is second copy of the shared memory which reflects data exactly as it is in the shmem_Table, there will be an issue in reading approximate data. 
    //FIXED - the hash looks like is being calculated incorrectly. In the third iteration of writes, all the hashes for which there is compression are exactly same. Need to check. Could be related to the problem above
    for (int chunk = 0; chunk < APP_SHMEM_TABLE_ENTRIES_PER_BANK; chunk++){
        if (b == 20 && blk == 0 && print)  {
            printf("\nStarting new iteration b = %d, chunk = %d, tx = %d, ty = %d, blk = %d\n", b, chunk, tx, ty, blk);
        }
        foundMatch = 0;
        hash_diverged = 0;
        diverge_index = -1;
        for (int i = 0; i< APP_SHMEM_TABLE_ENTRIES_PER_BANK; i++){
            if(shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].valid == 1){ //Check whether there is anything valid and partial
                int idx;
                if (b == 20 && blk == 0 && print)  {
                    printf("Trying to find match b = %d, i = %d, tx = %d, ty = %d, blk = %d\n", b, i, tx, ty, blk);
                }
                idx = findInQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk, 0, tx, ty, 0);
                if (idx != -1) { 
                    foundMatch = 1;
                    if (b == 20 && blk == 0 && print)  {
                        printf("Found match b = %d, i = %d, tx = %d, ty = %d, blk = %d\n", b, i, tx, ty, blk);
                    }
                    temp_hash = get_hash(approx_shmem, blk, b, chunk);

                    if (sizeQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue) > 1) {//This means already compressed
                        if ((shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - temp_hash) <= HASH_ERROR && (shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - temp_hash) >= (-1*HASH_ERROR)){  
                            //Set the new hash even though it just slightly off
                            shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] = temp_hash;
                        }
                        else {
                            //Allocate
                            hash_diverged = 1;
                            diverge_index = i;
                            if (b == 20 && blk == 0 && print)  {
                                printf("Hash diverged, b = %d, i = %d, tx = %d, ty = %d, blk = %d, temp_hash = %f\n", b, i, tx, ty, blk, temp_hash);
                            }
                            currentCompress[0]--;
                            totalDiverge[0]++;
                        }
                    }
                    else {
                        //Do nothing just update the hash. No need to check if hash matched or not, just update it regardless
                        shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] = temp_hash;
                    }
                }
            }
        }
        if (b == 20 && blk == 0 && print)  {
            printf("FoundMatch = %d, hash_diverged = %d\n", foundMatch, hash_diverged);
        }
        if(hash_diverged == 1)
        {
            int minIdx = getMinValueIndexQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue);
            int idx;
            idx = findInQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk, 0, tx, ty, 0);
            
            if (idx == -1) {
                //if (blk == 0 && b == 20)  {
                printf("ERROR: Something is wrong since idx cannot be -1 here, blk = %d, bank = %d, tx = %d, ty = %d, i = %d, chunk = %d\n", blk, b, tx, ty, 0, chunk);
                //}
            }
            else
            {
                if(minIdx == idx)
                {
                    if (blk == 0 && b == 20 && print)  {
                        printf("Inside minIdx == idx case - blk = %d, bank = %d, tx = %d, ty = %d, i = %d, chunk = %d, diverge_index = %d\n", blk, b, tx, ty, 0, chunk, diverge_index);
                    }

                    //case 1: minIdx == idx: shift other elements to new min; 
                    int size = sizeQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue);

                    //get values in the addr queue of the chunk that diverged
                    struct queueNode *tempQueue = NULL;
                    for(int j=0; j<size; j++)
                        if(j != minIdx)
                            pushQueue(&tempQueue, getValueQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, j));

                    // find new min value in the queue (essentially, the second min value in the original queue) 
                    int newMinIdx = getMinValueIndexQueue(&tempQueue);
                    int shmemEntryIndex = getValueQueue(&tempQueue, newMinIdx);
                    if (blk == 0 && b == 20 && print)  {
                        printf("NewMinIdx = %d, shmemEntryIndex = %d, tempQueue=", newMinIdx, shmemEntryIndex);
                        printQueue(tempQueue, 0, tx, ty);
                        printf("\n");
                    }
                    // clear the queue at the existing index and reassign to temp queue
                    clearQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].vaddrQueue);
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].vaddrQueue = tempQueue;
                    //update hash
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].hash[0] = shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].hash[0];
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].valid = 1;
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+shmemEntryIndex].status = 15;

                    // update the queue of the chunk that diverged: clear existing queue and push chunk addr to queue
                    clearQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue);
                    pushQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk);
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].hash[0] = temp_hash;
                    if (blk == 0 && b == 20 && print)  {
                        printf("NewMinIdx = %d, shmemEntryIndex = %d, divergeIndexHash = %f, divergeIndexVaddrQueue=", newMinIdx, shmemEntryIndex, shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].hash[0]);
                        printQueue(shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, 0, tx, ty);
                        printf("\n");
                    }
                }
                else
                {

                    if (blk == 0 && b == 20 && print)  {
                        printf("Inside minIdx != idx case - blk = %d, bank = %d, tx = %d, ty = %d, i = %d, chunk = %d, diverge_index=%d\n", blk, b, tx, ty, 0, chunk, diverge_index);
                    }
                    //case 2: minIdx != idx: move idx to original location (chunk) & remove idx from vaddrQueue
                    clearQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+chunk].vaddrQueue);
                    pushQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+chunk].vaddrQueue, chunk);
                    delFromQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, idx);
                    if (blk == 0 && b == 20 && print)  {
                        printf("chunk = %d,  chunk indx Queue=", chunk);
                        printQueue(shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+chunk].vaddrQueue, 0, tx, ty);
                        printf("\n");
                        printf("diverge_index = %d,  diverge_index Queue=", diverge_index);
                        printQueue(shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, 0, tx, ty);
                        printf("\n");
                    }

                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+chunk].valid = 1;
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+chunk].status = 15;
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+chunk].hash[0] = temp_hash;
                }
            }
        }
        
        if (foundMatch == 0) {
            int i = 0;
            int allocated = 0;
            while (allocated == 0 && i < APP_SHMEM_TABLE_ENTRIES_PER_BANK){
                if (shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].valid == 0 && i == chunk) {
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].valid = 1;
                    temp_hash = get_hash(approx_shmem, blk, b, chunk);
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] = temp_hash;
                    clearQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue);
                    pushQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk);
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].status = 15;
                    allocated = 1;
                    if (b == 20 && blk == 0 && print) {
                        printf("Allocated b = %d, i = %d, tx = %d, ty = %d, blk = %d, chunk = %d\n", b, i, tx, ty, blk, chunk);
                    }
                }
                i++;
            }
            if (allocated == 0) {
                printf("ERROR: Couldn't find free entry for first time - something is wrong, bank = %d, tx = %d, ty= %d\n", b, tx, ty);
            }
        }
        for (int i = 0; i< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; i++){
            if (shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].valid == 0)
                continue;
            for (int j = i+1; j< SHMEMSIZE/4/SHMEM_NUM_BANKS/SHMEM_CHUNK_SIZE; j++){
                if (shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].valid == 0)
                    continue;
                if ((shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].hash[0]) <= HASH_ERROR && (shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].hash[0]) >= (-1*HASH_ERROR)){  
                    if (b == 20 && blk == 0 && print) {
                        printf("Hash matched in the conservative check so some magic will happen\n");
                    }
                    for (int k = 0; k < sizeQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue); k++)
                        pushQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, getValueQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue, k));
                    shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].valid = 0;
                    totalCompress[0]++;
                    currentCompress[0]++;
                    clearQueue(&shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue);
                    if (blk == 0 && b == 20 && print)  {
                        printf("i =%d, hash= %f, VaddrQueue=", i, shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0]);
                        printQueue(shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, 0, tx, ty);
                        printf("\n");
                        printf("j =%d, hash= %f, VaddrQueue=", j, shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].hash[0]);
                        printQueue(shmemTable[blk][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue, 0, tx, ty);
                        printf("\n");
                    }
                }
            }
        }
    }
    if (b == 20 && blk == 0 && print)  {
        printf("End of function\n");
    }
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file){

	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];

	if( (fp = fopen(file, "w" )) == 0 )
          printf( "The file was not opened\n" );


	for (i=0; i < grid_rows; i++) 
	 for (j=0; j < grid_cols; j++)
	 {

		 sprintf(str, "%d\t%.6f\n", index, vect[i*grid_cols+j]);
		 fputs(str,fp);
		 index++;
	 }
		
      fclose(fp);	
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file){

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            printf( "The file was not opened\n" );


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fatal("invalid file format");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
							   int border_cols,  // border offset 
							   int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx, 
                               float Ry, 
                               float Rz, 
                               float step, 
                               float time_elapsed,
                               float** approx_shmem,
                               struct shmemTableEntry** shmemTable,
                               int blockCols){
	
    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    int APPROX = 1;
    int print = 0;

	float amb_temp = 80.0;
    float step_div_Cap;
    float Rx_1,Ry_1,Rz_1;
        
	int bx = blockIdx.x;
    int by = blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	step_div_Cap=step/Cap;
	
	Rx_1=1/Rx;
	Ry_1=1/Ry;
	Rz_1=1/Rz;
	
    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkY = small_block_rows*by-border_rows;
    int blkX = small_block_cols*bx-border_cols;
    int blkYmax = blkY+BLOCK_SIZE-1;
    int blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

    // load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
    int index = grid_rows*loadYidx+loadXidx;

    int bank = get_bank(ty, tx);
    int offset_temp_on_cuda = get_offset(ty, tx, 0);
    int offset_power_on_cuda = get_offset(ty, tx, 1);
    int offset_temp_t = get_offset(ty, tx, 2);
    int blk = by*blockCols+bx;
    int totalCompress[1];
    int totalDiverge[1];
    int currentCompress[1];
    totalCompress[0] = 0;
    totalDiverge[0] = 0;
    currentCompress[0] = 0;

    //printf("ty = %d, tx = %d, bank = %d, offset_temp_on_cuda = %d, offset_power = %d, offset_temp_t = %d, blk = %d\n", ty, tx, bank, offset_temp_on_cuda, offset_power_on_cuda, offset_temp_t, blk);

    if(isOneThreadPerBank(ty)){
        for(int i = 0; i < SHMEM_TABLE_NUM_ENTRIES; i++) {
            shmemTable[blk][(bank)*SHMEM_TABLE_NUM_ENTRIES + i].paddr = i;
            clearQueue(&shmemTable[blk][(bank)*SHMEM_TABLE_NUM_ENTRIES + i].vaddrQueue);
            shmemTable[blk][(bank)*SHMEM_TABLE_NUM_ENTRIES + i].vaddrQueue = NULL;
            shmemTable[blk][(bank)*SHMEM_TABLE_NUM_ENTRIES + i].valid = 0;
            shmemTable[blk][(bank)*SHMEM_TABLE_NUM_ENTRIES + i].hash[0] = 0;
        }
    }

	__syncthreads();
       
	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp[index];  // Load the temperature data from global memory to shared memory
            approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+offset_temp_on_cuda] = temp_on_cuda[ty][tx];
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
            approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+offset_power_on_cuda] = power_on_cuda[ty][tx];
	}

	__syncthreads();

    
    if ( isOneThreadPerBank(ty)) {
        approxShmem(approx_shmem, shmemTable, blk, tx, ty, bank, print, totalCompress, totalDiverge, currentCompress); 
    }

    __syncthreads();
    if (blk == 0 && tx == 0 && ty == 0) {
        printShmemTable(shmemTable, bank, blk, tx, ty, approx_shmem);
        //int num = get_fp8(0.4532);
        //float hash = get_hash_2(approx_shmem, 0, 6, 1);
        //printf("num = %d, hash = %f\n", num, hash);
    }

    __syncthreads();

    // effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

    int N = ty-1;
    int S = ty+1;
    int W = tx-1;
    int E = tx+1;
    
    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    /*if ((ty == 8 || ty == 6) && tx == 7){
        printf("N=%d, S=%d, W=%d, E=%d\n", N, S, W, E);
    }*/

    bool computed;
    if (APPROX) {
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  float temp_1 = getApproxShmem(get_offset(S, tx, 0), get_bank(S, tx), approx_shmem, shmemTable, blk, tx, ty) + getApproxShmem(get_offset(N, tx, 0), get_bank(N, tx), approx_shmem, shmemTable, blk, tx, ty) - 2.0*getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty);
                  //float temp_1_1 = getApproxShmem(int(S/2), (S*16+tx)%32, approx_shmem, shmemTable, blk, tx, ty);
                  //float temp_1_2 = getApproxShmem((int)N/2, (N*16+tx)%32, approx_shmem, shmemTable, blk, tx, ty);
                  //float temp_1_3 = getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty);
                  float temp_2 = (getApproxShmem(get_offset(ty, E, 0), get_bank(ty, E), approx_shmem, shmemTable, blk, tx, ty) + getApproxShmem(get_offset(ty, W, 0), get_bank(ty, W), approx_shmem, shmemTable, blk, tx, ty) - 2.0*getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty));
                  float temp_3 = (amb_temp - getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty));
                  temp_t[ty][tx] =   getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty) + step_div_Cap * (getApproxShmem(offset_power_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty) + (temp_1)* Ry_1 + (temp_2)  * Rx_1 +  (temp_3)* Rz_1);
                  /*if ((ty == 8 || ty == 6) && tx == 7){
                    printf("temp_t[%d][%d] = %f, temp_1 =%f, temp_1_1 = %f, temp_1_2 = %f, temp_1_3 = %f, temp_2 = %f, temp_3 = %f\n", ty, tx, temp_t[ty][tx], temp_1, temp_1_1, temp_1_2, temp_1_3,temp_2, temp_3);
                  }*/
                 approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+offset_temp_t] = temp_t[ty][tx];
        
            }
            __syncthreads();

            if(isOneThreadPerBank(ty)){
                approxShmem(approx_shmem, shmemTable, blk, tx, ty, bank, print, totalCompress, totalDiverge, currentCompress); 
            }
            __syncthreads();

            if (bank == 0 && blk == 0 && tx == 0 && ty == 0 && print) {
                printShmemTable(shmemTable, bank, blk, tx, ty, approx_shmem);
            }
            
            __syncthreads();


            if(i==iteration-1)
                break;
            if(computed){	 //Assign the computation range
                temp_on_cuda[ty][tx]= getApproxShmem(offset_temp_t, bank, approx_shmem, shmemTable, blk, tx, ty);
                approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+offset_temp_on_cuda] = temp_on_cuda[ty][tx];
            }
            __syncthreads();

            if(isOneThreadPerBank(ty)){
                approxShmem(approx_shmem, shmemTable, blk, tx, ty, bank, print, totalCompress, totalDiverge, currentCompress); 
            }
            __syncthreads();

            if (bank == 0 && blk == 0 && tx == 0 && ty == 0 && print) {
                printShmemTable(shmemTable, bank, blk, tx, ty, approx_shmem);
            }
          }

          __syncthreads();

          // update the global memory
          // after the last iteration, only threads coordinated within the 
          // small block perform the calculation and switch on ``computed''
          if (computed){
              temp[index]= getApproxShmem(offset_temp_t, bank, approx_shmem, shmemTable, blk, tx, ty);		
          }

          __syncthreads();
    }
    else {
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  float temp_1 = temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx];
                  /*float temp_1_1 = temp_on_cuda[S][tx];
                  float temp_1_2 = temp_on_cuda[N][tx];
                  float temp_1_3 = temp_on_cuda[ty][tx];*/
                  float temp_2 = temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx];
                  float temp_3 = amb_temp - temp_on_cuda[ty][tx];
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
                     (temp_1) * Ry_1 + 
                     (temp_2) * Rx_1 + 
                     (temp_3) * Rz_1);
                  /*if ((ty == 8 || ty == 6) && tx == 7){
                    printf("temp_t[%d][%d] = %f, temp_1 =%f, temp_1_1 = %f, temp_1_2 = %f, temp_1_3 = %f, temp_2 = %f, temp_3 = %f\n", ty, tx, temp_t[ty][tx], temp_1, temp_1_1, temp_1_2, temp_1_3,temp_2, temp_3);
                  }*/
        
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }

          // update the global memory
          // after the last iteration, only threads coordinated within the 
          // small block perform the calculation and switch on ``computed''
          if (computed){
              temp[index]= temp_t[ty][tx];		
          }
    }
    if (isOneThreadPerBank(ty))
        printf("blk = %d, bank = %d, totalCompress_addr = %p, TotalCompress = %d, totalDiverge = %d, CurrentCompress = %d\n", blk, bank, totalCompress, totalCompress[0], totalDiverge[0], currentCompress[0]);
}

/*
   compute N time steps
*/

void compute_tran_temp(float *MatrixPower,float *MatrixTemp, int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows, float **approx_shmem, struct shmemTableEntry **shmemTable) 
{
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(blockCols, blockRows);  
    printf("BLOCK_SIZE = %d, blockCols = %d, blockRows = %d, total_iterations = %d, num_iterations = %d\n", BLOCK_SIZE, blockCols, blockRows, total_iterations, num_iterations);
	
	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	float t = 0;
        float time_elapsed;
	time_elapsed=0.001;

	//for (t = 0; t < total_iterations; t+=num_iterations) {
	calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower,MatrixTemp,\
		col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed, approx_shmem, shmemTable, blockCols);
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
        else {
            printf("CUDA kernel success\n");
        }

	//}
}

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp,*FilesavingPower,*MatrixOut; 
    char tfile[]="./temp.dat";
    char pfile[]="./power.dat";
    char ofile[]="./output_pyramid.dat";
    const char* goldfile = NULL;
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
    if (argc >= 2)
    {
		grid_rows = atoi(argv[1]);
		grid_cols = atoi(argv[1]);
    }
    if (argc >= 3){
        pyramid_height = atoi(argv[2]);
	}
    if (argc >= 4) {
        total_iterations = atoi(argv[3]);
	}
	if (argc >= 5) {
		goldfile = NULL;
	}
    if (argc>=6) {
		printf("Wrong Usage\n");
		exit(0);
    }

    size=grid_rows*grid_cols;

    /* --------------- pyramid parameters --------------- */
    # define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp || !MatrixOut)
        fatal("unable to allocate memory");

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
	pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);
	
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

    float *MatrixTemp,*MatrixPower;
    cudaMalloc((void**)&MatrixTemp, sizeof(float)*size);
    cudaMemcpy(MatrixTemp, FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice);

    float **approx_shmem;
    struct shmemTableEntry **shmemTable;
    float **h_approx_shmem = (float**)malloc(blockCols*blockRows*sizeof(float*));
    struct shmemTableEntry **h_shmemTable = (struct shmemTableEntry**)malloc(grid_rows*grid_cols*sizeof(struct shmemTableEntry*));
    printf("Malloc 1, SHMEM_ELEMENTS_PER_BANK = %d, SHMEM_TABLE_NUM_ENTRIES = %d\n", SHMEM_ELEMENTS_PER_BANK, SHMEM_TABLE_NUM_ENTRIES);
    for (int blks = 0; blks < blockCols*blockRows; blks++) {
        cudaMalloc((void**)&h_approx_shmem[blks], (SHMEM_NUM_BANKS*SHMEM_ELEMENTS_PER_BANK) * sizeof(float));
        cudaMalloc((void**)&h_shmemTable[blks], (SHMEM_NUM_BANKS*SHMEM_TABLE_NUM_ENTRIES) * sizeof(struct shmemTableEntry));
    }
    printf("Malloc 2\n");

    cudaMalloc((void***)&approx_shmem, blockCols*blockRows*sizeof(float*));
    cudaMalloc((void***)&shmemTable, blockCols*blockRows*sizeof(struct shmemTableEntry*));
    cudaMemcpy(approx_shmem, h_approx_shmem, blockCols*blockRows*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shmemTable, h_shmemTable, blockCols*blockRows*sizeof(struct shmemTableEntry*), cudaMemcpyHostToDevice);
    printf("Malloc 3\n");


    unsigned long long cycles;
    pin_stats_reset();
    compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
	 total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows, approx_shmem, shmemTable);

    cudaMemcpy(MatrixOut, MatrixTemp, sizeof(float)*size, cudaMemcpyDeviceToHost);

    pin_stats_pause(cycles);
    pin_stats_dump(cycles);

    writeoutput(MatrixOut,grid_rows, grid_cols, ofile);

    cudaFree(MatrixTemp);
    free(MatrixOut);
	/*if(goldfile){
		FILE *gold = fopen(goldfile, "r");
		FILE *result = fopen(ofile, "r");
		int index_result=0, index_gold=0;
		float value_result=0.0, value_gold=0.0;
		int result_error=0;
		float total_error=0.0, avg_error=0.0;

		while( fscanf(gold,"%d\t%f\n",&index_gold,&value_gold)==2 && fscanf(result,"%d\t%f\n",&index_result,&value_result)==2 ) {
			if(index_gold != index_result) {
				result_error = 1;
				break;
			}
			float error = (value_gold - value_result) / value_gold;
			error = error>=0.0?error:-error;
			total_error += error;
		}
		avg_error = total_error/(float)(index_gold+1);
		printf("total_error=%.9f\navg_error=%.9f\n", total_error, avg_error);

		if(avg_error > 0.0001)
			result_error = 1;

		if((feof(gold)^feof(result)) | result_error) {
			printf("\nFAILED\n");
		} else {
			printf("\nPASSED\n");
		}

		fclose(gold);
		fclose(result);
	}*/

}
