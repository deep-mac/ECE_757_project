#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "timer.h"
#include "hotspot.h"
#define BLOCK_SIZE 16
#define STR_SIZE 256

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

#define HASH_ERROR 0.00000000001

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
        for (int i= 0; i<APP_SHMEM_TABLE_ENTRIES_PER_BANK; i++){
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
    int match = 0;
    //b = tx;
    //input_vaddr = (int)(ty/2);
    //int by = blockDim.y;
    
    for (int i = 0; i < APP_SHMEM_TABLE_ENTRIES_PER_BANK; i++){
        if (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].valid) {
            //printf("In getApproxShmem, Found valid entry\n");
            //printf("Vaddr = %x\n", vaddr>>2);
            int idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk, 0, tx, ty, 0);
            if (idx != -1) {  //There is a match so check for status
                //there is match
                //printf("In getApproxShmem, Found entry in queue\n");
                value = approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+ ((shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].paddr << 2) + (input_vaddr & 3))];
                match = 1;
                break;
            }
        }
    }
    if (match == 0) {
        //if (b == 0 && by == 53)
        printf("There is no match in the vaddrQueue for reads, chunk = %d, b = %d, tx = %d, ty = %d, input_vaddr = %d, blk = %d\n", chunk, b, tx, ty, input_vaddr, by);
    }
    //printf("get approxShmem, value = %f\n", value);
    return value;
}

__device__ void approxShmem(float weight[][BLOCK_SIZE], float **approx_shmem, struct shmemTableEntry **shmemTable, int bx, int by, int tx, int ty, int bank, float *input_vaddr){
   
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
    if (b == 0 && by == 53)  {
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
    for (int chunk = 0; chunk < APP_SHMEM_TABLE_ENTRIES_PER_BANK; chunk++){
        if (b == 0 && by == 53) 
            printf("Starting new iteration b = %d, chunk = %d, tx = %d, ty = %d, bx = %d, by = %d\n", b, chunk, tx, ty, bx, by);
        foundMatch = 0;
        hash_diverged = 0;
        diverge_index = -1;
        for (int i = 0; i< APP_SHMEM_TABLE_ENTRIES_PER_BANK; i++){
            if(shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].valid == 1){ //Check whether there is anything valid and partial
                int idx;
                if (b == 0 && by == 53) 
                    printf("Trying to find match b = %d, i = %d, tx = %d, ty = %d, bx = %d, by = %d\n", b, i, tx, ty, bx, by);
                idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, chunk, 0, tx, ty, i);
                if (idx != -1) { 
                    foundMatch = 1;
                    if (b == 0 && by == 53) 
                        printf("Found match b = %d, i = %d, tx = %d, ty = %d, bx = %d, by = %d\n", b, i, tx, ty, bx, by);
                    temp_hash = approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+0)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+1)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+2)] +
                                approx_shmem[by][b*SHMEM_ELEMENTS_PER_BANK+(chunk*4+3)];

                    if (sizeQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue) > 1) {//This means already compressed
                        if ((shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - temp_hash) <= HASH_ERROR && (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - temp_hash) >= (-1*HASH_ERROR)){  
                            //Set the new hash even though it just slightly off
                            shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] = temp_hash;
                        }
                        else {
                            //Allocate
                            int new_entry = 0;
                            int allocated = 0;
                            hash_diverged = 1;
                            diverge_index = i;
                            
                            /*while (allocated == 0 && new_entry < APP_SHMEM_TABLE_ENTRIES_PER_BANK){
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
        if (b == 0 && by == 53) 
            printf("FoundMatch = %d, hash_diverged = %d\n", foundMatch, hash_diverged);
        if(hash_diverged == 1)
        {
            int minIdx = getMinValueIndexQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue);
            int idx;
            if (b == 0 && by == 53) 
                idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk, 1, tx, ty, 0);
            else
                idx = findInQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+diverge_index].vaddrQueue, chunk, 0, tx, ty, 0);
            
            
            if (idx == -1) {
                if (b == 0 && by == 53) 
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
            while (allocated == 0 && i < APP_SHMEM_TABLE_ENTRIES_PER_BANK){
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
                    if (b == 0 && by == 53) 
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
                if ((shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].hash[0]) <= HASH_ERROR && (shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].hash[0] - shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].hash[0]) >= (-1*HASH_ERROR)){  
                    if (b == 0 && by == 53) 
                        printf("Hash matched in the conservative check so some magic will happen\n");
                    for (int k = 0; k < sizeQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue); k++)
                        pushQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+i].vaddrQueue, getValueQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue, k));
                    shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].valid = 0;
                    clearQueue(&shmemTable[by][b*SHMEM_TABLE_NUM_ENTRIES+j].vaddrQueue);
                }
            }
        }
    }
    if (b == 0 && by == 53)  {
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
                               struct shmemTableEntry** shmemTable){
	
    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    int APPROX = 1;

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

    int bank = (ty*16+tx)%32;
    int offset_temp_on_cuda = ty/2;
    int offset_power_on_cuda = ty/2+8;
    int offset_temp_t = ty/2+16;
    int blk = by*8+bx;

    if(ty < 2){
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

    
    if ( ty < 2) {
        approxShmem(temp_on_cuda, approx_shmem, shmemTable, bx, blk, tx, ty, bank, &temp_on_cuda[ty][tx]); 
    }

    __syncthreads();
    if (bank == 0 && blk == 53 && tx == 0 && ty == 0) {
        printf("temp_on_cuda[%d][%d] = %f\n", ty, tx, temp_on_cuda[ty][tx]);
        printShmemTable(shmemTable, bank, blk, tx, ty);
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

    bool computed;
    if (APPROX) {
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty) + step_div_Cap * (getApproxShmem(offset_power_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty) + (getApproxShmem(int(S/2), (S*16+tx)%32, approx_shmem, shmemTable, blk, tx, ty) + getApproxShmem((int)N/2, (N*16+tx)%32, approx_shmem, shmemTable, blk, tx, ty) - 2.0*getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty)) * Ry_1 + (getApproxShmem(offset_temp_on_cuda, (ty*16+E)%32, approx_shmem, shmemTable, blk, tx, ty) + getApproxShmem(offset_temp_on_cuda, (ty*16+W)%32, approx_shmem, shmemTable, blk, tx, ty) - 2.0*getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty)) * Rx_1 + (amb_temp - getApproxShmem(offset_temp_on_cuda, bank, approx_shmem, shmemTable, blk, tx, ty)) * Rz_1);
                 approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+offset_temp_t] = temp_t[ty][tx];
        
            }
            __syncthreads();

            if ( ty < 2) {
                approxShmem(temp_on_cuda, approx_shmem, shmemTable, bx, blk, tx, ty, bank, &temp_on_cuda[ty][tx]); 
            }
            __syncthreads();

            if(i==iteration-1)
                break;
            if(computed){	 //Assign the computation range
                temp_on_cuda[ty][tx]= getApproxShmem(offset_temp_t, bank, approx_shmem, shmemTable, blk, tx, ty);
                approx_shmem[blk][bank*SHMEM_ELEMENTS_PER_BANK+offset_temp_on_cuda] = temp_on_cuda[ty][tx];
            }
            __syncthreads();

            if ( ty < 2) {
                approxShmem(temp_on_cuda, approx_shmem, shmemTable, bx, blk, tx, ty, bank, &temp_on_cuda[ty][tx]); 
            }
            __syncthreads();
          }

          // update the global memory
          // after the last iteration, only threads coordinated within the 
          // small block perform the calculation and switch on ``computed''
          if (computed){
              temp[index]= getApproxShmem(offset_temp_t, bank, approx_shmem, shmemTable, blk, tx, ty);		
          }
    }
    else {
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
                     (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
                     (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
                     (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
        
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
		col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed, approx_shmem, shmemTable);
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
    float **h_approx_shmem = (float**)malloc(grid_rows*grid_cols*sizeof(float*));
    struct shmemTableEntry **h_shmemTable = (struct shmemTableEntry**)malloc(grid_rows*grid_cols*sizeof(struct shmemTableEntry*));
    printf("Malloc 1, SHMEM_ELEMENTS_PER_BANK = %d, SHMEM_TABLE_NUM_ENTRIES = %d\n", SHMEM_ELEMENTS_PER_BANK, SHMEM_TABLE_NUM_ENTRIES);
    for (int blks = 0; blks < grid_rows*grid_cols; blks++) {
        cudaMalloc((void**)&h_approx_shmem[blks], (SHMEM_NUM_BANKS*SHMEM_ELEMENTS_PER_BANK) * sizeof(float));
        cudaMalloc((void**)&h_shmemTable[blks], (SHMEM_NUM_BANKS*SHMEM_TABLE_NUM_ENTRIES) * sizeof(struct shmemTableEntry));
    }
    printf("Malloc 2\n");

    cudaMalloc((void***)&approx_shmem, grid_rows*grid_cols*sizeof(float*));
    cudaMalloc((void***)&shmemTable, grid_rows*grid_cols*sizeof(struct shmemTableEntry*));
    cudaMemcpy(approx_shmem, h_approx_shmem, grid_rows*grid_cols*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shmemTable, h_shmemTable, grid_rows*grid_cols*sizeof(struct shmemTableEntry*), cudaMemcpyHostToDevice);
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
