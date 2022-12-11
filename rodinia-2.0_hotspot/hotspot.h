#define SHMEM_NUM_BANKS 32
#define SHMEMSIZE 49152
#define SHMEM_CHUNK_SIZE 4

#define SHMEM_ELEMENTS SHMEMSIZE/4
#define SHMEM_ELEMENTS_PER_BANK SHMEM_ELEMENTS/SHMEM_NUM_BANKS
#define SHMEM_TABLE_NUM_ENTRIES SHMEM_ELEMENTS_PER_BANK/SHMEM_CHUNK_SIZE
#define APP_TOTAL_SHMEM_ELEMENTS 256*3
#define APP_SHMEM_ELEMENTS_PER_BANK APP_TOTAL_SHMEM_ELEMENTS/SHMEM_NUM_BANKS
#define APP_SHMEM_TABLE_ENTRIES_PER_BANK APP_SHMEM_ELEMENTS_PER_BANK/SHMEM_CHUNK_SIZE

struct shmemTableEntry {
    int valid;
    struct queueNode *vaddrQueue;
    int paddr;
    float hash[3]; //0 = min, 1 = max, 2 = avg
    int status; //0000 = empty, 0001 = only first, 0010 = only second, 0100 = only third, 1000 only 4th, 1111 = all
};

