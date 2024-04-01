/************************************************************************/
/* File: interPrediction.c                                              */
/* Author: Nuno Roma <Nuno.Roma@tecnico.ulisboa.pt                      */
/* Date: February 23th, 2024                                            */
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FSBM 0 // Full-Search Block Matching (FSBM) motion estimation algorithm
#define SS 1   // (Three/Four) Step-Search (SS) block Matching motion estimation algorithm
#define TZS 2  // Test Zonal Search (TZS) block matching motion estimation algorithm

#define SEARCH_RANGE 64 // Search range (at each direction)
#define BLOCK_SIZE 32   // Block size (at each direction)
#define iRASTER 5       // TZS iRaster parameter

#define BigSAD 999999 // it could be any other big integer...

typedef struct
{
    char *video_name; // YUV input file
    int width;        // luminance width
    int height;       // luminance height
    int frames;       // number of frames to process
    int algorithm;    // motion estimation algorithm
    int searchRange;  // search range (at each direction)
    int blockSize;    // block size (at each direction)
    int iRaster;      // TZS iRaster parameter
    int debug;        // verbose mode
} Parameters;

typedef struct
{
    int vec_x;
    int vec_y;
    int sad;
    int bestDist;
} BestResult;

/************************************************************************************/
void getLumaFrame(int *frame_mem, FILE *yuv_file, Parameters p)
{
    int count;
    for (int r = 0; r < p.height; r++)
        for (int c = 0; c < p.width; c++)
            count = fread(&(frame_mem[r * p.width + c]), 1, 1, yuv_file);
    count++; // avoid warning

    // Skips the color Cb and Cr components in the YUV 4:2:0 file
    fseek(yuv_file, p.width * p.height / 2, SEEK_CUR);
}
/************************************************************************************/
void setLumaFrame(int **frame_mem, FILE *yuv_file, Parameters p)
{
    __uint8_t temp;
    for (int r = 0; r < p.height; r++)
        for (int c = 0; c < p.width; c++)
        {
            temp = (__uint8_t)frame_mem[r][c];
            fwrite(&temp, 1, 1, yuv_file);
        }
    // writes 2*(height/2)*(width/2) values to fill in chrominance part with 128
    temp = (__uint8_t)128;
    for (int r = 0; r < p.height / 2; r++)
        for (int c = 0; c < p.width; c++)
        {
            fwrite(&temp, 1, 1, yuv_file);
        }
}
/************************************************************************************/
void reconstruct(int **rec_frame, int *ref_frame, int i, int j, Parameters p, BestResult *MV)
{
    for (int a = i; a < i + p.blockSize; a++)
        for (int b = j; b < j + p.blockSize; b++)
            if ((0 <= a + MV->vec_x) && (a + MV->vec_x < p.height) && (0 <= b + MV->vec_y) && (b + MV->vec_y < p.width))
                rec_frame[a][b] = ref_frame[(a + MV->vec_x) * (p.width) + b + MV->vec_y];
}
/************************************************************************************/
unsigned long long computeResidue(int **res_frame, int *curr_frame, int **rec_frame, Parameters p)
{
    unsigned long long accumulatedDifference = 0;
    int difference;
    for (int a = 0; a < p.height; a++)
        for (int b = 0; b < p.width; b++)
        {
            difference = curr_frame[a * p.width + b] - rec_frame[a][b];
            if (difference < 0)
                difference = -difference;
            if (255 < difference)
                difference = 255;
            res_frame[a][b] = difference;
            accumulatedDifference += difference;
        }
    return (accumulatedDifference);
}
/************************************************************************************/
__global__ void getBlock_GPU(int *block, int *frame, int i, int j, Parameters p)
{
    unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    block[row * p.blockSize + column] = frame[(i + row) * p.width + j + column];
}

void getBlock_CPU(int *block, int *frame, int i, int j, Parameters p)
{
    for (int m = 0; m < p.blockSize; m++)
    {
        for (int n = 0; n < p.blockSize; n++)
        {
            block[m * p.blockSize + n] = frame[(i + m) * p.width + j + n];
        }
    }
}

/************************************************************************************/
void getSearchArea_CPU(int *searchArea, int *frame, int i, int j, Parameters p)
{
    int step = 2 * p.searchRange + p.blockSize;
    for (int m = -p.searchRange; m < p.searchRange + p.blockSize; m++)
        for (int n = -p.searchRange; n < p.searchRange + p.blockSize; n++)
            if (((0 <= (i + m)) && ((i + m) < p.height)) && ((0 <= (j + n)) && ((j + n) < p.width)))
            {
                searchArea[(p.searchRange + m) * step + (p.searchRange + n)] = frame[(i + m) * p.width + j + n];
            }
            else
            {
                searchArea[(p.searchRange + m) * step + (p.searchRange + n)] = 0;
            }
}

/************************************************************************************/
__global__ void getSearchArea_GPU(int *searchArea, int *frame, int i, int j, Parameters p, int step)
{
    unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    searchArea[row * step + column] = frame[(i + row - p.searchRange) * p.width + j + column - p.searchRange];
}

/************************************************************************************/
void SAD_CPU(BestResult *bestResult, int *CurrentBlock, int *SearchArea, int rowIdx, int colIdx, int k, int m, Parameters p)
{
    // k, m: displacement (motion vector) under analysis (in the search area)

    int posX = p.searchRange + k; // normalized coordinates within search area, between 0 and 2*searchRange
    int posY = p.searchRange + m; // normalized coordinates within search area, between 0 and 2*searchRange
    // checks if search area range is valid (inside frame borders) and if current block range is valid (inside frame borders)
    if ((0 <= (rowIdx + posX)) && ((rowIdx + posX) < p.height) &&
        (0 <= (colIdx + posY)) && ((colIdx + posY) < p.width))
    {
        int sad = 0;
        int step_search = 2 * p.searchRange + p.blockSize;
        // computes SAD disparity, by comparing the current block with the reference block at (k,m)
        for (int i = 0; i < p.blockSize; i++)
        {
            for (int j = 0; j < p.blockSize; j++)
            {
                sad += abs(CurrentBlock[i * p.blockSize + j] - SearchArea[(posX + i) * step_search + (posY + j)]);
            }
        }
        // compares the obtained sad with the best so far for that block
        if (sad < bestResult->sad)
        {
            bestResult->sad = sad;
            bestResult->vec_x = k;
            bestResult->vec_y = m;
        }
    }
}

/************************************************************************************/
void fullSearch_CPU(BestResult *bestResult, int *CurrentBlock, int *SearchArea, int rowIdx, int colIdx, Parameters p)
{
    bestResult->sad = BigSAD;
    bestResult->bestDist = 0;
    bestResult->vec_x = 0;
    bestResult->vec_y = 0;
    for (int iStartX = -p.searchRange; iStartX < p.searchRange; iStartX++)
    {
        for (int iStartY = -p.searchRange; iStartY < p.searchRange; iStartY++)
        {
            SAD_CPU(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, iStartX, iStartY, p);
        }
    }
}

__device__ void warpReduce(volatile int *shared_data, int tid)
{
    shared_data[tid] += shared_data[tid + 32];
    shared_data[tid] += shared_data[tid + 16];
    shared_data[tid] += shared_data[tid + 8];
    shared_data[tid] += shared_data[tid + 4];
    shared_data[tid] += shared_data[tid + 2];
    shared_data[tid] += shared_data[tid + 1];
}

/************************************************************************************/
__global__ void SAD_GPU(int *d_CurrentBlock, int *d_SearchArea, int rowIdx, int colIdx, Parameters p, int *d_results)
{
    {
        int step_search = 2 * SEARCH_RANGE + BLOCK_SIZE;
        int posX = blockIdx.y * blockDim.y + threadIdx.y;
        int tid = (blockIdx.x * blockDim.x + threadIdx.x);
        int i = tid * 4;
        int posY = blockIdx.z * blockDim.z + threadIdx.z;
        int f = i >> 5;
        int g = i & 31;
        __shared__ int sad_shared[256];

        // computes SAD disparity, by comparing the current block with the reference block at (k,m)
        sad_shared[tid] = abs(d_CurrentBlock[i] - d_SearchArea[(posX + f) * step_search + (posY + g)]) + abs(d_CurrentBlock[i + 1] - d_SearchArea[(posX + f) * step_search + (posY + g + 1)]) + abs(d_CurrentBlock[i + 2] - d_SearchArea[(posX + f) * step_search + (posY + g + 2)]) + abs(d_CurrentBlock[i + 3] - d_SearchArea[(posX + f) * step_search + (posY + g + 3)]);
        __syncthreads();

        // if (tid < 256)
        //     sad_shared[tid] += sad_shared[tid + 256];
        // __syncthreads();

        if (tid < 128)
            sad_shared[tid] += sad_shared[tid + 128];
        __syncthreads();

        if (tid < 64)
            sad_shared[tid] += sad_shared[tid + 64];
        __syncthreads();

        if (tid < 32)
        {
            warpReduce(sad_shared, tid);
        }
        // compares the obtained sad with the best so far for that block
        if (tid == 0)
        {
            d_results[posY * 2 * SEARCH_RANGE + posX] = sad_shared[0];
        }
    }
}

__global__ void Best_GPU(int *d_results, int step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ int shared[128];
    __shared__ int values_of_x[128];
    shared[x] = d_results[y * step + x];
    values_of_x[x] = x;
    __syncthreads();
    if (x < 64)
    {
        if (shared[x] < shared[x + 64])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
        }
        else
        {
            shared[x] = shared[x + 64];
            values_of_x[x] = values_of_x[x + 64];
        }
    }
    __syncthreads();
    if (x < 32)
    {
        if (shared[x] < shared[x + 32])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
        }
        else
        {
            shared[x] = shared[x + 32];
            values_of_x[x] = values_of_x[x + 32];
        }
    }
    __syncthreads();
    if (x < 16)
    {
        if (shared[x] < shared[x + 16])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
        }
        else
        {
            shared[x] = shared[x + 16];
            values_of_x[x] = values_of_x[x + 16];
        }
    }
    __syncthreads();
    if (x < 8)
    {
        if (shared[x] < shared[x + 8])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
        }
        else
        {
            shared[x] = shared[x + 8];
            values_of_x[x] = values_of_x[x + 8];
        }
    }
    __syncthreads();
    if (x < 4)
    {
        if (shared[x] < shared[x + 4])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
        }
        else
        {
            shared[x] = shared[x + 4];
            values_of_x[x] = values_of_x[x + 4];
        }
    }
    __syncthreads();
    if (x < 2)
    {
        if (shared[x] < shared[x + 2])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
        }
        else
        {
            shared[x] = shared[x + 2];
            values_of_x[x] = values_of_x[x + 2];
        }
    }
    __syncthreads();
    if (x < 1)
    {
        if (shared[x] < shared[x + 1])
        {
            d_results[y * 2] = shared[x];
            d_results[y * 2 + 1] = values_of_x[x] - SEARCH_RANGE;
        }
        else
        {
            d_results[y * 2] = shared[x + 1];
            d_results[y * 2 + 1] = values_of_x[x + 1] - SEARCH_RANGE;
        }
    }
}

__global__ void Best_GPU_final(int *d_results)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared[128];
    __shared__ int values_of_x[128];
    __shared__ int values_of_y[128];
    shared[x] = d_results[2 * x];
    values_of_x[x] = d_results[2 * x + 1];
    values_of_y[x] = x;
    __syncthreads();
    if (x < 64)
    {
        if (shared[x] < shared[x + 64])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
            values_of_y[x] = values_of_y[x];
        }
        else
        {
            shared[x] = shared[x + 64];
            values_of_x[x] = values_of_x[x + 64];
            values_of_y[x] = values_of_y[x + 64];
        }
    }
    __syncthreads();
    if (x < 32)
    {
        if (shared[x] < shared[x + 32])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
            values_of_y[x] = values_of_y[x];
        }
        else
        {
            shared[x] = shared[x + 32];
            values_of_x[x] = values_of_x[x + 32];
            values_of_y[x] = values_of_y[x + 32];
        }
    }
    __syncthreads();
    if (x < 16)
    {
        if (shared[x] < shared[x + 16])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
            values_of_y[x] = values_of_y[x];
        }
        else
        {
            shared[x] = shared[x + 16];
            values_of_x[x] = values_of_x[x + 16];
            values_of_y[x] = values_of_y[x + 16];
        }
    }
    __syncthreads();
    if (x < 8)
    {
        if (shared[x] < shared[x + 8])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
            values_of_y[x] = values_of_y[x];
        }
        else
        {
            shared[x] = shared[x + 8];
            values_of_x[x] = values_of_x[x + 8];
            values_of_y[x] = values_of_y[x + 8];
        }
    }
    __syncthreads();
    if (x < 4)
    {
        if (shared[x] < shared[x + 4])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
            values_of_y[x] = values_of_y[x];
        }
        else
        {
            shared[x] = shared[x + 4];
            values_of_x[x] = values_of_x[x + 4];
            values_of_y[x] = values_of_y[x + 4];
        }
    }
    __syncthreads();
    if (x < 2)
    {
        if (shared[x] < shared[x + 2])
        {
            shared[x] = shared[x];
            values_of_x[x] = values_of_x[x];
            values_of_y[x] = values_of_y[x];
        }
        else
        {
            shared[x] = shared[x + 2];
            values_of_x[x] = values_of_x[x + 2];
            values_of_y[x] = values_of_y[x + 2];
        }
    }
    __syncthreads();
    if (x < 1)
    {
        if (shared[x] < shared[x + 1])
        {
            d_results[0] = values_of_x[x];
            d_results[1] = values_of_y[x] - SEARCH_RANGE;
            d_results[2] = shared[x];
        }
        else
        {
            d_results[0] = values_of_x[x + 1];
            d_results[1] = values_of_y[x + 1] - SEARCH_RANGE;
            d_results[2] = shared[x + 1];
        }
    }
}

// 4216103
// 31547568
/************************************************************************************/
void fullSearch_GPU(BestResult *bestResult, int *d_CurrentBlock, int *d_SearchArea, int rowIdx, int colIdx, Parameters p, int *d_results)
{
    int step = 2 * SEARCH_RANGE;
    dim3 gridSize(1, 2 * SEARCH_RANGE, 2 * SEARCH_RANGE);
    dim3 blockSize(BLOCK_SIZE * 8, 1, 1);
    dim3 grid_Best(1, 128, 1);
    dim3 block_Best(128, 1, 1);
    dim3 grid_Best_final(1, 1, 1);
    SAD_GPU<<<gridSize, blockSize>>>(d_CurrentBlock, d_SearchArea, rowIdx, colIdx, p, d_results);
    Best_GPU<<<grid_Best, block_Best>>>(d_results, step);
    Best_GPU_final<<<grid_Best_final, block_Best>>>(d_results);
    if (cudaMemcpy(bestResult, d_results, 3 * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("FAILED TO COPY bestResults(d_results) DATA TO THE host: %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(0);
    }
}
/************************************************************************************/
void MotionEstimation(BestResult **motionVectors, int *curr_frame, int *d_curr_frame, int *ref_frame, int *d_ref_frame, Parameters p,
                      int *d_results, int *d_CurrentBlock, int *d_SearchArea, int *CurrentBlock, int *SearchArea)
{
    BestResult *bestResult;
    int border = 2 * p.searchRange;

    for (int rowIdx = 0; rowIdx < (p.height - p.blockSize + 1); rowIdx += p.blockSize)
    {
        for (int colIdx = 0; colIdx < (p.width - p.blockSize + 1); colIdx += p.blockSize)
        {
            // Gets current block and search area data
            dim3 gridBlock(1, 1, 1);                      // X -> p.blockSize (threadIdx.x)
            dim3 blockBlock(p.blockSize, p.blockSize, 1); // Y    -> p.blockSize (blockIdx.y)
            dim3 gridSearch(5, 5, 1);
            dim3 blockSearch(32, 32, 1);
            if (rowIdx >= p.searchRange && colIdx >= p.searchRange && rowIdx < p.height - border && colIdx < p.width - border)
            {
                getBlock_GPU<<<gridBlock, blockBlock>>>(d_CurrentBlock, d_curr_frame, rowIdx, colIdx, p);
                getSearchArea_GPU<<<gridSearch, blockSearch>>>(d_SearchArea, d_ref_frame, rowIdx, colIdx, p, (2 * p.searchRange + p.blockSize));
                bestResult = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);
                fullSearch_GPU(bestResult, d_CurrentBlock, d_SearchArea, rowIdx, colIdx, p, d_results);
            }
            else
            {
                getBlock_CPU(CurrentBlock, curr_frame, rowIdx, colIdx, p);
                getSearchArea_CPU(SearchArea, ref_frame, rowIdx, colIdx, p);
                bestResult = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);
                fullSearch_CPU(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
            }
        }
    }
}

/************************************************************************************/
int main(int argc, char **argv)
{

    struct timespec t0, t1;
    unsigned long long accumulatedResidue = 0;

    // Read input parameters
    if (argc != 7)
    {
        printf("USAGE: %s <videoPath> <Width> <Height> <NFrames> <ME Algorithm: 0=FSBM; 1=SS; 2=TZS> <Debug Mode: 0=silent; 1=verbose>\n", argv[0]);
        exit(1);
    }
    Parameters p;
    p.video_name = argv[1];
    p.width = atoi(argv[2]);
    p.height = atoi(argv[3]);
    p.frames = atoi(argv[4]);
    p.algorithm = atoi(argv[5]);
    p.searchRange = SEARCH_RANGE; // Search range (at each direction)
    p.blockSize = BLOCK_SIZE;     // Block size (at each direction)
    p.iRaster = iRASTER;          // TZS iRaster parameter
    p.debug = atoi(argv[6]);

    switch (p.algorithm)
    {
    case FSBM:
        printf("Running FSBM algorithm\n");
        break;
    case SS:
        printf("Running Step-Search algorithm\n");
        break;
    case TZS:
        printf("Running TZS algorithm\n");
        break;
    default:
        printf("ERROR: Invalid motion estimation algorithm\n");
        exit(-1);
    }

    // Video files
    FILE *video_in;
    FILE *residue_out;
    FILE *reconst_out;
    video_in = fopen(p.video_name, "rb");
    residue_out = fopen("residue.yuv", "wb");
    reconst_out = fopen("reconst.yuv", "wb");
    if (!video_in || !residue_out || !reconst_out)
    {
        printf("Opening input/output file error\n");
        exit(1);
    }

    // Frame memory allocation
    int SizeInBytes_frame = p.width * p.height * sizeof(int);
    int *ref_frame, *curr_frame;
    if (cudaMallocHost((int **)&curr_frame, SizeInBytes_frame) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE curr_frame");
        exit(0);
    }
    if (cudaMallocHost((int **)&ref_frame, SizeInBytes_frame) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE ref_frame");
        exit(0);
    }
    int **res_frame = (int **)malloc(p.height * sizeof(int *));
    int **rec_frame = (int **)malloc(p.height * sizeof(int *));
    for (int i = 0; i < p.height; i++)
    {
        res_frame[i] = (int *)malloc(p.width * sizeof(int));
        rec_frame[i] = (int *)malloc(p.width * sizeof(int));
    }

    // Memory allocation of result table
    BestResult **motionVectors = (BestResult **)malloc(p.height / p.blockSize * sizeof(BestResult *));
    for (int i = 0; i < p.height / p.blockSize; i++)
        motionVectors[i] = (BestResult *)malloc(p.width / p.blockSize * sizeof(BestResult));
    BestResult *MV;

    clock_gettime(CLOCK_REALTIME, &t0);
    // Read first frame
    getLumaFrame(curr_frame, video_in, p); // curr_frame contains the current luminance frame
    //

    int *d_curr_frame, *d_ref_frame, *d_results, *d_CurrentBlock, *d_SearchArea, *CurrentBlock, *SearchArea;
    int SizeInBytes_curr = p.blockSize * p.blockSize * sizeof(int);
    int SizeInBytes_search = (2 * p.searchRange + p.blockSize) * (2 * p.searchRange + p.blockSize) * sizeof(int);

    if (cudaMallocHost((int **)&CurrentBlock, SizeInBytes_curr) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE CurrentBlock");
        exit(0);
    }
    if (cudaMallocHost((int **)&SearchArea, SizeInBytes_search) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE SearchArea");
        exit(0);
    }

    if (cudaMalloc((void **)&d_curr_frame, SizeInBytes_frame) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_curr_frame");
        exit(0);
    }

    if (cudaMalloc((void **)&d_ref_frame, SizeInBytes_frame) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_ref_frame");
        exit(0);
    }

    if (cudaMemcpy(d_ref_frame, curr_frame, SizeInBytes_frame, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("FAILED TO COPY ReferenceFrame DATA TO THE DEVICE\n");
        exit(0);
    }
    if (cudaMalloc((void **)&d_CurrentBlock, SizeInBytes_curr) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_CurrentBlock");
        exit(0);
    }
    if (cudaMalloc((void **)&d_results, 4 * SEARCH_RANGE * SEARCH_RANGE * sizeof(int)) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_results");
        exit(0);
    }
    if (cudaMalloc((void **)&d_SearchArea, SizeInBytes_search) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_SearchArea");
        exit(0);
    }

    for (int frameNum = 0; frameNum < p.frames; frameNum++)
    {
        if (frameNum != 0)
        {
            int *d_temp;
            d_temp = d_ref_frame;
            d_ref_frame = d_curr_frame; // ref_frame contains the previous (reference) luminance frame
            d_curr_frame = d_temp;
        }
        int *temp;
        temp = ref_frame;
        ref_frame = curr_frame; // ref_frame contains the previous (reference) luminance frame
        curr_frame = temp;
        // MUDAR ISTO DEPOIS PARA PASSAR POR REFERENCIA
        getLumaFrame(curr_frame, video_in, p); // curr_frame contains the current luminance frame

        if (cudaMemcpy(d_curr_frame, curr_frame, SizeInBytes_frame, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            printf("FAILED TO COPY Currentframe DATA TO THE DEVICE\n");
            exit(0);
        }

        // Process the current frame, one block at a time, to obatin an array with the motion vectors and SAD values
        MotionEstimation(motionVectors, curr_frame, d_curr_frame, ref_frame, d_ref_frame, p, d_results, d_CurrentBlock, d_SearchArea, CurrentBlock, SearchArea);
        // Recustruct the predicted frame using the obtained motion vectors
        for (int rowIdx = 0; rowIdx < p.height - p.blockSize + 1; rowIdx += p.blockSize)
        {
            for (int colIdx = 0; colIdx < p.width - p.blockSize + 1; colIdx += p.blockSize)
            {
                // Gets best candidate block information
                MV = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);

                // Reconstructs current block using  the obtained motion estimation information
                reconstruct(rec_frame, ref_frame, rowIdx, colIdx, p, MV);

                // Print vector information
                if (p.debug)
                    printf("Frame %d : Block [%4d , %4d] = (%3d,%3d), SAD= %d\n", frameNum, colIdx, rowIdx, MV->vec_y, MV->vec_x, MV->sad);
            }
        }
        // Reconstructs borders of the frame not convered by motion estimation
        for (int r = 0; r < p.height; r++)
            for (int c = 0; c < p.width; c++)
                if (r > (p.height - p.blockSize + 1) || c > (p.width - p.blockSize + 1))
                    rec_frame[r][c] = ref_frame[r * p.width + c];

        // Compute residue block
        accumulatedResidue += computeResidue(res_frame, curr_frame, rec_frame, p);

        // Save reconstructed and residue frames
        setLumaFrame(rec_frame, reconst_out, p);
        setLumaFrame(res_frame, residue_out, p);
    }
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("%lf seconds elapsed \n", (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9);
    printf("Accumulated Residue = %llu \n", accumulatedResidue);

    // Frame memory free
    for (int i = 0; i < p.height; i++)
    {
        free(res_frame[i]);
        free(rec_frame[i]);
    }
    cudaFreeHost(curr_frame);
    cudaFreeHost(ref_frame);
    cudaFreeHost(CurrentBlock);
    cudaFreeHost(SearchArea);
    free(res_frame);
    free(rec_frame);
    cudaFree(d_curr_frame);
    cudaFree(d_CurrentBlock);
    cudaFree(d_results);
    cudaFree(d_SearchArea);
    return 0;
}