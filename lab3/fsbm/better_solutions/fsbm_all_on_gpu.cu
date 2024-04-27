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

__global__ void getSearchArea_GPU(int *searchArea, int *frame, int i, int j, Parameters p, int step)
{
    unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    int m = row - p.searchRange;
    int n = column - p.searchRange;
    if (((0 <= (i + m)) && ((i + m) < p.height)) && ((0 <= (j + n)) && ((j + n) < p.width)))
    {
        searchArea[row * step + column] = frame[(i + row - p.searchRange) * p.width + j + column - p.searchRange];
    }
    else
    {
        searchArea[row * step + column] = 0;
    }
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

__global__ void SAD_GPU(int *d_CurrentBlock, int *d_SearchArea, int rowIdx, int colIdx, Parameters p, int *d_results)
{
    {
        int sad = 0;
        int step_search = 2 * SEARCH_RANGE + BLOCK_SIZE;
        int posX = blockIdx.y * blockDim.y + threadIdx.y;
        int posY = blockIdx.x * blockDim.x + threadIdx.x;
        if ((0 <= (rowIdx + posX)) && ((rowIdx + posX) < p.height) &&
            (0 <= (colIdx + posY)) && ((colIdx + posY) < p.width))
        {
            // computes SAD disparity, by comparing the current block with the reference block at (k,m)
            for (int i = 0; i < BLOCK_SIZE; i++)
            {
                for (int j = 0; j < BLOCK_SIZE; j++)
                {
                    sad += abs(d_CurrentBlock[i * BLOCK_SIZE + j] - d_SearchArea[(posX + i) * step_search + (posY + j)]);
                }
            }
            // compares the obtained sad with the best so far for that block
            d_results[posY * 2 * SEARCH_RANGE + posX] = sad;
        }
	else
	    d_results[posY * 2 * SEARCH_RANGE + posX] = BigSAD;
    }
}

void fullSearch_GPU(BestResult *bestResult, int *d_CurrentBlock, int *d_SearchArea, int rowIdx, int colIdx, Parameters p, int *d_results, int *results)
{
    bestResult->sad = BigSAD;
    bestResult->bestDist = 0;
    bestResult->vec_x = 0;
    bestResult->vec_y = 0;
    dim3 gridSize(1, 2 * SEARCH_RANGE, 1);
    dim3 blockSize(2 * SEARCH_RANGE, 1, 1);
    SAD_GPU<<<gridSize, blockSize>>>(d_CurrentBlock, d_SearchArea, rowIdx, colIdx, p, d_results);
    if (cudaMemcpy(results, d_results, 4 * SEARCH_RANGE * SEARCH_RANGE * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("FAILED TO COPY results DATA TO THE host: %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(0);
    }
    for (int i = 0; i < 2 * SEARCH_RANGE; i++)
    {
        for (int j = 0; j < 2 * SEARCH_RANGE; j++)
        {
            if (results[i * 2 * SEARCH_RANGE + j] < bestResult->sad)
            {
                bestResult->sad = results[i * 2 * SEARCH_RANGE + j];
                bestResult->vec_x = j - SEARCH_RANGE;
                bestResult->vec_y = i - SEARCH_RANGE;
            }
        }
    }
}
/************************************************************************************/
void MotionEstimation(BestResult **motionVectors, int *curr_frame, int *d_curr_frame, int *ref_frame, int *d_ref_frame, Parameters p)
{
    BestResult *bestResult;
    int SizeInBytes_curr = p.blockSize * p.blockSize * sizeof(int);
    int SizeInBytes_search = (2 * p.searchRange + p.blockSize) * (2 * p.searchRange + p.blockSize) * sizeof(int);
    int *d_CurrentBlock, *d_results, *d_SearchArea;
    // int *CurrentBlock = (int *)malloc(SizeInBytes_curr);
    // int *SearchArea = (int *)malloc(SizeInBytes_search);
    int *results = (int *)malloc(4 * SEARCH_RANGE * SEARCH_RANGE * sizeof(int));
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

    for (int rowIdx = 0; rowIdx < (p.height - p.blockSize + 1); rowIdx += p.blockSize)
    {
        for (int colIdx = 0; colIdx < (p.width - p.blockSize + 1); colIdx += p.blockSize)
        {
            // Gets current block and search area data
            dim3 gridBlock(1, 1, 1);                      // X -> p.blockSize (threadIdx.x)
            dim3 blockBlock(p.blockSize, p.blockSize, 1); // Y    -> p.blockSize (blockIdx.y)
            dim3 gridSearch(2, 16, 1);
            dim3 blockSearch(80, 10, 1);
            // if (rowIdx >= p.searchRange && colIdx >= p.searchRange && rowIdx < p.height - border && colIdx < p.width - border && true)
            // {
            getBlock_GPU<<<gridBlock, blockBlock>>>(d_CurrentBlock, d_curr_frame, rowIdx, colIdx, p);
            getSearchArea_GPU<<<gridSearch, blockSearch>>>(d_SearchArea, d_ref_frame, rowIdx, colIdx, p, (2 * p.searchRange + p.blockSize));
            bestResult = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);
            // Runs the motion estimation algorithm on this block
            switch (p.algorithm)
            {
            case FSBM:
                fullSearch_GPU(bestResult, d_CurrentBlock, d_SearchArea, rowIdx, colIdx, p, d_results, results);
                break;
            default:
                break;
            }
            // }
            // else
            // {
            //     getBlock_CPU(CurrentBlock, curr_frame, rowIdx, colIdx, p);
            //     getSearchArea_CPU(SearchArea, ref_frame, rowIdx, colIdx, p);
            //     bestResult = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);
            //     // Runs the motion estimation algorithm on this block
            //     switch (p.algorithm)
            //     {
            //     case FSBM:
            //         fullSearch_CPU(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
            //         break;
            //     default:
            //         break;
            //     }
            // }
        }
    }
    cudaFree(d_CurrentBlock);
    cudaFree(d_results);
    free(results);
    // free(SearchArea);
    // free(CurrentBlock);
    cudaFree(d_SearchArea);
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
    int *curr_frame = (int *)malloc(p.width * p.height * sizeof(int *));
    int *ref_frame = (int *)malloc(p.width * p.height * sizeof(int *));
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

    int *d_curr_frame, *d_ref_frame;
    int SizeInBytes_frame = p.width * p.height * sizeof(int);
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
        MotionEstimation(motionVectors, curr_frame, d_curr_frame, ref_frame, d_ref_frame, p);
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
    free(curr_frame);
    free(ref_frame);
    free(res_frame);
    free(rec_frame);
    cudaFree(d_curr_frame);
    return 0;
}
