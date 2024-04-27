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
void getBlock(int *block, int *frame, int i, int j, Parameters p)
{
    for (int m = 0; m < p.blockSize; m++)
        for (int n = 0; n < p.blockSize; n++)
            block[m * p.blockSize + n] = frame[(i + m) * p.width + j + n];
}
/************************************************************************************/
void getSearchArea(int *searchArea, int *frame, int i, int j, Parameters p)
{
    for (int m = -p.searchRange; m < p.searchRange + p.blockSize; m++)
        for (int n = -p.searchRange; n < p.searchRange + p.blockSize; n++)
            if (((0 <= (i + m)) && ((i + m) < p.height)) && ((0 <= (j + n)) && ((j + n) < p.width)))
                searchArea[(p.searchRange + m) * (2 * p.searchRange + p.blockSize) + (p.searchRange + n)] = frame[(i + m) * p.width + j + n];
            else
                searchArea[(p.searchRange + m) * (2 * p.searchRange + p.blockSize) + (p.searchRange + n)] = 0;
}
/************************************************************************************/

__global__ void sum_SAD(int *curr, int *search, int *d_results, int posX, int posY, int step_search)
{
    unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    unsigned int column_search = column + posY;
    unsigned int row_search = row + posX;
    unsigned int tid = threadIdx.x;
    __shared__ int sad[BLOCK_SIZE];
    sad[tid] = abs(curr[row * blockDim.x + column] - search[row_search * step_search + column_search]);
    __syncthreads();
    for (int i = blockDim.x >> 1; i > 0; i = i >> 1)
    {
        if (tid < i)
        {
            sad[tid] += sad[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_results[blockIdx.y] = sad[0];
    }
}

__global__ void sum_results(int *d_results)
{
    unsigned int tid = threadIdx.x;
    __shared__ int sad[BLOCK_SIZE];
    sad[tid] = d_results[tid];
    __syncthreads();
    for (int i = blockDim.x >> 1; i > 0; i = i >> 1)
    {
        if (tid < i)
        {
            sad[tid] += sad[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_results[blockIdx.y] = sad[0];
    }
}
void SAD(BestResult *bestResult, int *d_CurrentBlock, int *d_SearchArea, int rowIdx, int colIdx, int k, int m, Parameters p, int *d_results)
{
    // k, m: displacement (motion vector) under analysis (in the search area)

    int sad[1] = {0};
    int posX = p.searchRange + k; // normalized coordinates within search area, between 0 and 2*searchRange
    int posY = p.searchRange + m; // normalized coordinates within search area, between 0 and 2*searchRange
    // checks if search area range is valid (inside frame borders) and if current block range is valid (inside frame borders)
    if ((-p.searchRange <= k) && (k <= p.searchRange) &&
        (-p.searchRange <= m) && (m <= p.searchRange) &&
        (0 <= (rowIdx + posX)) && ((rowIdx + posX) < p.height) &&
        (0 <= (colIdx + posY)) && ((colIdx + posY) < p.width))
    {
        // computes SAD disparity, by comparing the current block with the reference block at (k,m)
        int tamanhoy = BLOCK_SIZE;        // SE mudar alterar o tamanho do block results na função fullSearch
        dim3 gridDist(1, p.blockSize, 1); // X -> p.blockSize (threadIdx.x)
        dim3 blockDist(tamanhoy, 1, 1);   // Y    -> tamanhoy (blockIdx.y)

        sum_SAD<<<gridDist, blockDist>>>(d_CurrentBlock, d_SearchArea, d_results, posX, posY, (2 * p.searchRange + p.blockSize));
        sum_results<<<1, tamanhoy>>>(d_results);
        if (cudaMemcpy(sad, d_results, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            printf("FAILED TO COPY results DATA TO THE host\n");
            exit(0);
        }

        // for (int i = 0; i < tamanhoy; i++)
        // {
        //     sad += results[i];
        // }
        // printf("GPU -> %d\n", sad);
        // compares the obtained sad with the best so far for that block
        if (sad[0] < bestResult->sad)
        {
            bestResult->sad = sad[0];
            bestResult->vec_x = k;
            bestResult->vec_y = m;
        }
    }
}

void fullSearch(BestResult *bestResult, int *CurrentBlock, int *SearchArea, int rowIdx, int colIdx, Parameters p)
{
    bestResult->sad = BigSAD;
    bestResult->bestDist = 0;
    bestResult->vec_x = 0;
    bestResult->vec_y = 0;
    int *d_CurrentBlock, *d_SearchArea;
    int SizeInBytes_curr = p.blockSize * p.blockSize * sizeof(int);
    int SizeInBytes_search = (2 * p.searchRange + p.blockSize) * (2 * p.searchRange + p.blockSize) * sizeof(int);
    int *d_results;
    if (cudaMalloc((void **)&d_CurrentBlock, SizeInBytes_curr) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_CurrentBlock");
        exit(0);
    }
    if (cudaMalloc((void **)&d_SearchArea, SizeInBytes_search) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_CurrentBlock");
        exit(0);
    }
    if (cudaMemcpy(d_CurrentBlock, CurrentBlock, SizeInBytes_curr, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("FAILED TO COPY CurrentBlock DATA TO THE DEVICE\n");
        exit(0);
    }
    if (cudaMemcpy(d_SearchArea, SearchArea, SizeInBytes_search, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("FAILED TO COPY SearchArea DATA TO THE DEVICE\n");
        exit(0);
    }

    if (cudaMalloc((void **)&d_results, BLOCK_SIZE * sizeof(int)) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_results");
        exit(0);
    }

    for (int iStartX = -p.searchRange; iStartX < p.searchRange; iStartX++)
    {
        for (int iStartY = -p.searchRange; iStartY < p.searchRange; iStartY++)
        {
            SAD(bestResult, d_CurrentBlock, d_SearchArea, rowIdx, colIdx, iStartX, iStartY, p, d_results);
        }
    }
    cudaFree(d_CurrentBlock);
    cudaFree(d_SearchArea);
    cudaFree(d_results);
}
/************************************************************************************/
void MotionEstimation(BestResult **motionVectors, int *curr_frame, int *ref_frame, Parameters p)
{
    BestResult *bestResult;

    int *CurrentBlock = (int *)malloc(p.blockSize * p.blockSize * sizeof(int));

    int *SearchArea = (int *)malloc((2 * p.searchRange + p.blockSize) * (2 * p.searchRange + p.blockSize) * sizeof(int));

    for (int rowIdx = 0; rowIdx < (p.height - p.blockSize + 1); rowIdx += p.blockSize)
        for (int colIdx = 0; colIdx < (p.width - p.blockSize + 1); colIdx += p.blockSize)
        {
            // Gets current block and search area data
            getBlock(CurrentBlock, curr_frame, rowIdx, colIdx, p);
            getSearchArea(SearchArea, ref_frame, rowIdx, colIdx, p);
            bestResult = &(motionVectors[rowIdx / p.blockSize][colIdx / p.blockSize]);
            // Runs the motion estimation algorithm on this block
            switch (p.algorithm)
            {
            case FSBM:
                fullSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
                break;
            // case TZS:
            //     TZSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
            //     break;
            // case SS:
            //     StepSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, p);
            //     break;
            default:
                break;
            }
        }
    free(CurrentBlock);
    free(SearchArea);
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
    // int *d_curr_frame, *d_ref_frame;
    // int SizeInBytes = p.width * p.height * sizeof(int);
    // if (cudaMalloc((void **)&d_curr_frame, SizeInBytes) != cudaSuccess)
    // {
    //     printf("CANNOT ALLOCATE d_curr_frame");
    //     exit(0);
    // }
    // if (cudaMalloc((void **)&d_ref_frame, SizeInBytes) != cudaSuccess)
    // {
    //     printf("CANNOT ALLOCATE d_ref_frame");
    //     exit(0);
    // }

    for (int frameNum = 0; frameNum < p.frames; frameNum++)
    {
        int *temp;
        temp = ref_frame;
        ref_frame = curr_frame; // ref_frame contains the previous (reference) luminance frame
        curr_frame = temp;
        // cudaMemcpy()
        getLumaFrame(curr_frame, video_in, p); // curr_frame contains the current luminance frame

        // Process the current frame, one block at a time, to obatin an array with the motion vectors and SAD values

        MotionEstimation(motionVectors, curr_frame, ref_frame, p);

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
    return 0;
}

// 190 mais ou menos, por baixo do SAD
/************************************************************************************/
// void StepSearch(BestResult *bestResult, int **CurrentBlock, int **SearchArea, int rowIdx, int colIdx, Parameters p)
// {

//     bestResult->sad = BigSAD;
//     bestResult->vec_x = 0;
//     bestResult->vec_y = 0;

//     // First prediction, at the center of the search area
//     int CenterX = 0;
//     int CenterY = 0;
//     // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX, CenterY, p);

//     // Furthest search center
//     int Distance = (p.searchRange) >> 1; // Initial distance = search range/2
//     while (Distance >= 1)
//     {
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX - Distance, CenterY - Distance, p); // Top-Left
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX - Distance, CenterY + 0, p);        // Top-Center
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX - Distance, CenterY + Distance, p); // Top-Right
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX + 0, CenterY - Distance, p);        // Center-Left
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX + 0, CenterY + Distance, p);        // Center-Right
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX + Distance, CenterY - Distance, p); // Top-Left
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX + Distance, CenterY + 0, p);        // Top-Center
//         // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, CenterX + Distance, CenterY + Distance, p); // Top-Right
//         //  At this point, (bestResult->vec_x,bestResult->vec_y) marks the best search point and will be considered as the next search center
//         CenterX = bestResult->vec_x;
//         CenterY = bestResult->vec_y;
//         // Divides the search distance by 2
//         Distance >>= 1;
//     }
// }
// /************************************************************************************/
// void xTZ8PointDiamondSearch(BestResult *bestResult, int **CurrentBlock, int **SearchArea, int rowIdx, int colIdx, int centroX, int centroY, int iDist, Parameters p)
// {
//     BestResult localBest;
//     localBest.sad = bestResult->sad;
//     localBest.bestDist = iDist;
//     localBest.vec_x = 0;
//     localBest.vec_y = 0;

//     if (iDist == 1)
//     {
//         // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX - iDist, centroY, p);
//         // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, centroY - iDist, p);
//         // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, centroY + iDist, p);
//         // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX + iDist, centroY, p);
//     }
//     else
//     {
//         int iTop = centroY - iDist;
//         int iBottom = centroY + iDist;
//         int iLeft = centroX - iDist;
//         int iRight = centroX + iDist;
//         if (iDist <= 8)
//         {
//             int iTop_2 = centroY - (iDist >> 1);
//             int iBottom_2 = centroY + (iDist >> 1);
//             int iLeft_2 = centroX - (iDist >> 1);
//             int iRight_2 = centroX + (iDist >> 1);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, iTop, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft, centroY, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight, centroY, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, iBottom, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft_2, iTop_2, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight_2, iTop_2, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft_2, iBottom_2, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight_2, iBottom_2, p);
//         }
//         else
//         {
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, iTop, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iLeft, centroY, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iRight, centroY, p);
//             // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, centroX, iBottom, p);
//             for (int index = 1; index < 4; index++)
//             {
//                 int iPosYT = iTop + ((iDist >> 2) * index);
//                 int iPosYB = iBottom - ((iDist >> 2) * index);
//                 int iPosXL = centroX - ((iDist >> 2) * index);
//                 int iPosXR = centroX + ((iDist >> 2) * index);
//                 // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXL, iPosYT, p);
//                 // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXR, iPosYT, p);
//                 // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXL, iPosYB, p);
//                 // SAD(&(localBest), CurrentBlock, SearchArea, rowIdx, colIdx, iPosXR, iPosYB, p);
//             }
//         }
//     }
//     if (localBest.sad < bestResult->sad)
//     {
//         bestResult->sad = localBest.sad;
//         bestResult->bestDist = localBest.bestDist;
//         bestResult->vec_x = localBest.vec_x;
//         bestResult->vec_y = localBest.vec_y;
//     }
// }
// /************************************************************************************/
// void TZSearch(BestResult *bestResult, int **CurrentBlock, int **SearchArea, int rowIdx, int colIdx, Parameters p)
// {
//     int bestX, bestY;
//     bestResult->sad = BigSAD;
//     bestResult->bestDist = 0;
//     bestResult->vec_x = 0;
//     bestResult->vec_y = 0;

//     // First prediction, at the center of the search area
//     // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, 0, 0, p);

//     // Initial Search: iDist in [1, 2, 4, 8, 16, 32, 64]
//     int iDist = 1;
//     while (iDist <= p.searchRange)
//     {
//         xTZ8PointDiamondSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, 0, 0, iDist, p);
//         iDist <<= 1;
//     }

//     // Raster Search
//     bestX = bestResult->vec_x;
//     bestY = bestResult->vec_y;
//     if ((bestX > p.iRaster) || (bestY > p.iRaster) || (-bestX > p.iRaster) || (-bestY > p.iRaster))
//     {
//         int Top = -(int)(p.searchRange / 2);
//         int Bottom = (int)(p.searchRange / 2);
//         int Left = -(int)(p.searchRange / 2);
//         int Right = (int)(p.searchRange / 2);
//         for (int iStartY = Top; iStartY < Bottom; iStartY += p.iRaster)
//             for (int iStartX = Left; iStartX < Right; iStartX += p.iRaster)
//                 // SAD(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, iStartX, iStartY, p);
//                 printf("Ahhhhhh");
//     }

//     // Refinement
//     bestX = bestResult->vec_x;
//     bestY = bestResult->vec_y;
//     int RefinementCount = 0;
//     if ((bestX != 0) || (bestY != 0))
//         while ((bestResult->vec_x == bestX) && (bestResult->vec_y == bestY))
//         {
//             iDist = 1;
//             while (iDist <= p.searchRange)
//             {
//                 xTZ8PointDiamondSearch(bestResult, CurrentBlock, SearchArea, rowIdx, colIdx, bestX, bestY, iDist, p);

//                 if (((4 <= iDist) && (bestResult->bestDist == 0)) ||
//                     ((8 <= iDist) && (bestResult->bestDist <= 1)) ||
//                     ((16 <= iDist) && (bestResult->bestDist <= 2)) ||
//                     ((32 <= iDist) && (bestResult->bestDist <= 4)))
//                     break;

//                 iDist <<= 1;
//             }
//             if (((bestResult->vec_x == bestX) && (bestResult->vec_y == bestY)) || (RefinementCount == 7))
//                 break;
//             else
//             {
//                 bestX = bestResult->vec_x;
//                 bestY = bestResult->vec_y;
//                 RefinementCount += 1;
//             }
//         }
// }
// /************************************************************************************/