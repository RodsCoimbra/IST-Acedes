#!/usr/bin/bash

# IMPORTANT: Uncomment *only* the line corresponding to the experiment you want to run. 
#            Do not uncomment all lines at once, to avoid overloading the server (shared 
#            by all students).

#######
#FSBM
clear
make clean
make
#./interPrediction_cpu ./video/video1_416x240_30.yuv 416 240 5 0 0
#./interPrediction_gpu ./video/video1_416x240_30.yuv 416 240 5 0 0
#TZS
#./interPrediction_cpu ./video/video1_416x240_30.yuv 416 240 5 1 0
#StepSearch
#./interPrediction_cpu ./video/video1_416x240_30.yuv 416 240 5 2 0


#######
#FSBM
#./interPrediction_cpu ./video/video2_1920x1080_4.yuv 1920 1080 4 0 0
./interPrediction_gpu ./video/video2_1920x1080_4.yuv 1920 1080 4 0 0
./interPrediction_gpu_teste ./video/video2_1920x1080_4.yuv 1920 1080 4 0 0
#TZS
#./interPrediction_cpu ./video/video2_1920x1080_4.yuv 1920 1080 4 1 0
#StepSearch
#./interPrediction_cpu ./video/video2_1920x1080_4.yuv 1920 1080 4 2 0


#######
#FSBM
#./interPrediction_cpu ./video/video3_1920x1080_60.yuv 1920 1080 60 0 0
#./interPrediction_gpu ./video/video3_1920x1080_60.yuv 1920 1080 60 0 0
#TZS
#./interPrediction_cpu ./video/video3_1920x1080_60.yuv 1920 1080 60 1 0
#StepSearch
#./interPrediction_cpu ./video/video3_1920x1080_60.yuv 1920 1080 60 2 0


#######
#FSBM
#./interPrediction_cpu ./video/video4_1920x1080_50.yuv 1920 1080 50 0 0
#./interPrediction_gpu ./video/video4_1920x1080_50.yuv 1920 1080 50 0 0
#TZS
#./interPrediction_cpu ./video/video4_1920x1080_50.yuv 1920 1080 50 1 0
#StepSearch
#./interPrediction_cpu ./video/video4_1920x1080_50.yuv 1920 1080 50 2 0
