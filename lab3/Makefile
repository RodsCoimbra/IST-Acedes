# C compiler
HOST_CC = gcc
HOST_CC_FLAGS = -O3

GPU_CC = nvcc
GPU_CC_FLAGS = -O3 -Wno-deprecated-gpu-targets
 
# Uncomment the following line to compile your GPU version
all: interPrediction_cpu interPrediction_gpu interPrediction_gpu_teste
#all: interPrediction_cpu

########### HOST SIDE (X86) COMPILATION ###########
interPrediction_cpu: interPrediction.c
	$(HOST_CC) $(HOST_CC_FLAGS) interPrediction.c -o interPrediction_cpu

########### GPU SIDE (HOST+GPU) COMPILATION ###########
interPrediction_gpu: interPrediction.cu 
	$(GPU_CC) $(GPU_CC_FLAGS) interPrediction.cu -o interPrediction_gpu

interPrediction_gpu_teste: interPrediction_teste.cu 
	$(GPU_CC) $(GPU_CC_FLAGS) interPrediction_teste.cu -o interPrediction_gpu_teste

clean:
	-@rm -rf interPrediction_cpu interPrediction_gpu interPrediction_gpu_teste reconst.yuv residue.yuv *~ *.o  2>/dev/null || true
