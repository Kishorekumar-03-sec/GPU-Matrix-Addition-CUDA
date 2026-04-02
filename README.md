# GPU Matrix Addition using CUDA

##  Description
This project demonstrates matrix addition using CUDA programming on GPU. It uses parallel processing to compute results faster compared to CPU.

##  Objective
- To understand GPU parallel processing
- To implement CUDA kernel functions
- To perform matrix addition using GPU

##  Technologies Used
- CUDA C/C++
- NVIDIA GPU

##  How It Works
Each element of the matrix is processed simultaneously using multiple GPU threads, improving performance.

## Program
    %%cuda
    #include <stdio.h>
    
    __global__ void matrixAdd(int *A, int *B, int *C, int n) {
        int i = threadIdx.x;
        if (i < n) {
            C[i] = A[i] + B[i];
        }
    }
    
    int main() {
        int n = 5;
        int A[5] = {1,2,3,4,5};
        int B[5] = {5,4,3,2,1};
        int C[5];
    
        int *d_A, *d_B, *d_C;
    
        cudaMalloc((void**)&d_A, n*sizeof(int));
        cudaMalloc((void**)&d_B, n*sizeof(int));
        cudaMalloc((void**)&d_C, n*sizeof(int));
    
        cudaMemcpy(d_A, A, n*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n*sizeof(int), cudaMemcpyHostToDevice);
    
        matrixAdd<<<1, n>>>(d_A, d_B, d_C, n);
    
        cudaMemcpy(C, d_C, n*sizeof(int), cudaMemcpyDeviceToHost);
    
        printf("Result:\n");
        for(int i=0;i<n;i++){
            printf("%d ", C[i]);
        }
    
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    
        return 0;
    }

    
##  Output
Result:
6 6 6 6 6

## 📷 Screenshot
<img width="549" height="67" alt="image" src="https://github.com/user-attachments/assets/f4991c11-8853-4ea7-a03f-684aad979e9b" />


## 📽️ Demo Video
["C:\Users\admin\OneDrive\文件\PCA\PCA project demo video.mp4"](https://drive.google.com/file/d/1kLY3axz0Z3RQ8DYkf1KoN-CTU6LSEQQs/view?usp=sharing)
