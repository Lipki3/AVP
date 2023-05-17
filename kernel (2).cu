#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#define BLOCK_SIZE 256


using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

double tetrahedronVolume(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3, double x4, double y4, double z4) {
    double a[3], b[3], c[3], d[3];
    double ab[3], ac[3], ad[3];

    // Расчет векторов
    a[0] = x1;
    a[1] = y1;
    a[2] = z1;
    b[0] = x2;
    b[1] = y2;
    b[2] = z2;
    c[0] = x3;
    c[1] = y3;
    c[2] = z3;
    d[0] = x4;
    d[1] = y4;
    d[2] = z4;

    for (int i = 0; i < 3; i++) {
        ab[i] = b[i] - a[i];
        ac[i] = c[i] - a[i];
        ad[i] = d[i] - a[i];
    }

    // Вычисление объема
    double det = ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0]) + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
    return abs(det) / 6.0;
}
bool pointInsideTetrahedron(double x1, double y1, double z1,
    double x2, double y2, double z2,
    double x3, double y3, double z3,
    double x4, double y4, double z4,
    double xp, double yp, double zp) {

    // Вычисляем объем тетраэдра по формуле Герона:
    double det =  tetrahedronVolume(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
    // Вычисляем объемы тетраэдров, образованных точкой, которую нужно проверить, и каждой грани тетраэдра:
    double det1 = tetrahedronVolume(x1, y1, z1, x2, y2, z2, x3, y3, z3, xp, yp, zp);
    double det2 = tetrahedronVolume(x1, y1, z1, x2, y2, z2, x4, y4, z4, xp, yp, zp);
    double det3 = tetrahedronVolume(x1, y1, z1, x4, y4, z4, x3, y3, z3, xp, yp, zp);
    double det4 = tetrahedronVolume(x4, y4, z4, x2, y2, z2, x3, y3, z3, xp, yp, zp);
    // Если объемы всех тетраэдров совпадают, то точка находится внутри тетраэдра:
    return abs(det - det1 - det2 - det3 - det4) < 1e-6;
}
void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << " name: " << deviceProp.name << std::endl;
       // std::cout << "Device " << device << " manufacturer: " << deviceProp.manufacturer << std::endl;
        std::cout << "Device " << device << " global memory size: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "Device " << device << " shared memory size per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
       // std::cout << "Device " << device << " compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    }
}





__device__ double tetrahedronVolume2(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3, double x4, double y4, double z4) {
    double a[3], b[3], c[3], d[3];
    double ab[3], ac[3], ad[3];

    // Расчет векторов
    a[0] = x1;
    a[1] = y1;
    a[2] = z1;
    b[0] = x2;
    b[1] = y2;
    b[2] = z2;
    c[0] = x3;
    c[1] = y3;
    c[2] = z3;
    d[0] = x4;
    d[1] = y4;
    d[2] = z4;

    for (int i = 0; i < 3; i++) {
        ab[i] = b[i] - a[i];
        ac[i] = c[i] - a[i];
        ad[i] = d[i] - a[i];
    }

    // Вычисление объема
    double det = ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0]) + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
    return abs(det) / 6.0;
}
__device__ bool pointInsideTetrahedron2( double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3, double x4, double y4, double z4, double xp, double yp, double zp) {
    // Вычисляем объем тетраэдра по формуле Герона:
    double det = tetrahedronVolume2(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
    // Вычисляем объемы тетраэдров, образованных точкой, которую нужно проверить, и каждой грани тетраэдра:
    double det1 = tetrahedronVolume2(x1, y1, z1, x2, y2, z2, x3, y3, z3, xp, yp, zp);
    double det2 = tetrahedronVolume2(x1, y1, z1, x2, y2, z2, x4, y4, z4, xp, yp, zp);
    double det3 = tetrahedronVolume2(x1, y1, z1, x4, y4, z4, x3, y3, z3, xp, yp, zp);
    double det4 = tetrahedronVolume2(x4, y4, z4, x2, y2, z2, x3, y3, z3, xp, yp, zp);
    // Если объемы всех тетраэдров совпадают, то точка находится внутри тетраэдра:
    return abs(det - det1 - det2 - det3 - det4) < 1e-6;
}
double* random(int N) {
    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-1, 1);
    double* a = new double[N * 3]; 
    // Генерация N случайных точек в ограничивающем параллелепипеде
    for (int i = 0; i < N * 3; i++) {
        a[i] = dist(gen);
    }
    return a;
}
__global__ void countPointsInTetrahedron(int N, int* count, double* arr) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;
    int gridSize = gridDim.x * blockSize;
    if (blockIdx.x * blockDim.x + threadIdx.x >= N) return;
    int count_inside = 0;
    // Инициализация генератора случайных чисел
    


   if (pointInsideTetrahedron2(0, 0, -1, -1, 0, 0, 0, -1, 0, 1, 1, 1, arr[(blockIdx.x * blockDim.x + threadIdx.x)*3], arr[(blockIdx.x * blockDim.x + threadIdx.x) * 3 +1], arr[(blockIdx.x * blockDim.x + threadIdx.x)*3+2])){
            count_inside++;
    }
            
    // аггрегация на уровне блоков - parallel reduce
    __shared__ int s_count[BLOCK_SIZE];
    s_count[tid] = count_inside;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }
    // аггрегация на уровне сетки - atomicAdd
    if (tid == 0) {
        atomicAdd(count, s_count[0]);
    }
}




int main() {


    printDeviceInfo();
    double V_parall = 8.0; // Объем ограничивающего параллелепипеда
    const int N = 100000; // Количество генерируемых точек
    int count_inside = 0; // Счетчик точек внутри фигуры
    int count = 0;
    double count_h = 0;
    int* d_count;
    double* a = random(N);
    double* d_a;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMalloc((void**)&d_a, N * 3 * sizeof(double));
    cudaMemcpy(d_a, a, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1, 1);

    countPointsInTetrahedron << <gridDim, blockDim >> > (N, d_count, d_a);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    count_h = (double)count;

    cout << "Volume_GPU: " << V_parall * (double)count_h / (double)N << endl;

    cudaFree(d_count);
    cudaFree(d_a);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (int i = 0; i < N*3; i+=3) {
        double x = a[i];
        double y = a[i+1];
        double z = a[i+2];

        if (pointInsideTetrahedron(0, 0, -1, -1, 0, 0, 0, -1, 0, 1, 1, 1, x, y, z)) {
            count_inside++;
        }

    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  
    // Вычисление объема фигуры по методу ММК
  
    double V_fig = V_parall * (double)count_inside / (double)N;
    cout << "Volume_CPU: " << V_fig  << endl;
    printf("Time: %.6F ms\n", time_span.count() * 1000);
    cout << "Volume:"<< tetrahedronVolume(0, 0, -1, -1, 0, 0, 0, -1, 0, 1, 1, 1);


    return 0;
}
