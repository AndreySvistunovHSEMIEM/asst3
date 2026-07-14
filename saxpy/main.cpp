#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cuda_runtime.h>

void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
void printCudaInfo();


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}


int main(int argc, char** argv)
{

    // default: arrays of 100M numbers
    int N = 1200 * 1600;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    const float alpha = 2.0f;
    // float* xarray = new float[N];
    // float* yarray = new float[N];
    // float* resultarray = new float[N];
    float* xarray;
    float* yarray;
    float* resultarray; 

    cudaError_t err = cudaMallocHost(&xarray, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallosHost failed %s\n", cudaGetErrorString(err));
    }

    err = cudaMallocHost(&yarray, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallosHost failed %s\n", cudaGetErrorString(err));
    }

    err = cudaMallocHost(&resultarray, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallosHost failed %s\n", cudaGetErrorString(err));
    }

    for (int i=0; i<N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
   }

    printCudaInfo();
    
    printf("Running 3 timing tests:\n");
    for (int i=0; i<3; i++) {
      saxpyCuda(N, alpha, xarray, yarray, resultarray);
    }

    cudaFreeHost(xarray);
    cudaFreeHost(yarray);
    cudaFreeHost(resultarray);

    return 0;
}
