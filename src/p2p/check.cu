#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                 \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        printf("This test requires at least two CUDA-capable devices.\n");
        return 0;
    }

    for (int device1 = 0; device1 < deviceCount; device1++) {
        for (int device2 = 0; device2 < deviceCount; device2++) {
            if (device1 == device2)
                continue;
            
            int canAccessPeer;
            CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, device1, device2));
            printf("P2P access from device %d to %d: %s\n", device1, device2, canAccessPeer ? "Yes" : "No");
        }
        printf("\n");
    }

    return 0;
}
