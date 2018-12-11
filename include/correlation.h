#ifndef _CORRELATION_H_
#define _CORRELATION_H_

#include <cuda.h>
#include <vector_types.h>

#define N_threads 64

__constant__ int3 d_shifts[27];
__constant__ float d_R;
__constant__ float3 d_L;
__constant__ int d_Nshells;

__device__ float get_separation(float3 r1, float3 r2) {
    return sqrtf((r1.x - r2.x)*(r1.x - r2.x) + (r1.y - r2.y)*(r1.y - r2.y) + (r1.z - r2.z)*(r1.z - r2.z));
}

__device__ int get_shell(float &d1, float &d2, float &d3) {
    if (d1 > d2) {
        float temp = d1;
        d1 = d2;
        d2 = temp;
    }
    if (d1 > d3) {
        float temp = d1;
        d1 = d3;
        d3 = temp;
    }
    if (d2 > d3) {
        float temp = d2;
        d2 = d3;
        d3 = temp;
    }
    int shell1 = int(d1*d_Nshells/d_R);
    int shell2 = int(d2*d_Nshells/d_R);
    int shell3 = int(d3*d_Nshells/d_R);
    return shell3 + d_Nshells*(shell2 + d_Nshells*shell1);
}

__device__ int get_index(int x, int y, int z, int i, int3 n, float3 &rShift) {
    x += d_shifts[i].x;
    y += d_shifts[i].y;
    z += d_shifts[i].z;
    rShift.x = 0.0;
    rShift.y = 0.0;
    rShift.z = 0.0;
    if (x == n.x) {
        x = 0;
        rShift.x = d_L.x;
    }
    if (y == n.y) {
        y = 0;
        rShift.y = d_L.y;
    }
    if (z == n.z) {
        z = 0;
        rShift.z = d_L.z;
    }
    if (x == -1) {
        x = n.x - 1;
        rShift.x = -d_L.x;
    }
    if (y == -1) {
        y = n.y - 1;
        rShift.y = -d_L.y;
    }
    if (z == -1) {
        z = n.z - 1;
        rShift.z = -d_L.z;
    }
    return z + n.z*(y + n.y*x);
}

__global__ void countPairs(float3 **d_p1, float3 **d_p2, int *p1_sizes, int *p2_sizes, 
                           int *d_partsPerShell, int3 n) {
    // Calculate the thread ID for the current GPU thread
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < n.x*n.y*n.z) {
        int size1 = p1_sizes[tid];
        for (int part1 = 0; part1 < size1; ++part1) {
            float3 r1 = d_p1[tid][part1];
            int x = r1.x/d_R;
            int y = r1.y/d_R;
            int z = r1.z/d_R;
            for (int i = 0; i < 27; ++i) {
                float3 rShift2;
                int index = get_index(x, y, z, i, n, rShift2);
                int size2 = p2_sizes[index];
                for (int part2 = 0; part2 < size2; ++part2) {
                    float3 r2 = d_p2[index][part2];
                    r2.x += rShift2.x;
                    r2.y += rShift2.y;
                    r2.z += rShift2.z;
                    float dist = get_separation(r1, r2);
                    if (dist < d_R && dist > 0) {
                        int shell = int(dist*d_Nshells/d_R);
                        atomicAdd(&d_partsPerShell[shell], 1);
                    }
                }
            }
        }
    }
}

__global__ void countTrianlges(float3 **d_p1, float3 **d_p2, float3 **d_p3, int *p1_sizes, int *p2_sizes,
                               int *p3_sizes, int *d_triangles, int3 n) {
    // Calculate the thread ID for the current GPU thread
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < n.x*n.y*n.z) {
        int size1 = sizes[tid];
        for (int part1 = 0; part1 < size1; ++part1) {
            float3 r1 = d_p1[tid][part1];
            int x = r1.x/d_R;
            int y = r1.y/d_R;
            int z = r1.z/d_R;
            for (int i = 0; i < 27; ++i) {
                float3 rShift2;
                int index = get_index(x, y, z, i, n, rShift2);
                int size2 = sizes[index];
                for (int part2 = 0; part2 < size2; ++part2) {
                    float3 r2 = d_p2[index][part2];
                    r2.x += rShift2.x;
                    r2.y += rShift2.y;
                    r2.z += rShift2.z;
                    float d1 = get_separation(r1, r2);
                    if (d1 < d_R && d1 > 0) {
                        for (int j = 0; j < 27; ++j) {
                            float3 rShift3;
                            int index2 = get_index(x, y, z, j, n, rShift3);
                            int size3 = sizes[index2];
                            for (int part3 = 0; part3 < size3; ++part3) {
                                float3 r3 = d_p3[index2][part3];
                                r3.x += rShift3.x;
                                r3.y += rShift3.y;
                                r3.z += rShift3.z;
                                float d2 = get_separation(r1, r3);
                                float d3 = get_separation(r2, r3);
                                if (d2 < d_R && d3 < d_R && d2 > 0 && d3 > 0) {
                                    int shell = get_shell(d1, d2, d3);
                                    atomicAdd(&d_triangles[shell], 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<int3> get_shifts() {
    std::vector<int3> shifts;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                int3 temp = {i, j, k};
                shifts.push_back(temp);
            }
        }
    }
    return shifts;
}
#endif
