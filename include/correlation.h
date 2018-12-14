#ifndef _CORRELATION_H_
#define _CORRELATION_H_

#include <cuda.h>
#include <vector_types.h>

#define N_threads 64

__constant__ int3 d_shifts[27];
__constant__ float d_R;
__constant__ float3 d_L;
__constant__ int d_Nshells;
__constant__ int d_Nparts;

__device__ float get_separation(float3 r1, float3 r2) {
    return sqrtf((r1.x - r2.x)*(r1.x - r2.x) + (r1.y - r2.y)*(r1.y - r2.y) + (r1.z - r2.z)*(r1.z - r2.z));
}

__device__ int get_shell(float d1, float d2, float d3) {
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

__device__ int4 get_index(int4 ngp, int i, int3 n, float3 &rShift) {
    ngp.x += d_shifts[i].x;
    ngp.y += d_shifts[i].y;
    ngp.z += d_shifts[i].z;
    rShift.x = 0.0;
    rShift.y = 0.0;
    rShift.z = 0.0;
    if (ngp.x == n.x) {
        ngp.x = 0;
        rShift.x = d_L.x;
    }
    if (ngp.y == n.y) {
        ngp.y = 0;
        rShift.y = d_L.y;
    }
    if (ngp.z == n.z) {
        ngp.z = 0;
        rShift.z = d_L.z;
    }
    if (ngp.x == -1) {
        ngp.x = n.x - 1;
        rShift.x = -d_L.x;
    }
    if (ngp.y == -1) {
        ngp.y = n.y - 1;
        rShift.y = -d_L.y;
    }
    if (ngp.z == -1) {
        ngp.z = n.z - 1;
        rShift.z = -d_L.z;
    }
    ngp.w = ngp.z + n.z*(ngp.y + n.y*ngp.x);
    return ngp;
}

__global__ void countPairs(float3 *d_p1, float3 **d_p2, int *p2_sizes, int *d_partsPerShell, int3 n) {
    // Calculate the thread ID for the current GPU thread
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < d_Nparts) {
        float3 r1 = d_p1[tid];
        int4 ngp1 = {int(r1.x/d_R), int(r1.y/d_R), int(r1.z/d_R), 0};
        for (int i = 0; i < 27; ++i) {
            float3 rShift2;
            int4 index = get_index(ngp1, i, n, rShift2);
            int size2 = p2_sizes[index.w];
            for (int part2 = 0; part2 < size2; ++part2) {
                float3 r2 = d_p2[index.w][part2];
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

__global__ void countTriangles(float3 *d_p1, float3 **d_p2, float3 **d_p3, int *p2_sizes, int *p3_sizes, 
                               int *d_triangles, int3 n) {
    // Calculate the thread ID for the current GPU thread
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (tid < d_Nparts) {
        float3 r1 = d_p1[tid];
        int4 ngp1 = {int(r1.x/d_R), int(r1.y/d_R), int(r1.z/d_R), 0};
        for (int i = 0; i < 27; ++i) {
            float3 rShift2;
            int4 index = get_index(ngp1, i, n, rShift2);
            int size2 = p2_sizes[index.w];
            for (int part2 = 0; part2 < size2; ++part2) {
                float3 r2 = d_p2[index.w][part2];
                r2.x += rShift2.x;
                r2.y += rShift2.y;
                r2.z += rShift2.z;
                float d1 = get_separation(r1, r2);
                if (d1 < d_R && d1 > 0) {
                    for (int j = 0; j < 27; ++j) {
                        float3 rShift3;
                        int4 index2 = get_index(ngp1, j, n, rShift3);
                        if (abs(index.x - index2.x) < 2 && abs(index.y - index2.y) < 2 && abs(index.z - index2.z) < 2) {
                            int size3 = p3_sizes[index2.w];
                            for (int part3 = 0; part3 < size3; ++part3) {
                                float3 r3 = d_p3[index2.w][part3];
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
