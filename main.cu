#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <vector_types.h>
#include "include/correlation.h"
#include "include/file_io.h"
#include "include/harppi.h"

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    std::cout << "Initial setup..." << std::endl;
    float3 L = {(float)p.getd("Lx"), (float)p.getd("Ly"), (float)p.getd("Lz")};
    float R = (float)p.getd("R");
    int N_shells = p.geti("N_shells");
    
    std::vector<int3> shifts = get_shifts();
    
    cudaMemcpyToSymbol(d_shifts, shifts.data(), shifts.size()*sizeof(int3));
    cudaMemcpyToSymbol(d_R, &R, sizeof(float));
    cudaMemcpyToSymbol(d_L, &L, sizeof(float3));
    cudaMemcpyToSymbol(d_Nshells, &N_shells, sizeof(int));
    
    int3 N = {int(L.x/R), int(L.y/R), int(L.z/R)};
    float3 r_min = {(float)p.getd("r_minx"), (float)p.getd("r_miny"), (float)p.getd("r_minz")};
    
    std::cout << "Reading in galaxies and randoms..." << std::endl;
    std::vector<std::vector<float3>> gals;
    std::vector<std::vector<float3>> rans;
    std::vector<float3> gs;
    std::vector<float3> rs;
    
    fileType galFileType = set_fileType(p.gets("galFileType"));
    fileType ranFileType = set_fileType(p.gets("ranFileType"));
    int num_gals = read_file(p.gets("gal_file"), galFileType, gals, L, R, r_min);
    int num_rans = read_file(p.gets("ran_file"), ranFileType, rans, L, R, r_min);
    std::cout << "num_gals = " << num_gals << std::endl;
    std::cout << "num_rans = " << num_rans << std::endl;
    std::cout << "gals.size() = " << gals.size() << std::endl;
    std::cout << "rans.size() = " << rans.size() << std::endl;
    cudaMemcpyToSymbol(d_Nparts, &num_gals, sizeof(int));
    
    std::cout << "Setting up storage..." << std::endl;
    int Nshells3 = N_shells*N_shells*N_shells;
    std::vector<int> DD(N_shells), DR(N_shells), DDD(Nshells3), DDR(Nshells3), DRR(Nshells3), RRR(Nshells3);
    std::vector<int> galSizes, ranSizes;
    int *d_DD, *d_DR, *d_DDD, *d_DDR, *d_DRR, *d_RRR;
    float3 **d_gals, **d_rans;
    int *d_galSizes, *d_ranSizes;
    float3 *d_gs, *d_rs;
    
    float3 **h_gals = (float3 **)malloc(gals.size()*sizeof(float3 *)); 
    float3 **h_rans = (float3 **)malloc(rans.size()*sizeof(float3 *));
    for (int i = 0; i < gals.size(); ++i) {
        galSizes.push_back(gals[i].size());
        ranSizes.push_back(rans[i].size());
        cudaMalloc((void **)&h_gals[i], gals[i].size()*sizeof(float3));
        cudaMemcpy(h_gals[i], gals[i].data(), gals[i].size()*sizeof(float3), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&h_rans[i], rans[i].size()*sizeof(float3));
        cudaMemcpy(h_rans[i], rans[i].data(), rans[i].size()*sizeof(float3), cudaMemcpyHostToDevice);
        for (int j = 0; j < gals[i].size(); ++j)
            gs.push_back(gals[i][j]);
        for (int j = 0; j < rans[i].size(); ++j)
            rs.push_back(rans[i][j]);
    }
    cudaMalloc(&d_gals, gals.size()*sizeof(float3 *));
    cudaMemcpy(d_gals, h_gals, gals.size()*sizeof(float3 *), cudaMemcpyHostToDevice);
    cudaMalloc(&d_rans, rans.size()*sizeof(float3 *));
    cudaMemcpy(d_rans, h_rans, rans.size()*sizeof(float3 *), cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&d_DD, DD.size()*sizeof(int));
    cudaMalloc((void **)&d_DR, DR.size()*sizeof(int));
    cudaMalloc((void **)&d_DDD, DDD.size()*sizeof(int));
    cudaMalloc((void **)&d_DDR, DDR.size()*sizeof(int));
    cudaMalloc((void **)&d_DRR, DRR.size()*sizeof(int));
    cudaMalloc((void **)&d_RRR, RRR.size()*sizeof(int));
    cudaMalloc((void **)&d_galSizes, galSizes.size()*sizeof(int));
    cudaMalloc((void **)&d_ranSizes, ranSizes.size()*sizeof(int));
    cudaMalloc((void **)&d_gs, gs.size()*sizeof(float3));
    cudaMalloc((void **)&d_rs, rs.size()*sizeof(float3));
    
    cudaMemcpy(d_DD, DD.data(), DD.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_DR, DR.data(), DR.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_DDD, DDD.data(), DDD.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_DDR, DDR.data(), DDR.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_DRR, DRR.data(), DRR.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RRR, RRR.data(), RRR.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_galSizes, galSizes.data(), galSizes.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ranSizes, ranSizes.data(), ranSizes.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gs, gs.data(), gs.size()*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rs, rs.data(), rs.size()*sizeof(float3), cudaMemcpyHostToDevice);
    
    int num_blocks = num_gals/N_threads + 1;
    std::cout << "Two point function..." << std::endl;
    countPairs<<<num_blocks, N_threads>>>(d_gs, d_gals, d_galSizes, d_DD, N);
    countPairs<<<num_blocks, N_threads>>>(d_gs, d_rans, d_ranSizes, d_DR, N);
    
    std::cout << "Three point function..." << std::endl;
    std::cout << "   DDD..." << std::endl;
    countTriangles<<<num_blocks, N_threads>>>(d_gs, d_gals, d_gals, d_galSizes, d_galSizes, d_DDD, N);
    std::cout << "   DDR..." << std::endl;
    countTriangles<<<num_blocks, N_threads>>>(d_gs, d_gals, d_rans, d_galSizes, d_ranSizes, d_DDR, N);
    std::cout << "   DRR..." << std::endl;
    countTriangles<<<num_blocks, N_threads>>>(d_gs, d_rans, d_rans, d_ranSizes, d_ranSizes, d_DRR, N);
    std::cout << "   RRR..." << std::endl;
    num_blocks = num_rans/N_threads + 1;
    cudaMemcpyToSymbol(d_Nparts, &num_rans, sizeof(int));
    countTriangles<<<num_blocks, N_threads>>>(d_rs, d_rans, d_rans, d_ranSizes, d_ranSizes, d_RRR, N);
    
    cudaMemcpy(DD.data(), d_DD, DD.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(DR.data(), d_DR, DR.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(DDD.data(), d_DDD, DDD.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(DDR.data(), d_DDR, DDR.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(DRR.data(), d_DRR, DRR.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(RRR.data(), d_RRR, RRR.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    write_2point_file(p.gets("twoPointFile"), DD, DR, num_gals, num_rans, R, N_shells);
    write_triangle_file(p.gets("threePointFile"), DDD, DDR, DRR, RRR, R, N_shells);
    
    num_blocks = num_gals/N_threads + 1;
    double n_bar = num_gals/(L.x*L.y*L.z);
    double Delta_r = R/N_shells;
    for (int i = 0; i < DDR.size(); ++i)
        DDR[i] = 0;
    cudaMemcpy(d_DDR, DDR.data(), DDR.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    getDDR<<<num_blocks, N_threads>>>(d_gs, d_gals, d_galSizes, d_DDR, N, n_bar);
    cudaMemcpy(DDR.data(), d_DDR, DDR.size()*sizeof(int), cudaMemcpyDeviceToHost);
    
    RRR = getRRR(Delta_r, n_bar, num_rans, N_shells);
    
    write_triangle_file("predicted.dat", DDD, DDR, RRR, RRR, R, N_shells);
    
    cudaFree(d_DD);
    cudaFree(d_DR);
    cudaFree(d_DDD);
    cudaFree(d_DDR);
    cudaFree(d_DRR);
    cudaFree(d_RRR);
    cudaFree(d_galSizes);
    cudaFree(d_ranSizes);
    cudaFree(d_gals);
    cudaFree(d_rans);
    cudaFree(d_gs);
    cudaFree(d_rs);
    delete[] h_gals;
    delete[] h_rans;
    
    return 0;
}
    
    
    
