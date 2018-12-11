#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <vector>
#include <string>
#include <vector_tpyes.h>

enum fileType{
    DR12,
    DR12Ran,
    Patchy,
    PatchyRan,
    LNKNLog,
    LNKNLogRan, 
    Gadget2
};

fileType set_fileType(std::string type);

void read_file(std::string file, fileType type, std::vector<std::vector<float3>> &parts, float3 L, float R, 
               float3 r_min);

void write_triangle_file(std::string file, std::vector<int> &DDD, std::vector<int> &DDR, std::vector<int> &DRR,
                         std::vector<int> &RRR, float R, int N_shells);

void write_2point_file(std::string file, std::vector<int> &DD, std::vector<int> &DR, int N_gal, int N_ran, 
                       float R, int N_shells);

#endif
