#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <vector_types.h>
#include <CCfits/CCfits>
#include "../include/file_io.h"

fileType set_fileType(std::string type) {
    if (type == "DR12" || type == "dr12") {
        return DR12;
    } else if (type == "DR12Ran" || type == "dr12ran") {
        return DR12Ran;
    } else if (type == "Patchy" || type == "patchy") {
        return Patchy;
    } else if (type == "PatchyRan" || type == "patchyran") {
        return PatchyRan;
    } else if (type == "LNKNLog") {
        return LNKNLog;
    } else if (type == "LNKNLogRan") {
        return LNKNLogRan;
    } else if (type == "LNKNLogRanBin") {
        return LNKNLogRanBin;
    } else if (type == "Gadget2") {
        return Gadget2;
    } else {
        std::stringstream errMsg;
        errMsg << "Unrecognized or unsupported file type." << std::endl;
        throw std::runtime_error(errMsg.str());
    }
}

int read_DR12(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, float3 r_min) {
    std::stringstream errMsg;
    errMsg << "DR12 support not yet implemented" << std::endl;
    throw std::runtime_error(errMsg.str());
}

int read_DR12Ran(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, 
                  float3 r_min) {
    std::stringstream errMsg;
    errMsg << "DR12Ran support not yet implemented" << std::endl;
    throw std::runtime_error(errMsg.str());
}

int read_Patchy(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, 
                 float3 r_min) {
    std::stringstream errMsg;
    errMsg << "Patchy support not yet implemented" << std::endl;
    throw std::runtime_error(errMsg.str());
}

int read_PatchyRan(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, 
                    float3 r_min) {
    std::stringstream errMsg;
    errMsg << "PatchyRan support not yet implemented" << std::endl;
    throw std::runtime_error(errMsg.str());
}

int read_LNKNLog(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, 
                  float3 r_min) {
    int3 N = {int(L.x/R), int(L.y/R), int(L.z/R)};
    parts.resize(N.x*N.y*N.z);
    int num = 0;
    std::ifstream fin(file);
    while (!fin.eof()) {
        double x, y, z, vx, vy, vz, b;
        fin >> x >> y >> z >> vx >> vy >> vz >> b;
        if (!fin.eof()) {
            float3 part = {(float)x, (float)y, (float)z};
            int i = part.x/R;
            int j = part.y/R;
            int k = part.z/R;
            int index = k + N.z*(j + N.y*i);
            parts[index].push_back(part);
            num++;
        }
    }
    fin.close();
    return num;
}

int read_LNKNLogRan(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, 
                     float3 r_min) {
    int3 N = {int(L.x/R), int(L.y/R), int(L.z/R)};
    parts.resize(N.x*N.y*N.z);
    int num = 0;
    std::ifstream fin(file);
    while (!fin.eof()) {
        double x, y, z;
        fin >> x >> y >> z;
        if (!fin.eof()) {
            float3 part = {(float)x, (float)y, (float)z};
            int i = part.x/R;
            int j = part.y/R;
            int k = part.z/R;
            int index = k + N.z*(j + N.y*i);
            parts[index].push_back(part);
            num++;
        }
    }
    fin.close();
    return num;
}

int read_LNKNLogRanBin(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, 
                       float3 r_min) {
    int3 N = {int(L.x/R), int(L.y/R), int(L.z/R)};
    parts.resize(N.x*N.y*N.z);
    int num;
    std::ifstream fin(file, std::ios::in|std::ios::binary);
    fin.read((char *)&num, sizeof(int));
    std::vector<float3> p(num);
    fin.read((char *)p.data(), num*sizeof(float3));
    for (int g = 0; g < p.size(); ++g) {
        float3 part = {p[g].x, p[g].y, p[g].z};
        int i = part.x/R;
        int j = part.y/R;
        int k = part.z/R;
        int index = k + N.z*(j + N.y*i);
        parts[index].push_back(part);
    }
    return num;
}

int read_Gadget2(std::string file, std::vector<std::vector<float3>> &parts, float3 L, float R, 
                  float3 r_min) {
    std::stringstream errMsg;
    errMsg << "Gadget2 support not yet implemented" << std::endl;
    throw std::runtime_error(errMsg.str());
}

int read_file(std::string file, fileType type, std::vector<std::vector<float3>> &parts, float3 L, float R, 
               float3 r_min) {
    int num = 0;
    switch (type) {
        case DR12:
            num = read_DR12(file, parts, L, R, r_min);
            break;
        case DR12Ran:
            num = read_DR12Ran(file, parts, L, R, r_min);
            break;
        case Patchy:
            num = read_Patchy(file, parts, L, R, r_min);
            break;
        case PatchyRan:
            num = read_PatchyRan(file, parts, L, R, r_min);
            break;
        case LNKNLog:
            num = read_LNKNLog(file, parts, L, R, r_min);
            break;
        case LNKNLogRan:
            num = read_LNKNLogRan(file, parts, L, R, r_min);
            break;
        case LNKNLogRanBin:
            num = read_LNKNLogRanBin(file, parts, L, R, r_min);
            break;
        case Gadget2:
            num = read_Gadget2(file, parts, L, R, r_min);
            break;
        default:
            std::stringstream errMsg;
            errMsg << "Unsupported file type" << std::endl;
            throw std::runtime_error(errMsg.str());
    }
    return num;
}

bool isTriangle(float r1, float r2, float r3) {
    if (r1 > r2) {
        float temp = r1;
        r1 = r2;
        r2 = temp;
    }
    if (r1 > r3) {
        float temp = r1;
        r1 = r3;
        r3 = temp;
    }
    if (r2 > r3) {
        float temp = r2;
        r2 = r3;
        r3 = temp;
    }
    if (r3 <= r1 + r2) {
        return true;
    } else {
        return false;
    }
}

void write_triangle_file(std::string file, std::vector<int> &DDD, std::vector<int> &DDR, std::vector<int> &DRR,
                         std::vector<int> &RRR, float R, int N_shells) {
    double dr = R/N_shells;
    std::ofstream fout(file);
    fout.precision(15);
    for (int i = 0; i < N_shells; ++i) {
        float r1 = (i + 0.5)*dr;
        for (int j = 0; j < N_shells; ++j) {
            float r2 = (j + 0.5)*dr;
            for (int k = 0; k < N_shells; ++k) {
                float r3 = (k + 0.5)*dr;
                if (isTriangle(r1, r2, r3)) {
                    int index = k + N_shells*(j + N_shells*i);
                    if (RRR[index] != 0) {
                        double result = ((double)DDD[index] - 3.0*double(DDR[index]) + 3.0*double(DRR[index]) - double(RRR[index]))/double(RRR[index]);
                        double result2 = ((double)DDD[index] - 3.0*DDR[index] + 2.0*RRR[index])/((double)RRR[index]);
                        fout << r1 << " " << r2 << " " << r3 << " " << DDD[index] << " " << DDR[index] << " ";
                        fout << DRR[index] << " " << RRR[index] << " " << result << " " << result2 << "\n";
                    }
                }
            }
        }
    }
    fout.close();
}

void write_triangle_file(std::string file, std::vector<int> &DDD, std::vector<float> &DDR, 
                         std::vector<int> &DRR, std::vector<int> &RRR, float R, int N_shells) {
    double dr = R/N_shells;
    std::ofstream fout(file);
    fout.precision(15);
    for (int i = 0; i < N_shells; ++i) {
        float r1 = (i + 0.5)*dr;
        for (int j = 0; j < N_shells; ++j) {
            float r2 = (j + 0.5)*dr;
            for (int k = 0; k < N_shells; ++k) {
                float r3 = (k + 0.5)*dr;
                if (isTriangle(r1, r2, r3)) {
                    int index = k + N_shells*(j + N_shells*i);
                    if (RRR[index] != 0) {
                        double result = ((double)DDD[index] - 3.0*double(DDR[index]) + 3.0*double(DRR[index]) - double(RRR[index]))/double(RRR[index]);
                        double result2 = ((double)DDD[index] - 3.0*DDR[index] + 2.0*RRR[index])/((double)RRR[index]);
                        fout << r1 << " " << r2 << " " << r3 << " " << DDD[index] << " " << DDR[index] << " ";
                        fout << DRR[index] << " " << RRR[index] << " " << result << " " << result2 << "\n";
                    }
                }
            }
        }
    }
    fout.close();
}

void write_2point_file(std::string file, std::vector<int> &DD, std::vector<int> &DR, int N_gal, int N_ran, 
                       float R, int N_shells) {
    double dr = R/N_shells;
    std::ofstream fout(file);
    fout.precision(15);
    for (int i = 0; i < N_shells; ++i) {
        float r = (i + 0.5)*dr;
        double result = (double(N_ran)*double(DD[i]))/(double(N_gal)*double(DR[i])) - 1.0;
        fout << r << " " << DD[i] << " " << DR[i] << " " << result << "\n";
    }
    fout.close();
}
