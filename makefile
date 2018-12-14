CXX = cuda-g++
ARCHS = -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61
VXX = nvcc $(ARCHS) -ccbin=cuda-g++
CXXOPTS = -march=native -mtune=native -O3
CCXFLAGS = -lCCfits -lcfitsio
VXXFLAGS = -Xptxas -dlcm=ca -lineinfo --compiler-options "$(CXXFLAGS) $(CXXOPTS)" -O3

build: harppi file_io main.cu
	$(VXX) $(VXXFLAGS) -o $(HOME)/bin/cagtpce main.cu obj/*.o
	
harppi: include/harppi.h source/harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXOPTS) -c -o obj/harppi.o source/harppi.cpp
	
file_io: include/file_io.h source/file_io.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/file_io.o source/file_io.cpp
	
