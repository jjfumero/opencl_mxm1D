all: gcc 

gcc:
	g++ -c readSource.cpp 
	g++ -std=c++0x mxm.cpp readSource.o -o mxm -lOpenCL

debug:
	g++ -g -c readSource.cpp 
	g++ -g -std=c++0x mxm.cpp readSource.o -o mxm -lOpenCL


run:
	./mxm 1024

clean:
	rm *.o mxm
