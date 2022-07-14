DPCPP_CXX = dpcpp
DPCPP_CXXFLAGS = -std=c++17 -g -o
DPCPP_LDFLAGS = -Werror
DPCPP_EXE_NAME = main
DPCPP_SOURCES = src/main.cpp


all:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)



run:
	./$(DPCPP_EXE_NAME)

clean: 
	rm -rf $(DPCPP_EXE_NAME)


