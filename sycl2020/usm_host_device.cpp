#include "common.h"

template<typename DATA_TYPE>
class USMHostDeviceBenchmark{
protected:
	USMBuffer<DATA_TYPE> buff1;
	BenchmarkArgs args;
public:

	USMHostDeviceBenchmark(BenchmarkArgs args) : args(args) {}

	void setup(){
		//TODO
	}

	void run(std::vector<sycl::event>& events){
		//TODO
	}


	bool verify(VerificationSetting& settings){
		//TODO
	}

	static std::string getBenchmarkName() {
    	std::stringstream name;
    	//TODO
    	return name.str();
  	}

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    // const double copiedGiB =
    //     getBufferSize<Dims, false>(args.problem_size).size() * sizeof(DataT) / 1024.0 / 1024.0 / 1024.0;
    // return {copiedGiB, "GiB"};
	//TODO
  }


}




int main(int argc, char** argv){

}