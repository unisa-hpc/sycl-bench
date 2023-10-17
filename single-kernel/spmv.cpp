#include <sycl/sycl.hpp>
#include <iostream>

#include "common.h"

namespace s = sycl;


class SpmvKernel; // kernel forward declaration


class Spmv
{
protected:
    size_t size; // user-defined size (input and output will be size x size)
    size_t local_size;
    BenchmarkArgs args;

	
    std::vector<int> row_b;
    std::vector<int> row_e;
    std::vector<int> vec;
    std::vector<int> output;
    
    std::vector<int> val;
    std::vector<int> col;

    PrefetchedBuffer<int, 1> buf_row_b;
    PrefetchedBuffer<int, 1> buf_row_e;   
    PrefetchedBuffer<int, 1> buf_vec;
    PrefetchedBuffer<int, 1> buf_output;
    PrefetchedBuffer<int, 1> buf_val;
    PrefetchedBuffer<int, 1> buf_col;   



public:
  Spmv(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {
    size = args.problem_size; // input size defined by the user
    local_size = args.local_size; // set local work_group size
    row_b.resize(size);
    row_e.resize(size);
    vec.resize(size);
    output.resize(size);

    srand(42);
	//int min = rsize* 1 / 100; // min number of value for row
	//int max = rsize* 3 / 100; // max..
	row_b[0] = 0;
	row_e[0] = 20;
	for(int i=1; i < size; ++i) {
		//int rnd_value = rand() % (max - min) + min;
		row_b[i] = row_b[i-1] + 20;
		row_e[i] = row_e[i-1] + 20;
		vec[i] = 1;
	}

    int len = row_e[size-1];
    val.resize(len);
    col.resize(len);

    for(int i=0; i < len; ++i){
		val[i] = 2;
		col[i] = rand() % (size-1);
	}
	
   
    // init buffer
    buf_row_b.initialize(args.device_queue, row_b.data(), s::range<1>(size));
    buf_row_e.initialize(args.device_queue, row_e.data(), s::range<1>(size));
    buf_vec.initialize(args.device_queue, vec.data(), s::range<1>(size));
    buf_val.initialize(args.device_queue, val.data(), s::range<1>(len));
    buf_col.initialize(args.device_queue, col.data(), s::range<1>(len));
    buf_output.initialize(args.device_queue, output.data(), s::range<1>(size));    
    
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
    auto row_b_acc = buf_row_b.get_access<s::access::mode::read>(cgh);
    auto row_e_acc = buf_row_e.get_access<s::access::mode::read>(cgh);
    auto vec_acc = buf_vec.get_access<s::access::mode::read>(cgh);
    auto val_acc = buf_val.get_access<s::access::mode::read>(cgh);
    auto col_acc = buf_col.get_access<s::access::mode::read>(cgh);
    auto output_acc = buf_output.get_access<s::access::mode::write>(cgh);


    s::range<1> ndrange{size};

      cgh.parallel_for<class SpmvKernel>(ndrange, [this,row_b_acc, row_e_acc, vec_acc, val_acc, col_acc, output_acc, num_elements = size](s::id<1> id) {
        int gid = id[0];
        if (gid >= num_elements) return;
        int sum = 0;
        int start = row_b_acc[gid];
        int stop  = row_e_acc[gid];
        for (int j = start; j < stop; ++j) {
            int c = col_acc[j];
            sum += val_acc[j] * vec_acc[c];
        }
        output_acc[gid] = sum;
      });
    }));
   }
    
  bool verify(VerificationSetting& ver) {
    unsigned int check = 1;
    buf_output.reset();
	for(unsigned int i = 0; i < size; ++i) {
		int sum = 0;
		int start = row_b[i];
		int stop =  row_e[i];
		for (int j = start; j < stop; ++j){
			int c = col[j];
			sum += val[j] * vec[c];
		}
		if(output[i] != sum) {
			check = 0;
			printf("= fail at %d, expected %d / actual %d\n", i, sum, output[i]);
			break;
		}
	}

	return check ? true : false;
  }


  static std::string getBenchmarkName() { return "Spmv"; }

}; // Spmv class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<Spmv>();  
  return 0;
}


