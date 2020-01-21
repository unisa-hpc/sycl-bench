#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

constexpr DATA_TYPE alpha = 123;
constexpr DATA_TYPE beta = 14512;

void init_arrays(DATA_TYPE* A, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < M; j++) {
			A[i * M + j] = ((DATA_TYPE)i * j) / N;
		}

		for(size_t j = 0; j < N; j++) {
			C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
		}
	}
}

void syrk(DATA_TYPE* A, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	/*  C := alpha*A*A' + beta*C */
	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			C[i * M + j] *= beta;
		}
	}

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			for(size_t k = 0; k < M; k++) {
				C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
			}
		}
	}
}

class Polybench_Syrk {
  public:
	Polybench_Syrk(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		C.resize(size * size);

		init_arrays(A.data(), C.data(), size);
	}

	void run() {
		using namespace cl::sycl;

		buffer<DATA_TYPE, 2> A_buffer{A.data(), range<2>(size, size)};
		buffer<DATA_TYPE, 2> C_buffer{C.data(), range<2>(size, size)};

		args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<class Syr2k2>(C_buffer.get_range(), [=, M_ = size](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				C[item] *= beta;

				for(size_t k = 0; k < M_; k++) {
					C[item] += alpha * A[{i, k}] * A[{j, k}];
				}
			});
		});
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> C_cpu(size * size);

		init_arrays(A.data(), C_cpu.data(), size);

		syrk(A.data(), C_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(C_cpu[i * size + j], C[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName() { return "Polybench_Syrk"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> C;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Syrk>();
	return 0;
}
