#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Gesummv;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

// void compareResults(const DATA_TYPE* y, const DATA_TYPE* y_outputFromGpu) {
// 	int i, fail;
// 	fail = 0;

// 	for(i = 0; i < (N); i++) {
// 		if(percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
// 	}

// 	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
// }

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		x[i] = 1;

		for(size_t j = 0; j < N; j++) {
			A[i * N + j] = 2;
			B[i * N + j] = 3;
		}
	}
}

void gesummv(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		tmp[i] = 0;
		y[i] = 0;
		for(size_t j = 0; j < N; j++) {
			tmp[i] = A[i * N + j] * x[j] + tmp[i];
			y[i] = B[i * N + j] * x[j] + y[i];
		}

		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

class Polybench_Gesummv {
  public:
	Polybench_Gesummv(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		x.resize(size);
		y.resize(size);
		tmp.resize(size);

		init(A.data(), B.data(), x.data(), size);
	}

	void run() {
		using namespace cl::sycl;

		buffer<DATA_TYPE, 2> A_buffer{A.data(), range<2>(size, size)};
		buffer<DATA_TYPE, 2> B_buffer{B.data(), range<2>(size, size)};
		buffer<DATA_TYPE, 1> x_buffer{x.data(), range<1>(size)};
		buffer<DATA_TYPE, 1> y_buffer{y.data(), range<1>(size)};
		buffer<DATA_TYPE, 1> tmp_buffer{tmp.data(), range<1>(size)};

		args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto x = x_buffer.get_access<access::mode::read>(cgh);
			auto y = y_buffer.get_access<access::mode::read_write>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Gesummv>(y.get_range(), [=, N_ = size](item<1> item) {
				const auto i = item[0];

				for(size_t j = 0; j < N_; j++) {
					tmp[item] += A[{i, j}] * x[j];
					y[item] += B[{i, j}] * x[j];
				}

				y[item] = ALPHA * tmp[item] + BETA * y[item];
			});
		});
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> y_cpu(size);
		std::vector<DATA_TYPE> tmp_cpu(size);

		gesummv(A.data(), B.data(), x.data(), y_cpu.data(), tmp_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			const auto diff = percentDiff(y_cpu[i], y[i]);
			if(diff > ERROR_THRESHOLD) return false;
		}

		return true;
	}

	static std::string getBenchmarkName() { return "Polybench_Gesummv"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> x;
	std::vector<DATA_TYPE> y;
	std::vector<DATA_TYPE> tmp;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Gesummv>();
	return 0;
}
