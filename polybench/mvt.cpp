#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Mvt1;
class Mvt2;

void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		x1[i] = 0.0;
		x2[i] = 0.0;
		y_1[i] = 0.0;
		y_2[i] = 0.0;

		for(size_t j = 0; j < N; j++) {
			a[i * N + j] = (DATA_TYPE)(i + j + 1.0) / N;
		}
	}
}

void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			x1[i] = x1[i] + a[i * N + j] * y1[j];
		}
	}

	for(size_t k = 0; k < N; k++) {
		for(size_t l = 0; l < N; l++) {
			x2[k] = x2[k] + a[k * N + l] * y2[l];
		}
	}
}

class Polybench_Mvt {
  public:
	Polybench_Mvt(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		a.resize(size * size);
		x1.resize(size);
		x2.resize(size);
		y1.resize(size);
		y2.resize(size);

		init_arrays(a.data(), x1.data(), x2.data(), y1.data(), y2.data(), size);
	}

	void run() {
		using namespace cl::sycl;

		buffer<DATA_TYPE, 2> a_buffer(a.data(), range<2>(size, size));
		buffer<DATA_TYPE, 1> x1_buffer(x1.data(), range<1>(size));
		buffer<DATA_TYPE, 1> x2_buffer(x2.data(), range<1>(size));
		buffer<DATA_TYPE, 1> y1_buffer(y1.data(), range<1>(size));
		buffer<DATA_TYPE, 1> y2_buffer(y2.data(), range<1>(size));

		args.device_queue.submit([&](handler& cgh) {
			auto a = a_buffer.get_access<access::mode::read>(cgh);
			auto y1 = y1_buffer.get_access<access::mode::read>(cgh);
			auto x1 = x1_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Mvt1>(x1_buffer.get_range(), [=, N_ = size](item<1> item) {
				const auto i = item[0];

				for(size_t j = 0; j < N_; j++) {
					x1[i] += a[{i, j}] * y1[j];
				}
			});
		});

		args.device_queue.submit([&](handler& cgh) {
			auto a = a_buffer.get_access<access::mode::read>(cgh);
			auto y2 = y2_buffer.get_access<access::mode::read>(cgh);
			auto x2 = x2_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Mvt2>(x1_buffer.get_range(), [=, N_ = size](item<1> item) {
				const auto k = item[0];

				for(size_t l = 0; l < N_; l++) {
					x2[k] += a[{k, l}] * y2[l];
				}
			});
		});
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> x1_cpu(size);
		std::vector<DATA_TYPE> x2_cpu(size);

		init_arrays(a.data(), x1_cpu.data(), x2_cpu.data(), y1.data(), y2.data(), size);

		runMvt(a.data(), x1_cpu.data(), x2_cpu.data(), y1.data(), y2.data(), size);

		for(size_t i = 0; i < size; i++) {
			auto diff = percentDiff(x1_cpu[i], x1[i]);
			if(diff > ERROR_THRESHOLD) return false;

			diff = percentDiff(x2_cpu[i], x2[i]);
			if(diff > ERROR_THRESHOLD) return false;
		}

		return true;
	}

	static std::string getBenchmarkName() { return "Polybench_Mvt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> a;
	std::vector<DATA_TYPE> x1;
	std::vector<DATA_TYPE> x2;
	std::vector<DATA_TYPE> y1;
	std::vector<DATA_TYPE> y2;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Mvt>();
	return 0;
}
