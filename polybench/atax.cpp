#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto NX = 4096;
auto NY = 4096;

constexpr auto DIM_THREAD_BLOCK_X = 256;
constexpr auto DIM_THREAD_BLOCK_Y = 1;

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* z, const DATA_TYPE* z_outputFromGpu) {
	int i, fail;
	fail = 0;

	for(i = 0; i < NY; i++) {
		if(percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(DATA_TYPE* x, DATA_TYPE* A) {
	int i, j;

	for(i = 0; i < NX; i++) {
		x[i] = i * M_PI;
		for(j = 0; j < NY; j++) {
			A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
		}
	}
}

void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp) {
	int i, j;

	for(i = 0; i < NY; i++) {
		y[i] = 0;
	}

	for(i = 0; i < NX; i++) {
		tmp[i] = 0;

		for(j = 0; j < NY; j++) {
			tmp[i] = tmp[i] + A[i * NY + j] * x[j];
		}

		for(j = 0; j < NY; j++) {
			y[j] = y[j] + A[i * NY + j] * tmp[i];
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		NX = problem_size;
		NY = problem_size;
	}
	std::cout << "Problem size: " << NX << "\n";

	std::vector<DATA_TYPE> A(NX * NY);
	std::vector<DATA_TYPE> x(NY);
	std::vector<DATA_TYPE> y(NY);
	std::vector<DATA_TYPE> tmp(NX);

	init_array(x.data(), A.data());

	if(shouldDoCpu()) {
		auto t_start = rtclock();
		atax_cpu(A.data(), x.data(), y.data(), tmp.data());
		auto t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		std::vector<DATA_TYPE> y_gpu(NY);
		std::vector<DATA_TYPE> tmp_gpu(NX);

		cl::sycl::queue queue;

		buffer<DATA_TYPE, 2> A_buffer{range<2>(NY, NX)};
		initDeviceBuffer(queue, A_buffer, A.data());

		buffer<DATA_TYPE, 2> x_buffer{range<2>(NY, 1)};
		initDeviceBuffer(queue, x_buffer, x.data());

		buffer<DATA_TYPE, 2> y_buffer{range<2>(NY, 1)};
		initDeviceBuffer(queue, y_buffer, y_gpu.data());

		buffer<DATA_TYPE, 2> tmp_buffer{range<2>(NX, 1)};
		initDeviceBuffer(queue, tmp_buffer, tmp_gpu.data());

		double t_start = rtclock();

		queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto x = x_buffer.get_access<access::mode::read>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::discard_write>(cgh);

			const auto range = nd_range<2>(tmp_buffer.get_range(), {DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y});

			cgh.parallel_for<class Atax1>(range, [=, NY_ = NY](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto i = item[0];

				for(size_t j = 0; j < NY_; j++) {
					tmp[item] += A[{i, j}] * x[{j, 0}];
				}
			});
		});

		queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::read>(cgh);
			auto y = y_buffer.get_access<access::mode::discard_write>(cgh);

			const auto range = nd_range<2>(y_buffer.get_range(), {DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y});

			cgh.parallel_for<class Atax2>(range, [=, NX_ = NX](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto j = item[0];

				for(size_t i = 0; i < NX_; i++) {
					y[item] += A[{i, j}] * tmp[{i, 0}];
				}
			});
		});

		queue.wait();
		double t_end = rtclock();

		auto out = y_buffer.get_access<access::mode::read>(y_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(y.data(), out.get_pointer());
	}
}
