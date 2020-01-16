#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto N = 4096;

constexpr auto DIM_X = 256;
constexpr auto DIM_Y = 1;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* x1, const DATA_TYPE* x1_outputFromGpu, const DATA_TYPE* x2, const DATA_TYPE* x2_outputFromGpu) {
	int i, fail;
	fail = 0;

	for(i = 0; i < N; i++) {
		if(percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;

		if(percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2) {
	int i, j;

	for(i = 0; i < N; i++) {
		x1[i] = 0.0;
		x2[i] = 0.0;
		y_1[i] = 0.0;
		y_2[i] = 0.0;

		for(j = 0; j < N; j++) {
			a[i * N + j] = (DATA_TYPE)(i + j + 1.0) / N;
		}
	}
}

void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2) {
	int i, j, k, l;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			x1[i] = x1[i] + a[i * N + j] * y1[j];
		}
	}

	for(k = 0; k < N; k++) {
		for(l = 0; l < N; l++) {
			x2[k] = x2[k] + a[k * N + l] * y2[l];
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) N = std::atoi(argv[1]);
	std::cout << "Problem size: " << N << "\n";

	std::vector<DATA_TYPE> a(N * N);
	std::vector<DATA_TYPE> x1(N);
	std::vector<DATA_TYPE> x2(N);
	std::vector<DATA_TYPE> y1(N);
	std::vector<DATA_TYPE> y2(N);

	init_arrays(a.data(), x1.data(), x2.data(), y1.data(), y2.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		runMvt(a.data(), x1.data(), x2.data(), y1.data(), y2.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		std::vector<DATA_TYPE> x1_gpu(N);
		std::vector<DATA_TYPE> x2_gpu(N);

		init_arrays(a.data(), x1_gpu.data(), x2_gpu.data(), y1.data(), y2.data());

		cl::sycl::queue queue;

		buffer<DATA_TYPE, 2> a_buffer(range<2>(N, N));
		initDeviceBuffer(queue, a_buffer, a.data());

		buffer<DATA_TYPE, 2> x1_buffer(range<2>(N, 1));
		initDeviceBuffer(queue, x1_buffer, x1_gpu.data());

		buffer<DATA_TYPE, 2> x2_buffer(range<2>(N, 1));
		initDeviceBuffer(queue, x2_buffer, x2_gpu.data());

		buffer<DATA_TYPE, 2> y1_buffer(range<2>(N, 1));
		initDeviceBuffer(queue, y1_buffer, y1.data());

		buffer<DATA_TYPE, 2> y2_buffer(range<2>(N, 1));
		initDeviceBuffer(queue, y2_buffer, y2.data());

		double t_start = rtclock();

		queue.submit([&](handler& cgh) {
			auto a = a_buffer.get_access<access::mode::read>(cgh);
			auto y1 = y1_buffer.get_access<access::mode::read>(cgh);
			auto x1 = x1_buffer.get_access<access::mode::read_write>(cgh);

			const auto pfor_range = nd_range<2>(x1_buffer.get_range(), {DIM_X, DIM_Y});

			cgh.parallel_for<class Mvt1>(pfor_range, [=, N_ = N](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto i = item[0];

				for(size_t j = 0; j < N_; j++) {
					x1[{i, 0}] += a[{i, j}] * y1[{j, 0}];
				}
			});
		});

		queue.submit([&](handler& cgh) {
			auto a = a_buffer.get_access<access::mode::read>(cgh);
			auto y2 = y2_buffer.get_access<access::mode::read>(cgh);
			auto x2 = x2_buffer.get_access<access::mode::read_write>(cgh);

			const auto pfor_range = nd_range<2>(x1_buffer.get_range(), {DIM_X, DIM_Y});

			cgh.parallel_for<class Mvt2>(pfor_range, [=, N_ = N](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto k = item[0];

				for(size_t l = 0; l < N_; l++) {
					x2[{k, 0}] += a[{k, l}] * y2[{l, 0}];
				}
			});
		});


		queue.wait();
		double t_end = rtclock();

		auto x1_gpu_result = x1_buffer.get_access<access::mode::read>(x1_buffer.get_range());
		auto x2_gpu_result = x2_buffer.get_access<access::mode::read>(x2_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(x1.data(), x1_gpu_result.get_pointer(), x2.data(), x2_gpu_result.get_pointer());
	}
}
