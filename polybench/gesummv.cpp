#include <fstream>
#include <iostream>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto N = 4096;

using DATA_TYPE = float;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void compareResults(const DATA_TYPE* y, const DATA_TYPE* y_outputFromGpu) {
	int i, fail;
	fail = 0;

	for(i = 0; i < (N); i++) {
		if(percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x) {
	int i, j;

	for(i = 0; i < N; i++) {
		x[i] = 1;

		for(j = 0; j < N; j++) {
			A[i * N + j] = 2;
			B[i * N + j] = 3;
		}
	}
}

void gesummv(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp) {
	int i, j;

	for(i = 0; i < N; i++) {
		tmp[i] = 0;
		y[i] = 0;
		for(j = 0; j < N; j++) {
			tmp[i] = A[i * N + j] * x[j] + tmp[i];
			y[i] = B[i * N + j] * x[j] + y[i];
		}

		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		N = problem_size;
	}
	std::cout << "Problem size: " << N << "\n";

	std::vector<DATA_TYPE> A(N * N);
	std::vector<DATA_TYPE> B(N * N);
	std::vector<DATA_TYPE> x(N);
	std::vector<DATA_TYPE> y(N);
	std::vector<DATA_TYPE> tmp(N);

	init(A.data(), B.data(), x.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		gesummv(A.data(), B.data(), x.data(), y.data(), tmp.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		cl::sycl::queue queue;

		std::vector<DATA_TYPE> y_gpu(N);
		std::vector<DATA_TYPE> tmp_gpu(N);

		buffer<DATA_TYPE, 2> A_buffer{range<2>(N, N)};
		initDeviceBuffer(queue, A_buffer, A.data());
		buffer<DATA_TYPE, 2> B_buffer{range<2>(N, N)};
		initDeviceBuffer(queue, B_buffer, B.data());
		buffer<DATA_TYPE, 1> x_buffer{range<1>(N)};
		initDeviceBuffer(queue, x_buffer, x.data());

		buffer<DATA_TYPE, 1> y_buffer{range<1>(N)};
		initDeviceBuffer(queue, y_buffer, y_gpu.data());

		buffer<DATA_TYPE, 1> tmp_buffer{range<1>(N)};
		initDeviceBuffer(queue, tmp_buffer, tmp_gpu.data());

		double t_start = rtclock();

		queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto x = x_buffer.get_access<access::mode::read>(cgh);
			auto y = y_buffer.get_access<access::mode::read_write>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::read_write>(cgh);

			const auto pfor_range = nd_range<1>(y.get_range(), {256});

			cgh.parallel_for<class Gesummv>(pfor_range, [=, N_ = N](nd_item<1> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto i = item[0];

				for(size_t j = 0; j < N_; j++) {
					tmp[item] += A[{i, j}] * x[j];
					y[item] += B[{i, j}] * x[j];
				}

				y[item] = ALPHA * tmp[item] + BETA * y[item];
			});
		});

		queue.wait();
		double t_end = rtclock();

		auto out = y_buffer.get_access<access::mode::read>(y_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(y.data(), out.get_pointer());

		{
			std::ofstream dump("dump.txt");
			for(size_t i = 0; i < out.get_range().size(); i++) {
				dump << out[i] << "\n";
			}
		}
	}
}
