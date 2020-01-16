#include <iostream>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size
auto M = 2048;
auto N = 2048;

using DATA_TYPE = float;

constexpr DATA_TYPE float_n = 3214212.01;

void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i <= 0; i++) {
		for(j = 0; j <= N; j++) {
			if(percentDiff(symmat[i * (N + 1) + j], symmat_outputFromGpu[i * (N + 1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_arrays(DATA_TYPE* data) {
	int i, j;

	for(i = 0; i < M; i++) {
		for(j = 0; j < N; j++) {
			data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
		}
	}
}

void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean) {
	int i, j, j1, j2;

	// Determine mean of column vectors of input data matrix
	for(j = 1; j <= M; j++) {
		mean[j] = 0.0;
		for(i = 1; i <= N; i++) {
			mean[j] += data[i * (M + 1) + j];
		}
		mean[j] /= float_n;
	}

	// Center the column vectors.
	for(i = 1; i <= N; i++) {
		for(j = 1; j <= M; j++) {
			data[i * (M + 1) + j] -= mean[j];
		}
	}

	// Calculate the m * m covariance matrix.
	for(j1 = 1; j1 <= M; j1++) {
		for(j2 = j1; j2 <= M; j2++) {
			symmat[j1 * (M + 1) + j2] = 0.0;
			for(i = 1; i <= N; i++) {
				symmat[j1 * (M + 1) + j2] += data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
			}
			symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		M = problem_size;
		N = problem_size;
	}
	std::cout << "Problem size: " << M << "\n";

	std::vector<DATA_TYPE> data((M + 1) * (N + 1));
	std::vector<DATA_TYPE> symmat((M + 1) * (M + 1));
	std::vector<DATA_TYPE> mean(M + 1);

	init_arrays(data.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		covariance(data.data(), symmat.data(), mean.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		std::vector<DATA_TYPE> data_gpu((M + 1) * (N + 1));
		std::vector<DATA_TYPE> mean_gpu(M + 1);
		std::vector<DATA_TYPE> symmat_gpu((M + 1) * (M + 1));

		init_arrays(data_gpu.data());

		cl::sycl::queue queue;

		buffer<DATA_TYPE, 2> data_buffer{range<2>(M + 1, N + 1)};
		initDeviceBuffer(queue, data_buffer, data_gpu.data());

		buffer<DATA_TYPE, 2> mean_buffer{range<2>(M + 1, 1)};
		initDeviceBuffer(queue, mean_buffer, mean_gpu.data());

		buffer<DATA_TYPE, 2> symmat_buffer{range<2>(M + 1, M + 1)};
		initDeviceBuffer(queue, symmat_buffer, symmat_gpu.data());

		double t_start = rtclock();

		queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto mean = mean_buffer.get_access<access::mode::discard_write>(cgh);

			const auto pfor_range = nd_range<2>(range<2>(M, 1), {256, 1}, id<2>(1, 0));

			cgh.parallel_for<class CovarianceMean>(pfor_range, [=, N_ = N](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto j = item[0];

				mean[item] = 0;
				for(size_t i = 1; i <= N_; i++) {
					mean[item] += data[{i, j}];
				}
				mean[item] /= float_n;
			});
		});

		queue.submit([&](handler& cgh) {
			auto mean = mean_buffer.get_access<access::mode::read>(cgh);
			auto data = data_buffer.get_access<access::mode::read_write>(cgh);

			const auto pfor_range = nd_range<2>(range<2>(M, N), {8, 32}, id<2>(1, 1));

			cgh.parallel_for<class CovarianceReduce>(pfor_range, [=](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto j = item[1];
				data[item] -= mean[{j, 0}];
			});
		});

		queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);
			auto symmat2 = symmat_buffer.get_access<access::mode::discard_write>(cgh);

			const auto pfor_range = nd_range<2>(range<2>(M, 1), {256, 1}, id<2>(1, 0));

			cgh.parallel_for<class CovarianceCovar>(pfor_range, [=, M_ = M, N_ = N](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto j1 = item[0];

				symmat[{j1, j1}] = 1.0;

				for(size_t j2 = j1; j2 <= M_; j2++) {
					symmat[{j1, j2}] = 0.0;
					for(size_t i = 1; i <= N_; i++) {
						symmat[{j1, j2}] += data[{i, j1}] * data[{i, j2}];
					}

					symmat2[{j2, j1}] = symmat[{j1, j2}];
				}
			});
		});

		queue.wait();
		double t_end = rtclock();

		auto out = symmat_buffer.get_access<access::mode::read>(symmat_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(symmat.data(), out.get_pointer());
	}
}
