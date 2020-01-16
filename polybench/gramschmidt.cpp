#include <iostream>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto M = 2048;
auto N = 2048;

constexpr auto DIM_X = 256;
constexpr auto DIM_Y = 1;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* A, const DATA_TYPE* A_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < M; i++) {
		for(j = 0; j < N; j++) {
			if(percentDiff(A[i * N + j], A_outputFromGpu[i * N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
				fail++;
				// printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i * N + j], A_outputFromGpu[i * N + j]);
			}
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(DATA_TYPE* A) {
	int i, j;

	for(i = 0; i < M; i++) {
		for(j = 0; j < N; j++) {
			A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
		}
	}
}

void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q) {
	int i, j, k;
	DATA_TYPE nrm;
	for(k = 0; k < N; k++) {
		nrm = 0;
		for(i = 0; i < M; i++) {
			nrm += A[i * N + k] * A[i * N + k];
		}

		R[k * N + k] = sqrt(nrm);
		for(i = 0; i < M; i++) {
			Q[i * N + k] = A[i * N + k] / R[k * N + k];
		}

		for(j = k + 1; j < N; j++) {
			R[k * N + j] = 0;
			for(i = 0; i < M; i++) {
				R[k * N + j] += Q[i * N + k] * A[i * N + j];
			}
			for(i = 0; i < M; i++) {
				A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
			}
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

	std::vector<DATA_TYPE> A(M * N);
	std::vector<DATA_TYPE> R(M * N);
	std::vector<DATA_TYPE> Q(M * N);

	init_array(A.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		gramschmidt(A.data(), R.data(), Q.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		std::vector<DATA_TYPE> A_gpu_init(M * N);
		init_array(A_gpu_init.data());

		cl::sycl::queue queue;

		buffer<DATA_TYPE, 2> A_buffer{cl::sycl::range<2>(M, N)};
		initDeviceBuffer(queue, A_buffer, A_gpu_init.data());

		buffer<DATA_TYPE, 2> R_buffer{range<2>(M, N)};
		buffer<DATA_TYPE, 2> Q_buffer{range<2>(M, N)};

		double t_start = rtclock();

		for(size_t k = 0; k < N; k++) {
			queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh);
				auto R = R_buffer.get_access<access::mode::write>(cgh);

				// TODO: use reduction
				cgh.parallel_for<class Gramschmidt1>(range<2>(1, 1), [=, M_ = M](item<2> item) {
					DATA_TYPE nrm = 0;
					for(size_t i = 0; i < M_; i++) {
						nrm += A[{i, k}] * A[{i, k}];
					}
					R[{k, k}] = sqrt(nrm);
				});
			});

			queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh);
				auto R = R_buffer.get_access<access::mode::read>(cgh);
				auto Q = Q_buffer.get_access<access::mode::write>(cgh);

				const auto pfor_range = nd_range<2>(range<2>(M, 1), {DIM_X, DIM_Y}, id<2>(0, k));

				cgh.parallel_for<class Gramschmidt2>(pfor_range, [=](nd_item<2> nd_item) {
					const auto item = nd_item.get_global_id();
					Q[item] = A[item] / R[{k, k}];
				});
			});

			queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read_write>(cgh);
				auto R = R_buffer.get_access<access::mode::write>(cgh);
				auto Q = Q_buffer.get_access<access::mode::read>(cgh);

				const auto pfor_range = nd_range<2>(range<2>(M, 1), {DIM_X, DIM_Y});

				cgh.parallel_for<class Gramschmidt3>(pfor_range, [=, M_ = M, N_ = N](nd_item<2> nd_item) {
					const auto item = nd_item.get_global_id();
					const auto j = item[0];

					if(j <= k || j >= N_) return;

					R[item] = 0;
					for(size_t i = 0; i < M_; i++) {
						R[item] += Q[{i, k}] * A[{i, j}];
					}

					for(size_t i = 0; i < M_; i++) {
						A[{i, j}] -= Q[{i, k}] * R[item];
					}
				});
			});
		}

		queue.wait();
		double t_end = rtclock();

		auto out = A_buffer.get_access<access::mode::read>(A_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(A.data(), out.get_pointer());
	}
}
