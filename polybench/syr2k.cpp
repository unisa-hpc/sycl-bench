#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size.
auto N = 2048;
auto M = 2048;

constexpr auto DIM_X = 32;
constexpr auto DIM_Y = 8;

using DATA_TYPE = float;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void compareResults(const DATA_TYPE* C, const DATA_TYPE* C_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			if(percentDiff(C[i * N + j], C_outputFromGpu[i * N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_arrays(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C) {
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
		}

		for(j = 0; j < M; j++) {
			A[i * N + j] = ((DATA_TYPE)i * j) / N;
			B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
		}
	}
}

void syr2k(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C) {
	int i, j, k;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			C[i * N + j] *= BETA;
		}
	}

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < M; k++) {
				C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
				C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
			}
		}
	}
}


int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		N = problem_size;
		M = problem_size;
	}
	std::cout << "Problem size: " << N << "\n";

	std::vector<DATA_TYPE> A(N * M);
	std::vector<DATA_TYPE> B(N * M);
	std::vector<DATA_TYPE> C(N * M);

	init_arrays(A.data(), B.data(), C.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		syr2k(A.data(), B.data(), C.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		std::vector<DATA_TYPE> C_gpu(N * M);

		init_arrays(A.data(), B.data(), C_gpu.data());

		cl::sycl::queue queue;

		buffer<DATA_TYPE, 2> A_buffer(range<2>(N, M));
		initDeviceBuffer(queue, A_buffer, A.data());

		buffer<DATA_TYPE, 2> B_buffer(range<2>(N, M));
		initDeviceBuffer(queue, B_buffer, B.data());

		buffer<DATA_TYPE, 2> C_buffer(range<2>(N, M));
		initDeviceBuffer(queue, C_buffer, C_gpu.data());

		double t_start = rtclock();

		queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			const auto pfor_range = nd_range<2>(C_buffer.get_range(), {DIM_Y, DIM_X});

			cgh.parallel_for<class Syr2k1>(pfor_range, [=, M_ = M](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto i = item[0];
				const auto j = item[1];

				C[item] *= BETA;

				for(size_t k = 0; k < M_; k++) {
					C[item] += ALPHA * A[{i, k}] * B[{j, k}] + ALPHA * B[{i, k}] * A[{j, k}];
				}
			});
		});

		queue.wait();
		double t_end = rtclock();

		auto C_gpu_result = C_buffer.get_access<access::mode::read>(C_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(C.data(), C_gpu_result.get_pointer());
	}
}
