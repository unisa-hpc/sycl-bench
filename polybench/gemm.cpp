#include <iostream>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Problem size
auto NI = 512;
auto NJ = 512;
auto NK = 512;

#define ALPHA 32412
#define BETA 2123

constexpr auto DIM_X = 32;
constexpr auto DIM_Y = 8;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* C, const DATA_TYPE* C_outputFromGpu) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NJ; j++) {
			if(percentDiff(C[i * NJ + j], C_outputFromGpu[i * NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C) {
	int i, j;

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NK; j++) {
			A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(i = 0; i < NK; i++) {
		for(j = 0; j < NJ; j++) {
			B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
		}
	}

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NJ; j++) {
			C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
		}
	}
}

void gemm(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C) {
	int i, j, k;

	for(i = 0; i < NI; i++) {
		for(j = 0; j < NJ; j++) {
			C[i * NJ + j] *= BETA;

			for(k = 0; k < NK; ++k) {
				C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
			}
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		NI = problem_size;
		NJ = problem_size;
		NK = problem_size;
	}
	std::cout << "Problem size: " << NI << "\n";

	std::vector<DATA_TYPE> A(NI * NK);
	std::vector<DATA_TYPE> B(NK * NJ);
	std::vector<DATA_TYPE> C(NI * NJ);

	init(A.data(), B.data(), C.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		gemm(A.data(), B.data(), C.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		std::vector<DATA_TYPE> C_gpu(NI * NJ);
		init(A.data(), B.data(), C_gpu.data());

		cl::sycl::queue queue;

		buffer<DATA_TYPE, 2> A_buffer{range<2>(NI, NK)};
		initDeviceBuffer(queue, A_buffer, A.data());
		buffer<DATA_TYPE, 2> B_buffer{range<2>(NK, NJ)};
		initDeviceBuffer(queue, B_buffer, B.data());
		buffer<DATA_TYPE, 2> C_buffer{range<2>(NI, NJ)};
		initDeviceBuffer(queue, C_buffer, C_gpu.data());

		double t_start = rtclock();

		queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			const auto pfor_range = nd_range<2>(C_buffer.get_range(), {DIM_Y, DIM_X});

			cgh.parallel_for<class Gemm>(pfor_range, [=, NK_ = NK](nd_item<2> nd_item) {
				const auto item = nd_item.get_global_id();
				const auto i = item[0];
				const auto j = item[1];

				C[item] *= BETA;

				for(size_t k = 0; k < NK_; k++) {
					C[item] += ALPHA * A[{i, k}] * B[{k, j}];
				}
			});
		});

		queue.wait();
		double t_end = rtclock();

		auto out = C_buffer.get_access<access::mode::read>(C_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(C.data(), out.get_pointer());
	}
}
