#include <iostream>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "polybenchUtilFuncts.h"
#include "syclUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

// Problem size
auto TMAX = 500;
auto NX = 2048;
auto NY = 2048;

const auto DIM_X = 32;
const auto DIM_Y = 8;

using DATA_TYPE = float;

void compareResults(const DATA_TYPE* hz1, const DATA_TYPE* hz2) {
	int i, j, fail;
	fail = 0;

	for(i = 0; i < NX; i++) {
		for(j = 0; j < NY; j++) {
			if(percentDiff(hz1[i * NY + j], hz2[i * NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) fail++;
		}
	}

	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz) {
	int i, j;

	for(i = 0; i < TMAX; i++) {
		_fict_[i] = (DATA_TYPE)i;
	}

	for(i = 0; i < NX; i++) {
		for(j = 0; j < NY; j++) {
			ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
			ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
			hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
		}
	}
}

void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz) {
	int t, i, j;

	for(t = 0; t < TMAX; t++) {
		for(j = 0; j < NY; j++) {
			ey[0 * NY + j] = _fict_[t];
		}

		for(i = 1; i < NX; i++) {
			for(j = 0; j < NY; j++) {
				ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
			}
		}

		for(i = 0; i < NX; i++) {
			for(j = 1; j < NY; j++) {
				ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
			}
		}

		for(i = 0; i < NX; i++) {
			for(j = 0; j < NY; j++) {
				hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
			}
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc >= 2) {
		const auto problem_size = std::atoi(argv[1]);
		NX = problem_size;
		NY = problem_size;
	}
	std::cout << "Problem size: " << NX << "   (" << TMAX << " time steps)\n";

	std::vector<DATA_TYPE> fict(TMAX);
	std::vector<DATA_TYPE> ex(NX * (NY + 1));
	std::vector<DATA_TYPE> ey((NX + 1) * NY);
	std::vector<DATA_TYPE> hz(NX * NY);

	init_arrays(fict.data(), ex.data(), ey.data(), hz.data());

	if(shouldDoCpu()) {
		double t_start = rtclock();
		runFdtd(fict.data(), ex.data(), ey.data(), hz.data());
		double t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	}

	{
		using namespace cl::sycl;

		std::vector<DATA_TYPE> fict_gpu(TMAX);
		std::vector<DATA_TYPE> ex_gpu(NX * (NY + 1));
		std::vector<DATA_TYPE> ey_gpu((NX + 1) * NY);
		std::vector<DATA_TYPE> hz_gpu(NX * NY);

		init_arrays(fict_gpu.data(), ex_gpu.data(), ey_gpu.data(), hz_gpu.data());

		cl::sycl::queue queue;

		buffer<DATA_TYPE, 2> fict_buffer{range<2>(TMAX, 1)};
		initDeviceBuffer(queue, fict_buffer, fict_gpu.data());

		buffer<DATA_TYPE, 2> ex_buffer{range<2>(NX, NY + 1)};
		initDeviceBuffer(queue, ex_buffer, ex_gpu.data());

		buffer<DATA_TYPE, 2> ey_buffer{range<2>(NX + 1, NY)};
		initDeviceBuffer(queue, ey_buffer, ey_gpu.data());

		buffer<DATA_TYPE, 2> hz_buffer{range<2>(NX, NY)};
		initDeviceBuffer(queue, hz_buffer, hz_gpu.data());

		double t_start = rtclock();

		for(size_t t = 0; t < TMAX; t++) {
			queue.submit([&](handler& cgh) {
				auto fict = fict_buffer.get_access<access::mode::read>(cgh);
				auto hz = hz_buffer.get_access<access::mode::read>(cgh);
				auto ey = ey_buffer.get_access<access::mode::read_write>(cgh);

				const auto pfor_range = nd_range<2>(range<2>(NX, NY), {DIM_Y, DIM_X});

				cgh.parallel_for<class Fdtd2d1>(pfor_range, [=](nd_item<2> nd_item) {
					const auto item = nd_item.get_global_id();
					const auto i = item[0];
					const auto j = item[1];

					if(i == 0) {
						ey[item] = fict[{t, 0}];
					} else {
						ey[item] = ey[item] - 0.5f * (hz[item] - hz[{(i - 1), j}]);
					}
				});
			});

			queue.submit([&](handler& cgh) {
				auto ex = ex_buffer.get_access<access::mode::read_write>(cgh);
				auto hz = hz_buffer.get_access<access::mode::read>(cgh);

				const auto pfor_range = nd_range<2>(range<2>(NX, NY), {DIM_Y, DIM_X});

				cgh.parallel_for<class Fdtd2d2>(pfor_range, [=, NX_ = NX, NY_ = NY](nd_item<2> nd_item) {
					const auto item = nd_item.get_global_id();
					const auto i = item[0];
					const auto j = item[1];

					if(j <= 0) return;

					ex[item] = ex[item] - 0.5 * (hz[item] - hz[{i, (j - 1)}]);
				});
			});

			queue.submit([&](handler& cgh) {
				auto ex = ex_buffer.get_access<access::mode::read>(cgh);
				auto ey = ey_buffer.get_access<access::mode::read>(cgh);
				auto hz = hz_buffer.get_access<access::mode::read_write>(cgh);

				const auto pfor_range = nd_range<2>(hz_buffer.get_range(), {DIM_Y, DIM_X});

				cgh.parallel_for<class Fdtd2d3>(pfor_range, [=](nd_item<2> nd_item) {
					const auto item = nd_item.get_global_id();
					const auto i = item[0];
					const auto j = item[1];

					hz[item] = hz[item] - 0.7 * (ex[{i, (j + 1)}] - ex[item] + ey[{(i + 1), j}] - ey[item]);
				});
			});
		}

		queue.wait();
		double t_end = rtclock();

		auto out = hz_buffer.get_access<access::mode::read>(hz_buffer.get_range());
		fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
		if(shouldDoCpu()) compareResults(hz.data(), out.get_pointer());
	}
}
