#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Polybench_2mm_2;
class Polybench_2mm_1;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NK; j++) {
			A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(size_t i = 0; i < NK; i++) {
		for(size_t j = 0; j < NJ; j++) {
			B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
		}
	}

	for(size_t i = 0; i < NL; i++) {
		for(size_t j = 0; j < NJ; j++) {
			C[i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
		}
	}

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
		}
	}
}

void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			for(size_t k = 0; k < NK; ++k) {
				C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			}
		}
	}

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			E[i * NL + j] = 0;
			for(size_t k = 0; k < NJ; ++k) {
				E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
			}
		}
	}
}

class Polybench_2mm {
  public:
	Polybench_2mm(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);
		D.resize(size * size);
		E.resize(size * size);

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
		C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
		D_buffer.initialize(args.device_queue, D.data(), cl::sycl::range<2>(size, size));
		E_buffer.initialize(args.device_queue, E.data(), cl::sycl::range<2>(size, size));
	}

	void run() {
		using namespace cl::sycl;

		args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Polybench_2mm_1>(C_buffer.get_range(), [=, size_ = size](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				for(size_t k = 0; k < size_; k++) {
					C[item] += A[{i, k}] * B[{k, j}];
				}
			});
		});

		args.device_queue.submit([&](handler& cgh) {
			auto C = C_buffer.get_access<access::mode::read>(cgh);
			auto D = D_buffer.get_access<access::mode::read>(cgh);
			auto E = E_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<Polybench_2mm_2>(E_buffer.get_range(), [=, size_ = size](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				E[item] = 0;
				for(size_t k = 0; k < size_; k++) {
					E[item] += C[{i, k}] * D[{k, j}];
				}
			});
		});
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		std::vector<DATA_TYPE> E_cpu(size * size);
		mm2_cpu(A.data(), B.data(), C.data(), D.data(), E_cpu.data(), size);

		auto E_acc = E_buffer.get_access<cl::sycl::access::mode::read>();

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(E_cpu[i * size + j], E_acc.get_pointer()[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName() { return "Polybench_2mm"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> C;
	std::vector<DATA_TYPE> D;
	std::vector<DATA_TYPE> E;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> D_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> E_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_2mm>();
	return 0;
}
