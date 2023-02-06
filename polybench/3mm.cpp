#include <string>
#include <vector>

#include <cstdlib>

#include <sycl/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Polybench_3mm_1;
class Polybench_3mm_2;
class Polybench_3mm_3;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;
	const auto NM = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NK; j++) {
			A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(size_t i = 0; i < NK; i++) {
		for(size_t j = 0; j < NJ; j++) {
			B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
		}
	}

	for(size_t i = 0; i < NJ; i++) {
		for(size_t j = 0; j < NM; j++) {
			C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
		}
	}

	for(size_t i = 0; i < NM; i++) {
		for(size_t j = 0; j < NL; j++) {
			D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
		}
	}
}

void mm3_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;
	const auto NM = size;

	/* E := A*B */
	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			E[i * NJ + j] = 0;
			for(size_t k = 0; k < NK; ++k) {
				E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			}
		}
	}

	/* F := C*D */
	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			F[i * NL + j] = 0;
			for(size_t k = 0; k < NM; ++k) {
				F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
			}
		}
	}

	/* G := E*F */
	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			G[i * NL + j] = 0;
			for(size_t k = 0; k < NJ; ++k) {
				G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
			}
		}
	}
}

class Polybench_3mm {
  public:
	Polybench_3mm(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);
		D.resize(size * size);
		E.resize(size * size);
		F.resize(size * size);
		G.resize(size * size);

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), sycl::range<2>(size, size));
		B_buffer.initialize(args.device_queue, B.data(), sycl::range<2>(size, size));
		C_buffer.initialize(args.device_queue, C.data(), sycl::range<2>(size, size));
		D_buffer.initialize(args.device_queue, D.data(), sycl::range<2>(size, size));
		E_buffer.initialize(args.device_queue, E.data(), sycl::range<2>(size, size));
		F_buffer.initialize(args.device_queue, F.data(), sycl::range<2>(size, size));
		G_buffer.initialize(args.device_queue, G.data(), sycl::range<2>(size, size));
	}

	void run(std::vector<sycl::event>& events) {
		using namespace sycl;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto E = E_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Polybench_3mm_1>(E_buffer.get_range(), [=, size_ = size](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				for(size_t k = 0; k < size_; k++) {
					E[item] += A[{i, k}] * B[{k, j}];
				}
			});
		}));

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto C = C_buffer.get_access<access::mode::read>(cgh);
			auto D = D_buffer.get_access<access::mode::read>(cgh);
			auto F = F_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Polybench_3mm_2>(F_buffer.get_range(), [=, size_ = size](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				for(size_t k = 0; k < size_; k++) {
					F[item] += C[{i, k}] * D[{k, j}];
				}
			});
		}));

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto E = E_buffer.get_access<access::mode::read>(cgh);
			auto F = F_buffer.get_access<access::mode::read>(cgh);
			auto G = G_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Polybench_3mm_3>(F_buffer.get_range(), [=, size_ = size](item<2> item) {
				const auto i = item[0];
				const auto j = item[1];

				for(size_t k = 0; k < size_; k++) {
					G[item] += E[{i, k}] * F[{k, j}];
				}
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		std::vector<DATA_TYPE> E_cpu(size * size);
		std::vector<DATA_TYPE> F_cpu(size * size);
		std::vector<DATA_TYPE> G_cpu(size * size);

		mm3_cpu(A.data(), B.data(), C.data(), D.data(), E_cpu.data(), F_cpu.data(), G_cpu.data(), size);

		auto G_acc = G_buffer.get_access<sycl::access::mode::read>();

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(G_cpu[i * size + j], G_acc.get_pointer()[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName() { return "Polybench_3mm"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> C;
	std::vector<DATA_TYPE> D;
	std::vector<DATA_TYPE> E;
	std::vector<DATA_TYPE> F;
	std::vector<DATA_TYPE> G;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> D_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> E_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> F_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> G_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_3mm>();
	return 0;
}
