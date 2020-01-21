#ifndef SYCL_UTIL_FUNCTS_H
#define SYCL_UTIL_FUNCTS_H

#include <CL/sycl.hpp>

template <typename T, int Dims>
void initDeviceBuffer(cl::sycl::queue& queue, cl::sycl::buffer<T, Dims>& buffer, T* data) {
	using namespace cl::sycl;

	queue.submit([&](handler& cgh) {
		auto accessor = buffer.template get_access<access::mode::discard_write>(cgh);
		cgh.copy(data, accessor);
	});

	queue.wait();
}

#endif
