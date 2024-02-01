#include "common.h"

namespace s = sycl;

// The data type to be copied. This was originally a single byte (char), however
// this causes device-side initialization kernels to quickly reach the
// platform limit for work items and or groups in some dimensions.
// By using a larger base type here, we can thus decrease the number of work items.
// Note that this has no effect on the actual benchmark performance.
using DataT = int64_t;

template <int Dims, bool Strided>
constexpr s::range<Dims> getBufferSize(size_t problemSize) {
  if constexpr(Dims == 1) {
    return s::range<1>(problemSize * problemSize * problemSize / sizeof(DataT)) +
           (Strided ? s::range<1>(8) : s::range<1>(0));
  }
  if constexpr(Dims == 2) {
    return s::range<2>(problemSize * problemSize / sizeof(DataT), problemSize) +
           (Strided ? s::range<2>(8, 16) : s::range<2>(0, 0));
  }
  if constexpr(Dims == 3) {
    return s::range<3>(problemSize / sizeof(DataT), problemSize, problemSize) +
           (Strided ? s::range<3>(8, 16, 32) : s::range<3>(0, 0, 0));
  }
}

template <int Dims, bool Strided>
constexpr s::id<Dims> getStridedCopyOffset() {
  if constexpr(Dims == 1) {
    return Strided ? s::id<1>(4) : s::id<1>(0);
  }
  if constexpr(Dims == 2) {
    return Strided ? s::id<2>(4, 8) : s::id<2>(0, 0);
  }
  if constexpr(Dims == 3) {
    return Strided ? s::id<3>(4, 8, 16) : s::id<3>(0, 0, 0);
  }
}

enum class CopyDirection { HOST_TO_DEVICE, DEVICE_TO_HOST };

template <int Dims, bool Strided>
class D2HInitKernel;

template <int Dims>
class H2DCopyKernel;

template <int Dims, bool Strided>
class H2DVerificationKernel;

/**
 * Microbenchmark measuring host<->device bandwidth for contiguous and strided copies.
 *
 * For non-strided copies we use a dummy kernel, as explicit copy operations are not
 * fully supported by some SYCL implementations.
 *
 * Strided copies use a larger SYCL buffer and copy a portion out of the middle.
 * For example, a (512, 512) element 2D-copy at offset (1, 1) out of a (514, 514)
 * element SYCL buffer. The host buffer is never strided (as this is not supported
 * by SYCL 1.2.1).
 *
 * To avoid SYCL implementations to just copy the entire buffer when using a strided accessor,
 * we use explicit copy operations for strided copies.
 */
template <int Dims, CopyDirection Direction, bool Strided>
class MicroBenchHostDeviceBandwidth {
protected:
  BenchmarkArgs args;
  // The buffer size to be copied. This is independent of stride.
  const s::range<Dims> copy_size;
  // The host buffer used as source or target for (some) copy operations.
  // This is always contiguous and has size "copy_size".
  std::vector<DataT> host_data;
  // The strided buffer size is either the same as "copy_size" (if we are not doing strided copies),
  // or includes a border in every dimension. This size is used for the SYCL buffer.
  const s::range<Dims> strided_buffer_size;
  std::unique_ptr<s::buffer<DataT, Dims>> buffer;

  static constexpr DataT TEST_VALUE = 33;

public:
  MicroBenchHostDeviceBandwidth(const BenchmarkArgs& args)
      : args(args), copy_size(getBufferSize<Dims, false>(args.problem_size)),
        strided_buffer_size(getBufferSize<Dims, Strided>(args.problem_size)) {}

  void setup() {
    if constexpr(!Strided) {
      if constexpr(Direction == CopyDirection::HOST_TO_DEVICE) {
        host_data.resize(copy_size.size(), TEST_VALUE);
        buffer = std::make_unique<s::buffer<DataT, Dims>>(host_data.data(), copy_size);
      }

      if constexpr(Direction == CopyDirection::DEVICE_TO_HOST) {
        // NOTE: We still provide a host pointer here, as it can make a substantial difference in performance.
        // This was observed using hipSYCL (CUDA) as well as ComputeCpp (PTX) on Turing.
        // The exact reasons are unclear at this point (potentially related to memory pinning).
        host_data.resize(copy_size.size());
        buffer = std::make_unique<s::buffer<DataT, Dims>>(host_data.data(), copy_size);
        // Initialize buffer on device
        args.device_queue.submit([&](s::handler& cgh) {
          auto acc = buffer->template get_access<s::access::mode::discard_write>(cgh);
          cgh.parallel_for<D2HInitKernel<Dims, Strided>>(copy_size, [=](s::id<Dims> gid) { acc[gid] = TEST_VALUE; });
        });
      }
    }

    if constexpr(Strided) {
      if constexpr(Direction == CopyDirection::HOST_TO_DEVICE) {
        host_data.resize(copy_size.size(), TEST_VALUE);
        buffer = std::make_unique<s::buffer<DataT, Dims>>(strided_buffer_size);
      }

      if constexpr(Direction == CopyDirection::DEVICE_TO_HOST) {
        host_data.resize(copy_size.size());
        buffer = std::make_unique<s::buffer<DataT, Dims>>(strided_buffer_size);
        // Initialize buffer on device
        args.device_queue.submit([&](s::handler& cgh) {
          auto acc = buffer->template get_access<s::access::mode::discard_write>(cgh);
          cgh.parallel_for<D2HInitKernel<Dims, Strided>>(copy_size, [=](s::id<Dims> gid) {
            auto offset = getStridedCopyOffset<Dims, true>();
            acc[gid + offset] = TEST_VALUE;
          });
        });
      }
    }
  }

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    const double copiedGiB =
        getBufferSize<Dims, false>(args.problem_size).size() * sizeof(DataT) / 1024.0 / 1024.0 / 1024.0;
    return {copiedGiB, "GiB"};
  }

  void run() {
    if constexpr(!Strided) {
      if constexpr(Direction == CopyDirection::HOST_TO_DEVICE) {
        // Submit NOP-kernel with read access to buffer, forcing it to be copied.
        args.device_queue.submit([&](s::handler& cgh) {
          buffer->template get_access<s::access::mode::read>(cgh);
          cgh.single_task<H2DCopyKernel<Dims>>([=]() { /* NOP */ });
        });
      }

      if constexpr(Direction == CopyDirection::DEVICE_TO_HOST) {
        // Request host accessor for data that has been written on device
        buffer->get_host_access();
      }
    }

    if constexpr(Strided) {
      if constexpr(Direction == CopyDirection::HOST_TO_DEVICE) {
        args.device_queue.submit([&](s::handler& cgh) {
          auto acc = buffer->template get_access<s::access::mode::discard_write>(
              cgh, copy_size, getStridedCopyOffset<Dims, true>());
          cgh.copy(host_data.data(), acc);
        });
      }

      if constexpr(Direction == CopyDirection::DEVICE_TO_HOST) {
        args.device_queue.submit([&](s::handler& cgh) {
          auto acc =
              buffer->template get_access<s::access::mode::read>(cgh, copy_size, getStridedCopyOffset<Dims, true>());
          cgh.copy(acc, host_data.data());
        });
      }
    }
  }

  bool verify(VerificationSetting&) {
    const auto verifyAccessor = [&](auto acc) {
      const auto strideOffset = getStridedCopyOffset<Dims, Strided>();
      for(size_t i = strideOffset[0]; i < copy_size[0]; ++i) {
        for(size_t j = (Dims < 2 ? 0 : strideOffset[1]); j < (Dims < 2 ? 1 : copy_size[1]); ++j) {
          for(size_t k = (Dims < 3 ? 0 : strideOffset[2]); k < (Dims < 3 ? 1 : copy_size[2]); ++k) {
            if constexpr(Dims == 1) {
              if(acc[i] != TEST_VALUE) {
                return false;
              }
            }
            if constexpr(Dims == 2) {
              if(acc[{i, j}] != TEST_VALUE) {
                return false;
              }
            }
            if constexpr(Dims == 3) {
              if(acc[{i, j, k}] != TEST_VALUE) {
                return false;
              }
            }
          }
        }
      }
      return true;
    };

    if constexpr(Direction == CopyDirection::HOST_TO_DEVICE) {
      // In this case we have to make SYCL think that the buffer has been updated on the device first,
      // as it might otherwise return the original host pointer.
      args.device_queue.submit([&](s::handler& cgh) {
        buffer->template get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<H2DVerificationKernel<Dims, Strided>>([=]() { /* NOP */ });
      });

      auto acc = buffer->get_host_access();
      return verifyAccessor(acc);
    }

    if constexpr(Direction == CopyDirection::DEVICE_TO_HOST) {
      if constexpr(!Strided) {
        auto acc = buffer->get_host_access();
        return verifyAccessor(acc);
      }

      if constexpr(Strided) {
        // Host buffer is always contiguous, so we can just loop in 1D.
        for(size_t i = 0; i < copy_size.size(); ++i) {
          if(host_data[i] != TEST_VALUE) {
            return false;
          }
        }
        return true;
      }
    }

    return false;
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "MicroBench_HostDeviceBandwidth_";
    name << Dims << "D_";
    name << (Direction == CopyDirection::HOST_TO_DEVICE ? "H2D_" : "D2H_");
    name << (Strided ? "Strided" : "Contiguous");
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchHostDeviceBandwidth<1, CopyDirection::HOST_TO_DEVICE, false>>();
  app.run<MicroBenchHostDeviceBandwidth<2, CopyDirection::HOST_TO_DEVICE, false>>();
  app.run<MicroBenchHostDeviceBandwidth<3, CopyDirection::HOST_TO_DEVICE, false>>();

  app.run<MicroBenchHostDeviceBandwidth<1, CopyDirection::DEVICE_TO_HOST, false>>();
  app.run<MicroBenchHostDeviceBandwidth<2, CopyDirection::DEVICE_TO_HOST, false>>();
  app.run<MicroBenchHostDeviceBandwidth<3, CopyDirection::DEVICE_TO_HOST, false>>();

  app.run<MicroBenchHostDeviceBandwidth<1, CopyDirection::HOST_TO_DEVICE, true>>();
  app.run<MicroBenchHostDeviceBandwidth<2, CopyDirection::HOST_TO_DEVICE, true>>();
  app.run<MicroBenchHostDeviceBandwidth<3, CopyDirection::HOST_TO_DEVICE, true>>();

  app.run<MicroBenchHostDeviceBandwidth<1, CopyDirection::DEVICE_TO_HOST, true>>();
  app.run<MicroBenchHostDeviceBandwidth<2, CopyDirection::DEVICE_TO_HOST, true>>();
  app.run<MicroBenchHostDeviceBandwidth<3, CopyDirection::DEVICE_TO_HOST, true>>();

  return 0;
}