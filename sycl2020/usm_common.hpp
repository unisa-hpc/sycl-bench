#ifndef USM_COMMON
#define USM_COMMON

#include <sycl/sycl.hpp>

std::string usm_to_string(sycl::usm::alloc usm_type) {
  if(usm_type == sycl::usm::alloc::device)
    return "device";
  else if(usm_type == sycl::usm::alloc::host)
    return "host";
  else if(usm_type == sycl::usm::alloc::shared)
    return "shared";
  else
    return "unknown";
}

#endif