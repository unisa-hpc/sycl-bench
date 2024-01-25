#pragma once
#include <sycl/sycl.hpp>

std::string usm_to_string(sycl::usm::alloc usm_type) {
  if(usm_type == sycl::usm::alloc::device) {
    return "device";
  }
  if(usm_type == sycl::usm::alloc::host) {
    return "host";
  }
  if(usm_type == sycl::usm::alloc::shared) {
    return "shared";
  }
  return "unknown";
}