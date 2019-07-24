#pragma once 
#include <CL/sycl.hpp>

// These functions are used in sobel and appear
// to be missing and/or nonfunctional in HipSYCL right now

cl::sycl::float4 f4(float v) {
  return {v,v,v,v};
}

namespace cl {
namespace sycl {

float4 operator*(const float4 v, float f) {
    return {v.x()*f, v.y()*f, v.z()*f, v.w()*f};
}

float4 clamp(float4 v, float4 min, float4 max) {
  return {
    v.x() < min.x() ? min.x() : (v.x() > max.x() ? max.x() : v.x()),
    v.y() < min.y() ? min.y() : (v.y() > max.y() ? max.y() : v.y()),
    v.z() < min.z() ? min.z() : (v.z() > max.z() ? max.z() : v.z()),
    v.w() < min.w() ? min.w() : (v.w() > max.w() ? max.w() : v.w()),
  };
}

float length(float4 v) {
  return sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z() + v.w()*v.w());
}

float4 hypot(float4 a, float4 b) {
  return {
      hypot(a.x(), b.x()),
      hypot(a.y(), b.y()),
      hypot(a.z(), b.z()),
      hypot(a.w(), b.w()),
  };
}

float4 fdim(float4 a, float4 b) {
  return {
      fdim(a.x(), b.x()),
      fdim(a.y(), b.y()),
      fdim(a.z(), b.z()),
      fdim(a.w(), b.w()),
  };
}

}
}