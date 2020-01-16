include_directories(common)

set(exes
  2DConvolution
  2mm
  3DConvolution
  3mm
  atax
  bicg
  correlation
  covariance
  fdtd2d
  gemm
  gesummv
  gramschmidt
  mvt
  syr2k
  syrk
)

foreach(exe IN LISTS exes)
  add_executable(${exe} ${exe}.cpp)
  add_sycl_to_target(TARGET ${exe} SOURCES ${exe}.cpp)
endforeach(exe)
