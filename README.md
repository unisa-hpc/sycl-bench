# SYCL-Bench
SYCL Benchmark Suite, work in progress

Benchmarks support the following command line arguments:
* `--size=<problem-size>` - total problem size. For most benchmarks, global range of work items. Default: 3072
* `--local=<local-size>` - local size/work group size, if applicable. Not all benchmarks use this. Default: 256
* `--num-runs=<N>` - the number of times that the problem should be run, e.g. for averaging runtimes. Default: 5
* `--device=<d>` - changes the SYCL device selector that is used. Supported values: `cpu`, `gpu`, `default`. Default: `default`
* `--output=<output>` - Specify where to store the output and how to format. If `<output>=stdio`, results are printed to standard output. For any other value, `<output>` is interpreted as a file where the output will be saved in csv format.
* `--verification-begin=<x,y,z>` - Specify the start of the 3D range that should be used for verifying results. Note: Most benchmarks do not implement this feature. Default: `0,0,0`
* `--verification-range=<x,y,z>` - Specify the size of the 3D range that should be used for verifying results. Note: Most benchmarks do not implement this feature. Default: `1,1,1`

