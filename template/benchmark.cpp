#include "common.h"


// Run all benchmark
int main(int argc, char** argv)
{
  BenchmarkApp app{argc, argv};
  app.run<MyBenchmark>();
}
