#ifndef BENCHMARK_HOOK_HPP
#define BENCHMARK_HOOK_HPP

#include "result_consumer.h"

class BenchmarkHook
{
public:
  virtual void atInit() = 0;
  virtual void preSetup() = 0;
  virtual void postSetup()= 0;
  virtual void preKernel() = 0;
  virtual void postKernel() = 0;
  virtual void emitResults(ResultConsumer&) {}

  virtual ~BenchmarkHook(){}
};

#endif
