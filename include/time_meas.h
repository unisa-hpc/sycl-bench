#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "benchmark_hook.h"
#include "benchmark_traits.h"
#include "command_line.h"
#include "result_consumer.h"

/**
 * Throughput metrics can be returned by benchmarks that implement the
 * getThroughputMetric() function. The returned value (and associated unit)
 * represents the metric underlying the throughput calculation associated with
 * a benchmark.
 *
 * Note that the metric is NOT the throughput. For example, a returned metric
 * for arithmetric throughput could be the total number of floating-point operations,
 * FLOP, not FLOP/s.
 */
struct ThroughputMetric {
  double metric = 0.0;
  std::string unit = "";
};


template <typename Benchmark>
class TimeMeasurement : public BenchmarkHook {
  using Clock = std::chrono::high_resolution_clock;

public:
  TimeMeasurement(const BenchmarkArgs& args) : BenchmarkHook(), args(args) {}

  virtual void atInit() override {}
  virtual void preSetup() override {}
  virtual void postSetup() override {}

  virtual void preKernel() override { t1 = Clock::now(); }

  virtual void postKernel() override {
    t2 = Clock::now();
    seconds.push_back(
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1.e-9);
  }

  virtual void emitResults(ResultConsumer& consumer) override {
    std::string full_result_string = "\"";
    for(auto s : seconds) {
      full_result_string += std::to_string(s);
      full_result_string += " ";
    }
    full_result_string += "\"";

    double mean_sec = std::accumulate(seconds.begin(), seconds.end(), 0.) / static_cast<double>(seconds.size());

    double stddev = 0.0;
    for(double x : seconds) {
      double dev = mean_sec - x;
      stddev += dev * dev;
    }
    if(seconds.size() <= 1)
      stddev = 0.0;
    else {
      stddev /= static_cast<double>(seconds.size() - 1);
      stddev = std::sqrt(stddev);
    }

    std::sort(seconds.begin(), seconds.end());
    const double median_sec = seconds[seconds.size() / 2];

    consumer.consumeResult("run-time", std::to_string(mean_sec), "s");
    consumer.consumeResult("run-time-stddev", std::to_string(stddev), "s");
    consumer.consumeResult("run-time-median", std::to_string(median_sec), "s");
    consumer.consumeResult("run-time-min", std::to_string(seconds[0]), "s");
    consumer.consumeResult("run-time-samples", full_result_string);

    double throughputMetric = 0.0;
    double throughput = 0.0;
    std::string unit = "";
    if constexpr(detail::BenchmarkTraits<Benchmark>::hasGetThroughputMetric) {
      const double min_sec = seconds[0];
      const auto tpm = Benchmark::getThroughputMetric(args);
      throughputMetric = tpm.metric;
      throughput = throughputMetric / min_sec;
      unit = tpm.unit;
    }
    if(throughputMetric > 0.0) {
      consumer.consumeResult("throughput-metric", std::to_string(throughputMetric), unit);
      consumer.consumeResult("throughput", std::to_string(throughput), unit + "/s");
    } else {
      consumer.consumeResult("throughput-metric", "N/A", "");
      consumer.consumeResult("throughput", "N/A", "");
    }
  }

private:
  const BenchmarkArgs args;

  std::chrono::time_point<std::chrono::high_resolution_clock> t1;
  std::chrono::time_point<std::chrono::high_resolution_clock> t2;

  std::vector<double> seconds;
};
