#pragma once

#include <chrono>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "benchmark_traits.h"
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
class TimeMetricsProcessor {
public:
  TimeMetricsProcessor(const BenchmarkArgs& args) : args(args) {}

  void addTimingResult(const std::string& name, std::chrono::nanoseconds time) {
    if(unavailableTimings.count(name) != 0) {
      throw std::invalid_argument{"Cannot add result for unavailable timing " + name};
    }
    timingResults[name].push_back(time);
  }

  /**
   * This is a bit of a hack that we need right now to ensure that all emitted results include the same
   * CSV columns, even if a timing is not available for a particular benchmark and/or SYCL implementation.
   *
   * TODO: Come up with a better solution
   */
  void markAsUnavailable(const std::string& name) {
    if(timingResults.count(name) != 0) {
      throw std::invalid_argument{"Cannot mark timing " + name + " with existing results as unavailable"};
    }
    unavailableTimings.insert(name);
  }

  void emitResults(ResultConsumer& consumer) const {
    // Begin by outputting the throughput metric (if available), as this does not depend on a timing.
    if constexpr(detail::BenchmarkTraits<Benchmark>::hasGetThroughputMetric) {
      const auto tpm = Benchmark::getThroughputMetric(args);
      consumer.consumeResult("throughput-metric", std::to_string(tpm.metric), tpm.unit);
    } else {
      consumer.consumeResult("throughput-metric", "N/A", "");
    }

    // We have to ensure that available and unavailable timings are always being emitted in the same order.
    // To this end, we copy all timing names into a sorted container and iterate over it afterwards.
    std::set<std::string> allTimings;
    for(const auto& name : unavailableTimings) {
      allTimings.insert(name);
    }
    for(const auto& [name, results] : timingResults) {
      allTimings.insert(name);
    }

    for(const auto& name : allTimings) {
      if(unavailableTimings.count(name) == 0) {
        std::vector<double> resultsSeconds;
        std::transform(timingResults.at(name).begin(), timingResults.at(name).end(), std::back_inserter(resultsSeconds),
            [](auto r) { return r.count() / 1.0e9; });
        std::sort(resultsSeconds.begin(), resultsSeconds.end());

        double mean = std::accumulate(resultsSeconds.begin(), resultsSeconds.end(), 0.0) /
                      static_cast<double>(resultsSeconds.size());

        double stddev = 0.0;
        for(double x : resultsSeconds) {
          double dev = mean - x;
          stddev += dev * dev;
        }
        if(resultsSeconds.size() <= 1) {
          stddev = 0.0;
        } else {
          stddev /= static_cast<double>(resultsSeconds.size() - 1);
          stddev = std::sqrt(stddev);
        }

        const double median = resultsSeconds[resultsSeconds.size() / 2];

        consumer.consumeResult(name + "-mean", std::to_string(mean), "s");
        consumer.consumeResult(name + "-stddev", std::to_string(stddev), "s");
        consumer.consumeResult(name + "-median", std::to_string(median), "s");
        consumer.consumeResult(name + "-min", std::to_string(resultsSeconds[0]), "s");

        // Emit individual samples as well
        std::stringstream samples;
        samples << "\"";
        for(int i = 0; i < resultsSeconds.size(); ++i) {
          samples << std::to_string(resultsSeconds[i]);
          if(i != resultsSeconds.size() - 1) {
            samples << " ";
          }
        }
        samples << "\"";
        consumer.consumeResult(name + "-samples", samples.str());

        double throughputMetric = 0.0;
        double throughput = 0.0;
        std::string unit = "";
        if constexpr(detail::BenchmarkTraits<Benchmark>::hasGetThroughputMetric) {
          const double min = resultsSeconds[0];
          const auto tpm = Benchmark::getThroughputMetric(args);
          throughputMetric = tpm.metric;
          throughput = throughputMetric / min;
          unit = tpm.unit;
        }
        if(throughputMetric > 0.0) {
          consumer.consumeResult(name + "-throughput", std::to_string(throughput), unit + "/s");
        } else {
          consumer.consumeResult(name + "-throughput", "N/A", "");
        }
      } else {
        // Now the hacky part: Emit columns also for unavailable timings.
        // FIXME: Come up with a cleaner solution.
        consumer.consumeResult(name + "-mean", "N/A");
        consumer.consumeResult(name + "-stddev", "N/A");
        consumer.consumeResult(name + "-median", "N/A");
        consumer.consumeResult(name + "-min", "N/A");
        consumer.consumeResult(name + "-samples", "N/A");
        consumer.consumeResult(name + "-throughput", "N/A");
      }
    }
  }

private:
  const BenchmarkArgs args;
  std::unordered_map<std::string, std::vector<std::chrono::nanoseconds>> timingResults;
  std::unordered_set<std::string> unavailableTimings;
};
