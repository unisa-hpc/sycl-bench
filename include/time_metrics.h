#pragma once

#include <chrono>
#include <cmath>
#include <numeric>
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

  void emitResults(ResultConsumer& consumer) {
    bool throughputMetricEmitted = false; // Only do this once
    for(const auto& [name, results] : timingResults) {
      std::vector<double> resultsSeconds;
      std::transform(
          results.begin(), results.end(), std::back_inserter(resultsSeconds), [](auto r) { return r.count() / 1.0e9; });
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
        if(!throughputMetricEmitted) {
          consumer.consumeResult("throughput-metric", std::to_string(throughputMetric), unit);
        }
        consumer.consumeResult(name + "-throughput", std::to_string(throughput), unit + "/s");
      } else {
        if(!throughputMetricEmitted) {
          consumer.consumeResult("throughput-metric", "N/A", "");
        }
        consumer.consumeResult(name + "-throughput", "N/A", "");
      }
      throughputMetricEmitted = true;
    }

    // Now the hacky part: Emit columns again for all unavailable timings.
    // FIXME: Come up with a cleaner solution.
    for(auto& name : unavailableTimings) {
      consumer.consumeResult(name + "-mean", "N/A");
      consumer.consumeResult(name + "-stddev", "N/A");
      consumer.consumeResult(name + "-median", "N/A");
      consumer.consumeResult(name + "-min", "N/A");
      consumer.consumeResult(name + "-samples", "N/A");
      consumer.consumeResult(name + "-throughput", "N/A");
    }
  }

private:
  const BenchmarkArgs args;
  std::unordered_map<std::string, std::vector<std::chrono::nanoseconds>> timingResults;
  std::unordered_set<std::string> unavailableTimings;
};
