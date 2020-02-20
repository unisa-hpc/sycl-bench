#ifndef RESULT_CONSUMER_HPP
#define RESULT_CONSUMER_HPP

#include <cassert>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

class ResultConsumer
{
public:
  virtual void proceedToBenchmark(const std::string& name) = 0;
  // Register a result in the result consumer
  virtual void consumeResult(const std::string& result_name,
                            const std::string& result,
                            const std::string& unit = "") = 0;

  // Guarantees that the results have been emitted to the output
  // as specified by the ResultConsumer implementation
  virtual void flush() = 0;

  // Discards the current benchmark's results, useful e.g. in case of errors.
  virtual void discard() {}

  virtual ~ResultConsumer(){}
  
};

class OstreamResultConsumer : public ResultConsumer
{
  std::ostream& output;
  std::string name;

public:
  OstreamResultConsumer(std::ostream& ostr)
  : output{ostr}
  {}

  virtual void proceedToBenchmark(const std::string& benchmark_name) override
  {
    name = benchmark_name;
    output << "********** Results for " << name 
           << "**********" << std::endl;
    
  }

  virtual void consumeResult(const std::string& result_name,
                            const std::string& result,
                            const std::string& unit = "") override
  {
    output << result_name << ": " << result;
    if(!unit.empty()) {
      output << " [" << unit << "]";
    }
    output << std::endl;
  }

  virtual void flush() override
  {
  }
};

// TODO ResultConsumer that appends to a csv
class AppendingCsvResultConsumer : public ResultConsumer
{
public:
  using benchmark_data = std::unordered_map<std::string, std::string>;

  AppendingCsvResultConsumer(const std::string& filename)
  : output{filename, std::ios::app}
  {}

  virtual void proceedToBenchmark(const std::string& benchmark_name) override
  {
    currentBenchmark = benchmark_name;
  }

  virtual void consumeResult(const std::string& result_name,
                            const std::string& result,
                            const std::string& unit = "") override
  {
    data[currentBenchmark][result_name] = result;
  }

  virtual void flush() override
  {
    std::unordered_set<std::string> columns;

    for(const auto& benchmark: data) {
      for(auto entry : benchmark.second) {
        columns.insert(entry.first);
      }
    }

    std::vector<std::string> sorted_columns;
    for(auto c : columns)
      sorted_columns.push_back(c);
    // To make sure order of columns is deterministic
    std::sort(sorted_columns.begin(),sorted_columns.end());

    output << "# Benchmark name";
    for(auto c : sorted_columns)
      output << "," << c;
    output << std::endl;

    for(const auto& benchmark : data) {
      output << benchmark.first;
      for(auto c : sorted_columns)
        output << "," << benchmark.second.at(c);
      output << std::endl;
    }

    data.clear();
    
  }

  void discard() override {
    assert(!currentBenchmark.empty());
    data.erase(currentBenchmark);
    currentBenchmark.clear();
  }

private:
  std::string currentBenchmark;

  std::unordered_map<std::string, benchmark_data> data;

  std::ofstream output;
};

#endif

