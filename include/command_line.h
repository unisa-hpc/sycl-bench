#ifndef BENCHMARK_COMMAND_LINE_HPP
#define BENCHMARK_COMMAND_LINE_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <CL/sycl.hpp>

using CommandLineArguments = std::unordered_map<std::string, std::string>;
using FlagList = std::unordered_set<std::string>;

class CommandLine
{
public:
  CommandLine(int argc, char** argv)
  {
    for (int i = 0; i < argc; ++i)
    {
      std::string arg = argv[i];
      auto pos = arg.find("=");
      if(pos != std::string::npos)
      {
        auto argName = arg.substr(0,pos);
        auto argVal = arg.substr(pos+1);

        if(args.find(argName) != args.end())
        {
          throw std::invalid_argument{
              "Encountered command line argument several times: " + argName};
        }

        args[argName] = argVal;
      }
      else
      {
        flags.insert(arg);
      }
    }
  }

  bool isArgSet(const std::string& arg) const
  {
    return args.find(arg) != args.end();
  }

  template<class T>
  T getOrDefault(const std::string& arg, const T& defaultVal) const
  {
    if(isArgSet(arg))
      return cast<T>(args.at(arg));
    return defaultVal;
  }

  template<class T>
  T get(const std::string& arg) const
  {
    try
    {
      return cast<T>(args.at(arg));
    }
    catch(std::out_of_range& e)
    {
      throw std::invalid_argument{"Command line argument was requested but missing: "+arg};
    }
  }

  bool isFlagSet(const std::string& flag) const
  {
    return flags.find(flag) != flags.end();
  }

  template<class T>
  static T cast(const std::string& s)
  {
    std::stringstream sstr{s};
    T result;
    sstr >> result;
    return result;
  }

  template<>
  static cl::sycl::range<3>
  cast(const std::string& s)
  {
    return parseSyclArray<cl::sycl::range<3>>(s, 1);
  }

  template<>
  static cl::sycl::id<3>
  cast(const std::string& s)
  {
    return parseSyclArray<cl::sycl::id<3>>(s, 0);
  }

private:
  template<class T>
  static std::vector<T> parseCommaDelimitedList(const std::string& s)
  {
    std::stringstream istr(s);
    std::string current;
    std::vector<T> result;

    while(std::getline(istr, current, ','))
      result.push_back(CommandLine::cast<T>(current));
    
    return result;
  }

  template<class SyclArraylike>
  static SyclArraylike parseSyclArray(const std::string& s, std::size_t defaultValue)
  {
    auto elements = parseCommaDelimitedList<std::size_t>(s);
    if(s.size() > 3)
      throw std::invalid_argument{"Invalid sycl range/id: "+s};
    else if(s.size() == 3)
      return SyclArraylike{elements[0], elements[1], elements[2]};
    else if(s.size() == 2)
      return SyclArraylike{elements[0], elements[1], defaultValue};
    else if(s.size() == 1)
      return SyclArraylike{elements[0], defaultValue, defaultValue};
    else
      throw std::invalid_argument{"Invalid sycl range/id: "+s};
  }

  CommandLineArguments args;
  FlagList flags;
};


struct VerificationSetting
{
  cl::sycl::id<3> begin;
  cl::sycl::range<3> range;
};

struct BenchmarkArgs
{
  size_t problem_size;
  size_t local_size; 
  cl::sycl::queue device_queue;
  VerificationSetting verification;
  // can be used to query additional information
  CommandLine cli;
};

class BenchmarkCommandLine
{
public:
  BenchmarkCommandLine(int argc, char **argv) 
  : cli_parser{argc, argv} {}

  BenchmarkArgs getBenchmarkArgs() const
  {
    std::size_t size = cli_parser.getOrDefault<std::size_t>("--size", 1024);
    std::size_t local_size = cli_parser.getOrDefault<std::size_t>("--local", 256);

    std::string device_selector = 
        cli_parser.getOrDefault<std::string>("--device", "default");
    cl::sycl::queue q = getQueue(device_selector);

    auto verification_begin = cli_parser.getOrDefault<cl::sycl::id<3>>(
      "--verification-begin", cl::sycl::id<3>{0,0,0});
    
    auto verification_range = cli_parser.getOrDefault<cl::sycl::range<3>>(
      "--verification-range", cl::sycl::range<3>{0,0,0});

    return BenchmarkArgs{
      size, local_size, q, 
      VerificationSetting{verification_begin, verification_range}, 
      cli_parser
    };
  }

private:
  

  cl::sycl::queue getQueue(const std::string& device_selector) const
  {
    if (device_selector == "cpu") {
      cl::sycl::cpu_selector selector;
      return selector.select_device();
    }
    else if(device_selector == "gpu") {
      cl::sycl::gpu_selector selector;
      return selector.select_device();
    }
    else if(device_selector == "default") {
      return cl::sycl::queue{};
    }
    else throw std::invalid_argument{
      "unknown device selector: "+device_selector};
  }

  CommandLine cli_parser;
};

#endif

