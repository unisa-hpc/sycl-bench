#ifndef RESULT_CONSUMER_HPP
#define RESULT_CONSUMER_HPP


class ResultConsumer
{
public:
  virtual void proceedToBenchmark(const std::string& name) = 0;
  // Register a result in the result consumer
  virtual void consumeResult(const std::string& result_name,
                            const std::string& result,
                            const std::string& comment = std::string{}) = 0;

  // Guarantees that the results have been emitted to the output
  // as specified by the ResultConsumer implementation
  virtual void flush() = 0;

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
                            const std::string& comment = std::string{}) override
  {
    output << result_name << ": " << result;
    if(comment.length() > 0)
      output << " Note: " << comment;
    output << std::endl;
  }

  virtual void flush() override
  {
  }
};

// TODO ResultConsumer that appends to a csv
class AppendingCsvResultConsumer
{
};

#endif

