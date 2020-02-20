#include "common.h"
#include <iostream>

#ifndef FLT_MAX
#define FLT_MAX 500000.0
#endif

//using namespace cl::sycl;
namespace s = cl::sycl;
template <typename T> class KmeansKernel;

template <typename T>
class KmeansBench
{
protected:    
    std::vector<T> features;
    std::vector<T> clusters;
    std::vector<int> membership;
    int nfeatures;
	  int nclusters;
    int feature_size;
    int cluster_size;
    BenchmarkArgs args;


    PrefetchedBuffer<T, 1> features_buf;
    PrefetchedBuffer<T, 1> clusters_buf;
    PrefetchedBuffer<int, 1> membership_buf;
public:
  KmeansBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    nfeatures = 2;
    nclusters = 3;

    feature_size = nfeatures*args.problem_size;
    cluster_size = nclusters*args.problem_size;

    features.resize(feature_size, 2.0f);
    clusters.resize(cluster_size, 1.0f);
    membership.resize(args.problem_size, 0);

    features_buf.initialize(args.device_queue, features.data(), s::range<1>(feature_size));
    clusters_buf.initialize(args.device_queue, clusters.data(), s::range<1>(cluster_size));
    membership_buf.initialize(args.device_queue, membership.data(), s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto features = features_buf.template get_access<s::access::mode::read>(cgh);
      auto clusters = clusters_buf.template get_access<s::access::mode::read>(cgh);
      auto membership = membership_buf.template get_access<s::access::mode::discard_write>(cgh);

      cl::sycl::range<1> ndrange(args.problem_size);

      cgh.parallel_for<class KmeansKernel<T>>(ndrange,
        [features, clusters, membership, problem_size = args.problem_size,
         nclusters_ = nclusters, nfeatures_ = nfeatures]
        (cl::sycl::id<1> idx){

        size_t gid = idx[0];

        if(gid < problem_size) {
          int index = 0;
          T min_dist = FLT_MAX;
          for(size_t i = 0; i < nclusters_; i++) {
            T dist = 0;
            for(size_t l = 0; l < nfeatures_; l++) {
              dist += (features[l * problem_size + gid] - clusters[i * nfeatures_ + l]) *
                      (features[l * problem_size + gid] - clusters[i * nfeatures_ + l]);
            }
            if(dist < min_dist) {
              min_dist = dist;
              index = gid;
            }
          }
          membership[gid] = index;
        }
      });
    }));
  }

  bool verify(VerificationSetting &ver) {
    auto membership_acc = membership_buf.template get_access<s::access::mode::read>();

    bool pass = true;
    unsigned int equal = 1;

    for(size_t x = 0; x < args.problem_size; ++x) {
      int index = 0;
      T min_dist = 500000.0f;
      for(size_t i = 0; i < nclusters; i++) {
        T dist = 0;
        for(size_t l = 0; l < nfeatures; l++) {
          dist += (features[l * args.problem_size + x] - clusters[i * nfeatures + l]) *
                  (features[l * args.problem_size + x] - clusters[i * nfeatures + l]);
        }
        if(dist < min_dist) {
          min_dist = dist;
          index = x;
        }
      }
      if(membership_acc[x] != index) {
        equal = 0;
        std::cout << "Fail at = " << x << "Expected = " << index << "Actual =" << membership[x] << std::endl;
        break;
      }
    }

    if(!equal) {
      pass = false;
    }
    return pass;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "Kmeans_";
    name << ReadableTypename<T>::name;
    return name.str();     
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<KmeansBench<float>> ();
  app.run<KmeansBench<double>> ();   
  return 0;
}
