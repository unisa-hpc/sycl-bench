#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"

namespace s = sycl;


class KnnKernel; // kernel forward declaration


class Knn {
protected:
  size_t size;
  size_t local_size;
  BenchmarkArgs args;
  int nRef;

  std::vector<float> ref;
  std::vector<float> query;
  std::vector<float> dists;
  std::vector<int> neighbors;


  PrefetchedBuffer<float, 1> buf_ref;
  PrefetchedBuffer<float, 1> buf_query;
  PrefetchedBuffer<float, 1> buf_dists;
  PrefetchedBuffer<int, 1> buf_neighbors;

public:
  Knn(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;     // input size defined by the user
    local_size = args.local_size; // set local work_group size

    nRef = 100000;

    ref.resize(nRef);
    query.resize(size);
    dists.resize(size);
    neighbors.resize(size);
    srand(42);

    for(int i = 0; i < nRef /*dim*/; ++i) {
      ref[i] = rand();
    }
    for(int i = 0; i < size /**dim*/; ++i) {
      query[i] = rand();
    }

    // init buffer
    buf_ref.initialize(args.device_queue, ref.data(), s::range<1>(nRef));
    buf_query.initialize(args.device_queue, query.data(), s::range<1>(size));
    buf_dists.initialize(args.device_queue, dists.data(), s::range<1>(size));
    buf_neighbors.initialize(args.device_queue, neighbors.data(), s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto ref_acc = buf_ref.get_access<s::access::mode::read>(cgh);
      auto query_acc = buf_query.get_access<s::access::mode::read>(cgh);
      auto dist_acc = buf_dists.get_access<s::access::mode::write>(cgh);
      auto neighbours_acc = buf_neighbors.get_access<s::access::mode::write>(cgh);


      s::range<1> ndrange{size};

      cgh.parallel_for<class KnnKernel>(
          ndrange, [this, ref_acc, query_acc, dist_acc, neighbours_acc, numRef = nRef, numQuery = size](s::id<1> id) {
            size_t gid = id[0];

            if(gid >= numQuery)
              return;

            size_t queryOffset = gid /* dim*/;

            size_t curNeighbour = 0;
            float curDist = MAXFLOAT;

            for(int i = 0; i < numRef; ++i) {
              float privateDist = 0;
              size_t refOffset = i /* dim*/;

#if F4
              float4 tmpDist = {0, 0, 0, 0};
              int d;
              for(d = 0; d < dim - 3; d += 4) {
                /* Cypress code */
                float4 a = {ref[refOffset + d], ref[refOffset + d + 1], ref[refOffset + d + 2], ref[refOffset + d + 3]};
                float4 b = {query[queryOffset + d], query[queryOffset + d + 1], query[queryOffset + d + 2],
                    query[queryOffset + d + 3]};
                float4 t = a - b; //(float4){ref[refOffset + d], ref[refOffset + d+1], ref[refOffset + d+2],
                                  //ref[refOffset + d+3]} - (float4){query[queryOffset + d], query[queryOffset + d+1],
                                  //query[queryOffset + d+2], query[queryOffset + d+3]};
                //            a = t * t;
                tmpDist += t * t;
                //            privateDist = a.s0 + a.s1 + a.s2 + a.s3;*/
              }

              for(; d < dim; d++) {
                float t = ref[refOffset + d] - query[queryOffset + d];
                privateDist += t * t;
              }
              privateDist = tmpDist.s0 + tmpDist.s1 + tmpDist.s2 + tmpDist.s3;
#else
        //      for(int d = 0; d < dim; d++)
                {
                    float t = ref_acc[refOffset/* + d*/] - query_acc[queryOffset/* + d*/];
                    privateDist += t * t; 
                }
#endif

              if(privateDist < curDist) {
                curDist = privateDist;
                curNeighbour = i;
              }
            }

            dist_acc[gid] = s::sqrt(curDist);
            neighbours_acc[gid] = curNeighbour;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    buf_dists.reset();
    buf_neighbors.reset();
    unsigned int check = 1;
    unsigned int sum = 0;
    for(int i = 0; i < size; ++i) {
      if(dists[i] < 0)
        check = 0;
      if(neighbors[i] < 0 || neighbors[i] >= nRef)
        check = 0;
    }

    return check ? true : false;
  }


  static std::string getBenchmarkName() { return "Knn"; }

}; // Knn class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Knn>();
  return 0;
}
