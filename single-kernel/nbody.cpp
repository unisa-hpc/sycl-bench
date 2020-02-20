#include "common.h"

#include <iostream>
#include <cassert>

using namespace cl;

template<class float_type> class NDRangeNBodyKernel;
template<class float_type> class HierarchicalNBodyKernel;


template<class float_type>
class NBody
{
protected:
  using particle_type = sycl::vec<float_type, 4>;
  using vector_type = sycl::vec<float_type, 3>;

  std::vector<particle_type> particles;
  std::vector<vector_type> velocities;

  BenchmarkArgs args;


  const float_type gravitational_softening;
  const float_type dt;

  PrefetchedBuffer<particle_type> output_particles;
  PrefetchedBuffer<vector_type> output_velocities;

  PrefetchedBuffer<particle_type> particles_buf;
  PrefetchedBuffer<vector_type> velocities_buf;
public:
  NBody(const BenchmarkArgs& _args)
      : args(_args), gravitational_softening{1.e-5f}, dt{1.e-2f} {
    assert(args.problem_size % args.local_size == 0);
  }

  void setup() {
    
    particles.resize(args.problem_size);
    velocities.resize(args.problem_size);

    for(std::size_t i = 0; i < args.problem_size; ++i) {

      float_type rel_i = static_cast<float_type>(i) / static_cast<float_type>(args.problem_size);

      particles[i].x() = rel_i * std::cos(3000.f * 2.f * M_PI * rel_i);
      particles[i].y() = rel_i * std::sin(3000.f * 2.f * M_PI * rel_i);
      particles[i].z() = rel_i;
      particles[i].w() = static_cast<float_type>(1.0);

      velocities[i].x() = 0;
      velocities[i].y() = 0;
      velocities[i].z() = 0;
    }

    particles_buf. initialize(args.device_queue, this->particles.data(), sycl::range<1>{this->args.problem_size});
    velocities_buf.initialize(args.device_queue, this->velocities.data(), sycl::range<1>{this->args.problem_size});

    output_particles. initialize(args.device_queue, sycl::range<1>{args.problem_size});
    output_velocities.initialize(args.device_queue, sycl::range<1>{args.problem_size});
  }


  bool verify(VerificationSetting &ver) {
    auto resulting_particles = output_particles.template get_access<sycl::access::mode::read>();
    auto resulting_velocities = output_velocities.template get_access<sycl::access::mode::read>();

    std::vector<particle_type> host_resulting_particles(particles.size());
    std::vector<vector_type> host_resulting_velocities(particles.size());
  

    for(std::size_t i = 0; i < particles.size(); ++i) {
      const particle_type my_p = particles[i];
      const vector_type my_v = velocities[i];
      vector_type acceleration{static_cast<float_type>(0.0f)};

      for(std::size_t j = 0; j < particles.size(); ++j) {

        if(i != j) {
          const particle_type p = particles[j];
          
          const vector_type R {
            p.x() - my_p.x(), 
            p.y() - my_p.y(),
            p.z() - my_p.z()
          };

          const float_type r_inv = sycl::rsqrt(R.x() * R.x() + R.y() * R.y() + R.z() * R.z() + 
                                              gravitational_softening);

          acceleration += static_cast<float_type>(p.w()) * r_inv * r_inv * r_inv * R;
        }
        
      }

      vector_type new_v = my_v + acceleration * dt;
      particle_type new_p = my_p;
      new_p.x() += new_v.x() * dt; 
      new_p.y() += new_v.y() * dt;
      new_p.z() += new_v.z() * dt;

      host_resulting_particles[i] = new_p;
      host_resulting_velocities[i] = new_v;
    }

    double deviation = std::sqrt(
        calculateSquaredDifference(host_resulting_particles.data(), resulting_particles.get_pointer(), particles.size()) +
        calculateSquaredDifference(host_resulting_velocities.data(), resulting_velocities.get_pointer(), particles.size()));

    return deviation < 1.e-6;
  }

protected:

  template<class T>
  double calculateSquaredDifference(sycl::vec<T,3> a, sycl::vec<T,3> b) {
    auto diff = a - b;
    diff *= diff;

    return static_cast<float_type>(diff.x()+diff.y()+diff.z());
  }

  template<class T>
  double calculateSquaredDifference(sycl::vec<T,4> a, sycl::vec<T,4> b) {
    auto diff = a - b;
    diff *= diff;

    return static_cast<float_type>(diff.x()+diff.y()+diff.z()+diff.w());
  }

  template<class T>
  double calculateSquaredDifference(const T* a, const T* b, std::size_t size) {

    double result = 0.0;

    for(std::size_t i = 0; i < size; ++i) {
      result += calculateSquaredDifference(a[i], b[i]);
    }

    return result;
  }

  void submitNDRange(sycl::buffer<particle_type>& particles, sycl::buffer<vector_type>& velocities) {
    args.device_queue.submit([&](sycl::handler& cgh) {
      sycl::nd_range<1> execution_range{sycl::range<1>{args.problem_size}, sycl::range<1>{args.local_size}};

      auto particles_access = particles.template get_access<sycl::access::mode::read>(cgh);
      auto velocities_access = velocities.template get_access<sycl::access::mode::read>(cgh);

      auto output_particles_access = output_particles.template get_access<sycl::access::mode::discard_write>(cgh);
      auto output_velocities_access = output_velocities.template get_access<sycl::access::mode::discard_write>(cgh);

      auto scratch = sycl::accessor<particle_type, 1, sycl::access::mode::read_write, sycl::access::target::local>{
          sycl::range<1>{args.local_size}, cgh};

      cgh.parallel_for<NDRangeNBodyKernel<float_type>>(execution_range,
          [=, dt = this->dt, gravitational_softening = this->gravitational_softening](sycl::nd_item<1> tid) {

            const size_t global_id = tid.get_global_id(0);
            const size_t local_id = tid.get_local_id(0);
            const size_t num_particles = tid.get_global_range()[0];
            const size_t local_size = tid.get_local_range()[0];

            vector_type v = velocities_access[global_id];

            vector_type acceleration{static_cast<float_type>(0.0f)};

            particle_type my_particle = (global_id < num_particles) ? particles_access[global_id]
                                                                    : particle_type{static_cast<float_type>(0.0f)};

            for(size_t offset = 0; offset < num_particles; offset += local_size) {
              scratch[local_id] = (global_id < num_particles) ? particles_access[offset + local_id]
                                                              : particle_type{static_cast<float_type>(0.0f)};

              tid.barrier();

              for(int i = 0; i < local_size; ++i) {
                const particle_type p = scratch[i];

                const vector_type R{p.x() - my_particle.x(), p.y() - my_particle.y(), p.z() - my_particle.z()};

                const float_type r_inv =
                    sycl::rsqrt(R.x() * R.x() + R.y() * R.y() + R.z() * R.z() + gravitational_softening);


                if(global_id != offset + i)
                  acceleration += static_cast<float_type>(p.w()) * r_inv * r_inv * r_inv * R;
              }

              tid.barrier();
            }

            // This is a dirt cheap Euler integration, but could be
            // converted into a much better leapfrog itnegration when properly
            // initializing the velocities
            v += acceleration * dt;

            // Update position
            my_particle.x() += v.x() * dt;
            my_particle.y() += v.y() * dt;
            my_particle.z() += v.z() * dt;

            if(global_id < num_particles) {
              output_velocities_access[global_id] = v;
              output_particles_access[global_id] = my_particle;
            }
          });
    });
  }

  void submitHierarchical(sycl::buffer<particle_type>& particles, sycl::buffer<vector_type>& velocities) {
    args.device_queue.submit([&](sycl::handler& cgh) {
      sycl::nd_range<1> execution_range{sycl::range<1>{args.problem_size}, sycl::range<1>{args.local_size}};

      auto particles_access = particles.template get_access<sycl::access::mode::read>(cgh);
      auto velocities_access = velocities.template get_access<sycl::access::mode::read>(cgh);

      auto output_particles_access = output_particles.template get_access<sycl::access::mode::discard_write>(cgh);
      auto output_velocities_access = output_velocities.template get_access<sycl::access::mode::discard_write>(cgh);

      auto scratch = sycl::accessor<particle_type, 1, sycl::access::mode::read_write, sycl::access::target::local>{
          sycl::range<1>{args.local_size}, cgh};


      const size_t local_size = args.local_size;
      const size_t problem_size = args.problem_size;
      cgh.parallel_for_work_group<HierarchicalNBodyKernel<float_type>>(sycl::range<1>{problem_size / local_size},
          sycl::range<1>{local_size},
          [=, dt = this->dt, gravitational_softening = this->gravitational_softening](sycl::group<1> grp) {
            sycl::private_memory<particle_type> my_particle{grp};
            sycl::private_memory<vector_type> acceleration{grp};

            grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
              acceleration(idx) = vector_type{static_cast<float_type>(0.0f)};
              my_particle(idx) = (idx.get_global_id(0) < problem_size) ? particles_access[idx.get_global_id(0)]
                                                                       : particle_type{static_cast<float_type>(0.0f)};
            });


            for(size_t offset = 0; offset < problem_size; offset += local_size) {
              grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
                scratch[idx.get_local_id(0)] = (idx.get_global_id(0) < problem_size)
                                                   ? particles_access[offset + idx.get_local_id(0)]
                                                   : particle_type{static_cast<float_type>(0.0f)};
              });

              grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
                for(int i = 0; i < local_size; ++i) {
                  const particle_type p = scratch[i];
                  const particle_type my_p = my_particle(idx);

                  const vector_type R{p.x() - my_p.x(), p.y() - my_p.y(), p.z() - my_p.z()};

                  const float_type r_inv =
                      sycl::rsqrt(R.x() * R.x() + R.y() * R.y() + R.z() * R.z() + gravitational_softening);


                  if(idx.get_global_id(0) != offset + i)
                    acceleration(idx) += static_cast<float_type>(p.w()) * r_inv * r_inv * r_inv * R;
                }
              });
            }

            grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
              const size_t global_id = idx.get_global_id(0);

              vector_type v = velocities_access[global_id];
              // This is a dirt cheap Euler integration, but could be
              // converted into a much better leapfrog integration when properly
              // initializing the velocities to the state at 0.5*dt
              v += acceleration(idx) * dt;

              // Update position
              particle_type my_p = my_particle(idx);
              my_p.x() += v.x() * dt;
              my_p.y() += v.y() * dt;
              my_p.z() += v.z() * dt;

              if(global_id < problem_size) {
                output_velocities_access[global_id] = v;
                output_particles_access[global_id] = my_p;
              }
            });
          });
    });
  }
};

template<class float_type>
class NBodyNDRange : public NBody<float_type>
{
public:
  using typename NBody<float_type>::particle_type;
  using typename NBody<float_type>::vector_type;

  NBodyNDRange(const BenchmarkArgs& _args)
  : NBody<float_type>{_args} {}


  void run(){
    this->submitNDRange(this->particles_buf.get(), this->velocities_buf.get());
  }

  std::string getBenchmarkName() {
    std::stringstream name;
    name << "NBody_NDRange_";
    name << ReadableTypename<float_type>::name;
    return name.str();
  }
};


template<class float_type>
class NBodyHierarchical : public NBody<float_type>
{
public:
  using typename NBody<float_type>::particle_type;
  using typename NBody<float_type>::vector_type;

  NBodyHierarchical(const BenchmarkArgs& _args)
  : NBody<float_type>{_args} {}


  void run(){
    this->submitHierarchical(this->particles_buf.get(), this->velocities_buf.get());
  }

  std::string getBenchmarkName() {
    std::stringstream name;
    name << "NBody_Hierarchical_";
    name << ReadableTypename<float_type>::name;
    
    return name.str();
  }
};

int main(int argc, char** argv)
{

  BenchmarkApp app(argc, argv);

  app.run< NBodyHierarchical<float> >();
  app.run< NBodyHierarchical<double> >();

  if(app.shouldRunNDRangeKernels()) {
    app.run< NBodyNDRange<float> >();
    app.run< NBodyNDRange<double> >();
  }

  return 0;
}