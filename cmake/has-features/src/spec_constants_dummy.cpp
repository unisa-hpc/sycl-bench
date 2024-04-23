#include <sycl/sycl.hpp>

#ifndef __ACPP__

static constexpr sycl::specialization_id<int> x;

int main(){
    sycl::queue q;

    q.submit([&](sycl::handler& cgh){
        cgh.set_specialization_constant<x>(5);
    });
}

#else

int main(){
    sycl::specialized<int> x(5);
}

#endif