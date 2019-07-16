
// Integer, single precision, double precision
template <typename DATA_TYPE, int N>
void pattern_a(cl::sycl::handler& cgh, std::vector<DATA_TYPE>& A, std::vector<DATA_TYPE>& B){
    cgh.parallel_for<class MicroBenchA>(ndrange,
        [=](cl::sycl::id<1> gid) 
    {
        DATA_TYPE r0, r1, r2, r3; 
        r0=A[threadId];
        r1=r2=r3=r0;
        for (int i=0;i<N;i++) {
            r0 = r0 * r0 + r1;
            r1 = r1 * r1 + r2;
            r2 = r2 * r2 + r3;
            r3 = r3 * r3 + r0;
        }
        B[threadId]=r0;        
    });
}
 
// Special functions
template <typename DATA_TYPE, int N>
void pattern_b(cl::sycl::handler& cgh){
    cgh.parallel_for<class MicroBenchB>(ndrange,
        [=](cl::sycl::id<1> gid) 
    {
        DATA_TYPE r0, r1, r2, r3; 
        r0=A[threadId];
        r1=r2=r3=r0;
        for (int i=0;i<N;i++) {
            r0 = log(r1);
            r1 = cos(r2);
            r2 = log(r3);
            r3 = sin(r0);
        }
        B[threadId]=r0;        
    });
}


// Shared memory
template <typename DATA_TYPE, int N>
void pattern_c(cl::sycl::handler& cgh){
  cgh.parallel_for<class MicroBenchC>(ndrange,
    [=](cl::sycl::id<1> gid) 
    {
    _shared__ DATA_TYPE shared[THREADS]; // FIXME
    DATA_TYPE r0;
    for(int i=0;i<COMP_ITERATIONS;i++) {
        r0 = shared[threadId];
        shared[THREADS - threadId - 1] = r0;
    }
    });
}

// L2 cache memory
template <typename DATA_TYPE, int N>
void pattern_d(cl::sycl::handler& cgh){
  cgh.parallel_for<class MicroBenchD>(ndrange,
    [=](cl::sycl::id<1> gid) 
    {

    DATA_TYPE r0;
    for(int i=0;i<COMP_ITERATIONS;i++) {
        r0 = cdin[threadId];
        cdout[threadId]=r0;
    }
    cdout[threadId]=r0;
    });
}

// DRAM
template <typename DATA_TYPE, int N>
void pattern_e(cl::sycl::handler& cgh){
    cgh.parallel_for<class MicroBenchE>(ndrange,
        [=](cl::sycl::id<1> gid) 
    {
    DATA_TYPE r0, r1;
    r0=A[threadId];
    r1=r0;
    for (int i=0;i<N;i++) {
        r0 = r0 * r0 + r1;
        r1 = r1 * r1 + r0;
    }
    B[threadId]=r0;
    });
}

// MIX FIXME TODO
template <typename DATA_TYPE, int N>
void pattern_mix(cl::sycl::handler& cgh){
    cgh.parallel_for<class MicroBenchF>(ndrange,
        [=](cl::sycl::id<1> gid) 
    {
    DATA_TYPE r0, r1;
    r0=A[threadId];
    r1=r0;
    for (int i=0;i<N;i++) {
        r0 = r0 * r0 + r1;
        r1 = r1 * r1 + r0;
    }
    B[threadId]=r0;
    });
}

