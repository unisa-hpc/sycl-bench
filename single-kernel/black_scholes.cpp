#include <sycl.hpp>
#include <iostream>

#include "common.h"

namespace s = sycl;
// various constants used in the core BlackScholes computations
#define ZERO		0.0f
#define ONE		1.0f
#define HALF		0.5f
#define A1		0.319381530f
#define A2		-0.356563782f
#define A3		1.781477937f
#define A4		-1.821255978f
#define A5		1.330274429f
#define INV_ROOT2PI	0.39894228f
#define NCDF		0.2316419f

#define FLOAT float
#define FIXED uint
#define SFIXED int

#define SELECT(_a, _b, _c) (_c ? _b : _a)
#define DIVIDE(_x,_y) (_x/_y)
#define RECIP(_x) (1.0f/(_x))

#define SQRT(_x) s::sqrt(_x)
#define LOG(_x) s::log(_x)
#define EXP(_x) s::exp(_x)


class BlackScholesKernel; // kernel forward declaration

/*
  A Sobel filter with a convolution matrix 3x3.
  Input and output are two-dimensional buffers of floats.     
 */
class BlackScholes
{
protected:
    size_t w, h; // size of the input picture
    size_t size; // user-defined size (input and output will be size x size)
    size_t local_size;
    BenchmarkArgs args;


    PrefetchedBuffer<FIXED, 1> cpflag_buf;  
    PrefetchedBuffer<FLOAT, 1> S0_buf;    
    PrefetchedBuffer<FLOAT, 1> K_buf;    
    PrefetchedBuffer<FLOAT, 1> r_buf;    
    PrefetchedBuffer<FLOAT, 1> sigma_buf;    
    PrefetchedBuffer<FLOAT, 1> T_buf; 
    PrefetchedBuffer<FLOAT, 1> answer_buf;
        
 double N(double x)
    {
        double k, n;
        double accum;
        double candidate_answer;
        int flag;
        flag = (x < 0);
        x = (x < 0) ? -x : x;
        k = 1.0 / (1.0 + 0.2316419 * x);
        accum = A4 + A5 * k;
        accum = k * accum + A3;
        accum = k * accum + A2;
        accum = k * accum + A1;
        accum = k * accum;
        n = exp(-0.5 * x * x);
        n *= INV_ROOT2PI;
        candidate_answer = 1.0 - n * accum;
        return (flag ? 1.0 - candidate_answer : candidate_answer);
    }


    double bsop_reference(int cpflag, double S0, double K, double r,
                        double sigma, double T) {
        double d1, d2, c, p, Nd1, Nd2, expval, answer;
        d1 = log(S0 / K) + (r + 0.5 * sigma * sigma) * T;
        d1 /= (sigma * sqrt(T));
        expval = exp(-r * T);
        d2 = d1 - sigma * sqrt(T);
        Nd1 = N(d1);
        Nd2 = N(d2);
        c = S0 * Nd1 - K * expval * Nd2;
        p = K * expval * (1.0 - Nd2) - S0 * (1.0 - Nd1);
        answer = cpflag ? c : p;
        return answer;
    }

public:
  BlackScholes(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    local_size = args.local_size; // set local work_group size
    // declare some variables for intializing data 
    int idx;
    int S0Kdex, rdex, sigdex, Tdex;
    FLOAT S0_array[4] = { 42.0, 30.0, 54.0, 66.0 };
    FLOAT K_array[16] = { 40.0, 36.0, 44.0, 48.0,
          24.0, 28.0, 32.0, 36.0,
          48.0, 52.0, 56.0, 60.0,
          60.0, 64.0, 68.0, 72.0
            };
    FLOAT r_array[4] = { 0.1, 0.09, 0.11, 0.12 };
    FLOAT sigma_array[4] = { 0.2, 0.15, 0.25, 0.30 };
    FLOAT T_array[4] = { 0.5, 0.25, 0.75, 1.0 };
    idx = 0;
    /* Pointers used to allocate memory and split that memory into input and output arrays */
	  /* These pointers point to the data buffers needed for Black Scholes computation */
    FIXED *cpflag;
    FLOAT *S0, *K, *r, *sigma, *T, *answer;

    cpflag = (FIXED*)malloc(size * sizeof(int));
    S0 = (FLOAT*)malloc(size * sizeof(FLOAT));
    K = (FLOAT*)malloc(size * sizeof(FLOAT));
    r = (FLOAT*)malloc(size * sizeof(FLOAT));
    sigma = (FLOAT*)malloc(size * sizeof(FLOAT));
    T = (FLOAT*)malloc(size * sizeof(FLOAT));
    answer = (FLOAT*)malloc(size * sizeof(FLOAT));

    /* Here we load some values to simulate real-world options parameters.
	* Users who wish to provide live data would replace this clause
	* with their own initialization of the arrays. */
    for (int k = 0; k < size; ++k) {
      int *temp_int;
      Tdex = (idx >> 1) & 0x3;
      sigdex = (idx >> 3) & 0x3;
      rdex = (idx >> 5) & 0x3;
      S0Kdex = (idx >> 7) & 0xf;

      temp_int = (int *) &cpflag[k];
      temp_int[0] = (idx & 1) ? 0xffffffff : 0;
      if (sizeof(FLOAT) == 8) temp_int[1] = (idx & 1) ? 0xffffffff : 0;

      S0[k] = S0_array[S0Kdex >> 2];
      K[k] = K_array[S0Kdex];
      r[k] = r_array[rdex];
      sigma[k] = sigma_array[sigdex];
      T[k] = T_array[Tdex];
      answer[k] = 0.0f;
      idx++;
    }

    // Il kernel è monodimensionale
    // init buffer
    cpflag_buf.initialize(args.device_queue, cpflag, s::range<1>(size));
    S0_buf.initialize(args.device_queue, S0, s::range<1>(size));
    K_buf.initialize(args.device_queue, K, s::range<1>(size));
    r_buf.initialize(args.device_queue, r, s::range<1>(size));
    sigma_buf.initialize(args.device_queue, sigma, s::range<1>(size));
    T_buf.initialize(args.device_queue, T, s::range<1>(size));
    answer_buf.initialize(args.device_queue, answer, s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
    auto dm_cpflag = cpflag_buf.get_access<s::access::mode::read>(cgh);
    auto dm_S0 = S0_buf.get_access<s::access::mode::read>(cgh);
    auto dm_K = K_buf.get_access<s::access::mode::read>(cgh);
    auto dm_r = r_buf.get_access<s::access::mode::read>(cgh);
    auto dm_sigma = sigma_buf.get_access<s::access::mode::read>(cgh);
    auto dm_T = T_buf.get_access<s::access::mode::read>(cgh);
    auto dm_answer = answer_buf.get_access<s::access::mode::read_write>(cgh);


    s::range<1> ndrange{size};

    cgh.parallel_for<class BlackScholesKernel>(ndrange, [dm_cpflag, dm_S0,dm_K,dm_r,dm_sigma,dm_T,dm_answer,  size_ = size](s::id<1> gid) {
        	uint tid = gid[0];
            if(tid >= size_)
                return;

            FIXED cpflag = dm_cpflag[tid];
            FLOAT S0 = dm_S0[tid];
            FLOAT K = dm_K[tid];
            FLOAT r = dm_r[tid];
            FLOAT sigma = dm_sigma[tid];
            FLOAT T = dm_T[tid];

            FLOAT d1, d2, Nd1, Nd2, expval;
            FLOAT k1, n1, k2, n2;
            FLOAT accum1, accum2;
            FLOAT candidate_answer1, candidate_answer2;
            FLOAT call, put;
            SFIXED flag1, flag2;
            d1 = LOG(DIVIDE(S0,K)) + (r + HALF * sigma * sigma) * T; //5 fo + 2 sf
            d1 = DIVIDE (d1, (sigma * SQRT(T)));    // 1 fo+ 2 sf
            expval = EXP(ZERO - r * T);             //2 fo + 1 sf
            d2 = d1 - sigma * SQRT(T);              // 2 fo + 1 sf
            flag1 = (d1 < ZERO);
            flag2 = (d2 < ZERO);
            d1 = fabs(d1);                          // up 2 fo
            d2 = fabs(d2);
            k1 = RECIP(ONE + NCDF * d1);            //2 fo + 1 sf
            k2 = RECIP(ONE + NCDF * d2);            //2 fo + 1 sf
            accum1 = A4 + A5 * k1;
            accum2 = A4 + A5 * k2;
            accum1 = k1 * accum1 + A3;
            accum2 = k2 * accum2 + A3;
            accum1 = k1 * accum1 + A2;
            accum2 = k2 * accum2 + A2;
            accum1 = k1 * accum1 + A1;
            accum2 = k2 * accum2 + A1;
            accum1 = k1 * accum1;
            accum2 = k2 * accum2;                   // up 18 fo
            n1 = EXP(ZERO - HALF * d1 * d1);        // 3 fo + 1 sf
            n2 = EXP(ZERO - HALF * d2 * d2);        // 3 fo + 1 sf
            n1 *= INV_ROOT2PI;
            n2 *= INV_ROOT2PI;
            candidate_answer1 = ONE - n1 * accum1;
            candidate_answer2 = ONE - n2 * accum2;  //up 6fo
            Nd1 = SELECT(candidate_answer1, (ONE - candidate_answer1), flag1);
            Nd2 = SELECT(candidate_answer2, (ONE - candidate_answer2), flag2);
            call = S0 * Nd1 - K * expval * Nd2;
            put = K * expval * (ONE - Nd2) - S0 * (ONE - Nd1);  //up 12 fo
            dm_answer[tid] = SELECT(put, call, cpflag);
      });
    }));
   }
   
  bool verify(VerificationSetting& ver) {
    auto S0_fptr = S0_buf.get().get_host_access();
    auto K_fptr = K_buf.get().get_host_access();
    auto r_fptr = r_buf.get().get_host_access();
    auto sigma_fptr = sigma_buf.get().get_host_access();
    auto T_fptr = T_buf.get().get_host_access();
    auto answer_fptr = answer_buf.get().get_host_access();
    auto cpflag_fptr = cpflag_buf.get().get_host_access();


    double maxouterr = -1.0;
    double maxouterrindex = -1;
    unsigned long i;
    for (i = 0; i < size; i += 1) {
    	double a, b, absb, del, abserr, relerr, outerr;
    	int *temp_int;
    	a = (double) answer_fptr[i];
    	temp_int = (int *) &cpflag_fptr[i];
    	b = bsop_reference(*temp_int, (double) S0_fptr[i],
    					   (double) K_fptr[i], (double) r_fptr[i],
    					   (double) sigma_fptr[i], (double) T_fptr[i]);
    	del = a - b;
    	abserr = del;
    	del = (del < 0.0f) ? -del : del;
    	absb = (b < 0.0f) ? -b : b;
    	relerr = del / absb;
    	outerr = (del > relerr) ? relerr : del;
    	if (outerr > maxouterr) {
    		maxouterr = outerr;
    	    maxouterrindex = i;
        }
    }
    if (maxouterr > 0.00002) {
		return false;
	} else {
		return true;
	}           
  }


  static std::string getBenchmarkName() { return "Black Scholes"; }

}; // BlackScholes class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<BlackScholes>();  
  return 0;
}


