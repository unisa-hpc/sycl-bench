#include <sycl.hpp>
#include <iostream>

#include "common.h"

namespace s = sycl;

typedef s::uchar4 pixel;
#define p(i) perm[i]


class PerlinNoiseKernel; // kernel forward declaration


class PerlinNoise
{
protected:
    size_t size; // user-defined size (input and output will be size x size)
    size_t local_size;
    BenchmarkArgs args;
    int width;
    float time;

    std::vector<s::uchar4> output;
    
 
    PrefetchedBuffer<s::uchar4, 1> buf_output;

    float fade(float t)
    {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    float grad(int hash, float x, float y, float z)
    {
    int h = hash & 15;            // Convert low 4 bits of hash code
    float u = (h < 8) ? x : y;    // into 12 gradient directions.
    float v = (h < 4) ? y : (h == 12 || h == 14) ? x : z;

    u = (h & 1) == 0 ? u : -u;
    v = (h & 2) == 0 ? v : -v;
    return u + v;
    }

    float noise3(float x, float y, float z)
    {
    int perm[512] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233,
    7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
    190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174,
    20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48,
    27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65,
    25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18,
    169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198,
    173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147,
    118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17,
    182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44,
    154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
    253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191,
    179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214,
    31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127,
    4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243,
    141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90,
    15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30,
    69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234,
    75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177,
    33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111,
    229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40,
    244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132,
    187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164,
    100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5,
    202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47,
    16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152,
    2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22,
    39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179,
    162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181,
    199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
    138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78,
    66, 215, 61, 156, 180
    };


    float floor_x = s::floor(x);
    float floor_y = s::floor(y);
    float floor_z = s::floor(z);

    int X = (int)(floor_x) & 255; // Find unit cube that
    int Y = (int)(floor_y) & 255; // contains point.
    int Z = (int)(floor_z) & 255;

    x -= floor_x;                 // Find relative x,y,z
    y -= floor_y;                 // of point in cube.
    z -= floor_z;

    float x1 = x - 1.0f;
    float y1 = y - 1.0f;
    float z1 = z - 1.0f;

    float u = fade(x);           // Compute fade curves
    float v = fade(y);           // for each of x,y,z.
    float w = fade(z);

    int A = p(X) + Y;
    int AA = p(A) + Z;
    int AB = p(A + 1) + Z;       // Hash coordinates of
    int B = p(X + 1) + Y;        // the 8 cube corners.
    int BA = p(B) + Z;
    int BB = p(B + 1) + Z;

    float g0 = grad(p(AA), x, y, z);
    float g1 = grad(p(BA), x1, y, z);
    float g2 = grad(p(AB), x, y1, z);
    float g3 = grad(p(BB), x1, y1, z);

    float g4 = grad(p(AA + 1), x, y, z1);
    float g5 = grad(p(BA + 1), x1, y, z1);
    float g6 = grad(p(AB + 1), x, y1, z1);
    float g7 = grad(p(BB + 1), x1, y1, z1);

    // Add blended results from 8 corners of cube.
    float u01 = s::mix(g0, g1, u);
    float u23 = s::mix(g2, g3, u);
    float u45 = s::mix(g4, g5, u);
    float u67 = s::mix(g6, g7, u);

    float v0 = s::mix(u01, u23, v);
    float v1 = s::mix(u45, u67, v);

    return s::mix(v0, v1, w);
    }

    /** Compute perlin noise algorithm given time offset, rowstride and image size. */
    void compute_perlin_noise(pixel * output, const float time, const unsigned int rowstride, int img_height, int img_width)
    {
    unsigned int i, j;
    float vy, vt;
    float vdx = 0.03125f;
    float vdy = 0.0125f;
    float vs = 2.0f;
    float bias = 0.35f;
    float vx = 0.0f;
    float red, green, blue;
    float xx, yy;

    for (j = 0; j < img_height; j++) {
        for (i = 0; i < img_width; i++) {
        vx = ((float) i) * vdx;
        vy = ((float) j) * vdy;
        vt = time * vs;

        xx = vx * vs;
        yy = vy * vs;

        red = noise3(xx, vt, yy);
        green = noise3(vt, yy, xx);
        blue = noise3(yy, xx, vt);

        red += bias;
        green += bias;
        blue += bias;

        // Clamp to within [0 .. 1]
        red = (red > 1.0f) ? 1.0f : red;
        green = (green > 1.0f) ? 1.0f : green;
        blue = (blue > 1.0f) ? 1.0f : blue;

        red = (red < 0.0f) ? 0.0f : red;
        green = (green < 0.0f) ? 0.0f : green;
        blue = (blue < 0.0f) ? 0.0f : blue;

        red *= 255.0f;
        green *= 255.0f;
        blue *= 255.0f;

        output[(j * rowstride) + i].x() = (unsigned char) red;
        output[(j * rowstride) + i].y() = (unsigned char) green;
        output[(j * rowstride) + i].z() = (unsigned char) blue;
        output[(j * rowstride) + i].w() = 255;
        }
    }
    }

    /** Perform the host-side perlin noise computation, report performance and optionally verify the device results. */
    int compute_host_and_verify(int iterations, 
                                unsigned * output_device, int rowstride,
                                int img_height, int img_width, int data_size)
    {
    int retval = 0;
    int i;  

    unsigned* output_host = (unsigned *) malloc(sizeof(s::uchar4) * data_size);

    float delta;
    float time;

    printf("Compute Host Data\n");
    time = 0.0f;
    
    for (i = 0; i < iterations; i++) {
        compute_perlin_noise((pixel *) output_host, time, rowstride, img_height, img_width);
        time += 0.05f;
    }      
        
        
        // Verify results
        //
        printf("Verifying....\n");

        int failure_count = 0;
        for (i = 0; i < data_size; i++) {
        pixel *device_pixel = (pixel *) & output_device[i];
        pixel *host_pixel = (pixel *) & output_host[i];

        //The rgb values may be off by 1 but still be correct because of rounding error.
        //For example, if the host computes 164.000015 and the device 163.999985,
        //the resulting values will be 164 and 163, respectively.

        if ((abs(device_pixel -> x() - host_pixel-> x() ) > 1) ||
            (abs(device_pixel -> y() - host_pixel-> y() ) > 1) ||
            (abs(device_pixel -> z() - host_pixel-> z() ) > 1) || 
            (abs(device_pixel -> w() - host_pixel-> w() ) > 1)) {
            retval = 1;
    fprintf(stderr, "Error [%d]: H 0x%08X 0x%08X\n", i, output_host[i], output_device[i]);
            failure_count++;
        }    

        
    }

    printf("Verification Complete: %d errors out of %d pixels\n", failure_count, data_size);

    free(output_host);
    return (retval==0);
    }


public:
  PerlinNoise(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {
    size = args.problem_size; // input size defined by the user
    local_size = args.local_size; // set local work_group size
    
    width = (int)floor(sqrt(size));
    size = width * width;
    time = 0;
    output.resize(size);

    buf_output.initialize(args.device_queue, output.data(), s::range<1>(size));    
    
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {

    auto output_acc = buf_output.get_access<s::access::mode::write>(cgh);


    s::range<1> ndrange{size};

      cgh.parallel_for<class PerlinNoiseKernel>(ndrange, [this, output_acc,time_= time, width_=width, num_elements = size](s::id<1> gid) {
           	int id = gid[0];
            if (id >= num_elements) return;
            int i = id % width_;
            int j = id / width_;

            const float vdx = 0.03125f;
            const float vdy = 0.0125f;
            const float vs = 2.0f;
            const float bias = 0.35f;
        
            float vx = static_cast<float>(i) * vdx;
            float vy = static_cast<float>(j) * vdy;
            float vt = time_ * 2.0f;

            s::float4 colors; // 0 = red, 1 = green, 2 = blue
            
            float xx = vx * vs;
            float yy = vy * vs;

            // helps the insieme kernel_analysis_utils
            float red = noise3(xx, vt, yy);
            float green = noise3(vt, yy, xx);
            float blue = noise3(yy, xx, vt);

            colors.x() = red;
            colors.y() = green;
            colors.z() = blue;
        
            colors = colors + bias;
            colors.w() = 1.0f; // alpha channel will be 255 in the end 

            s::clamp(colors, 0.0f, 1.0f);
            colors = colors * 255.0f;
            s::uchar4 final_color {(s::uchar) colors.x(), (s::uchar) colors.y(),(s::uchar) colors.z(), (s::uchar)colors.w()};
            output_acc[id] = final_color;
      });
    }));
   }
    
  bool verify(VerificationSetting& ver) {
	buf_output.reset();
    int check = compute_host_and_verify(1, (unsigned *)output.data(), width, width, width, size);
    
	return check;
  }


  static std::string getBenchmarkName() { return "PerlinNoise"; }

}; // PerlinNoise class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<PerlinNoise>();  
  return 0;
}


