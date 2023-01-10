#include <sycl.hpp>
#include <iostream>

#include "common.h"

namespace s = sycl;


class AesEncryptKernel; // kernel forward declaration
class AesDecKernel; // kernel forward declaration

//TODO: non produce il risultato giusto.
class AesEncrypt
{
protected:
    size_t w, h; // size of the input picture
    size_t size; // user-defined size (input and output will be size x size)
    size_t local_size;
    bool check_result;
    std::vector<s::uchar4> input;
    std::vector<s::uchar4> output; 
    std::vector<s::uchar4> dec_output; 
    std::vector<s::uchar> expandedKey;
    std::vector<s::uchar> roundKey; 
    std::vector<s::uchar> key;
    unsigned int keySize = 16; // 1 Byte = 8 bits
    unsigned int explandedKeySize; 
    const unsigned int rounds = 10;
    const unsigned int seed = 123;
    
    BenchmarkArgs args;

    PrefetchedBuffer<s::uchar4, 1> buf_input;    
    PrefetchedBuffer<s::uchar4, 1> buf_output; 
    PrefetchedBuffer<s::uchar4, 1> buf_dec_output;    
    PrefetchedBuffer<s::uchar, 1> buf_SBox; 
    PrefetchedBuffer<s::uchar, 1> buf_RSBox;    
    PrefetchedBuffer<s::uchar, 1> buf_roundKey;  

    unsigned char sbox[256] = 
    {  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76 //0
    , 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0 //1
    , 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15 //2
    , 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75 //3
    , 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84 //4
    , 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf //5
    , 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8 //6
    , 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2 //7
    , 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73 //8
    , 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb //9
    , 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79 //A
    , 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08 //B
    , 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a //C
    , 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e //D
    , 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf //E
    , 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};
    
    unsigned char rsbox[256] =
    { 0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb
    , 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb
    , 0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e
    , 0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25
    , 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92
    , 0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84
    , 0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06
    , 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b
    , 0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73
    , 0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e
    , 0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b
    , 0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4
    , 0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f
    , 0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef
    , 0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61
    , 0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};       
 
  unsigned char Rcon[255] = 
  { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a
  , 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39
  , 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a
  , 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8
  , 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef
  , 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc
  , 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b
  , 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3
  , 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94
  , 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
  , 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35
  , 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f
  , 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04
  , 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63
  , 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd
  , 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb      };
  
  s::uchar4 shiftRows(s::uchar4 row, unsigned int j)
  {
      s::uchar4 r = row;
      for(uint i=0; i < j; ++i)  
      {
          //r.xyzw() = r.yzwx();
          s::uchar x = r.x();
          s::uchar y = r.y();
          s::uchar z = r.z();
          s::uchar w = r.w();
          r = {y,z,w,x};
      }
      return r;
  }

  s::uchar4 shiftRowsInv(s::uchar4 row, unsigned int j)
  {
      s::uchar4 r = row;
      for(uint i=0; i < j; ++i)  
      {
          // r = r.wxyz();
          s::uchar x = r.x();
          s::uchar y = r.y();
          s::uchar z = r.z();
          s::uchar w = r.w();
          r = {w,x,y,z};
      }
      return r;
  }

        

  unsigned char galoisMultiplication(unsigned char a, unsigned char b)
  {
      unsigned char p = 0; 
      for(unsigned int i=0; i < 8; ++i)
      {
          if((b&1) == 1)
          {
              p^=a;
          }
          unsigned char hiBitSet = (a & 0x80);
          a <<= 1;
          if(hiBitSet == 0x80)
          {
              a ^= 0x1b;
          }
          b >>= 1;
      }
      return p;
  }

  inline s::uchar4 sboxRead(const s::uchar * SBox, s::uchar4 block)
  {
      return {SBox[block.x()], SBox[block.y()], SBox[block.z()], SBox[block.w()]};
  }

  s::uchar4 mixColumns(const s::uchar4 * block, const s::uchar4 * galiosCoeff, unsigned int j)
  {
      unsigned int bw = 4;

      s::uchar x, y, z, w;

      x = galoisMultiplication(block[0].x(), galiosCoeff[(bw-j)%bw].x());
      y = galoisMultiplication(block[0].y(), galiosCoeff[(bw-j)%bw].x());
      z = galoisMultiplication(block[0].z(), galiosCoeff[(bw-j)%bw].x());
      w = galoisMultiplication(block[0].w(), galiosCoeff[(bw-j)%bw].x());
    
      for(unsigned int k=1; k< 4; ++k)
      {
          x ^= galoisMultiplication(block[k].x(), galiosCoeff[(k+bw-j)%bw].x());
          y ^= galoisMultiplication(block[k].y(), galiosCoeff[(k+bw-j)%bw].x());
          z ^= galoisMultiplication(block[k].z(), galiosCoeff[(k+bw-j)%bw].x());
          w ^= galoisMultiplication(block[k].w(), galiosCoeff[(k+bw-j)%bw].x());
      }
      
      return {x, y, z, w};
  }

  template<typename T>
  int fillRandom(
      T * arrayPtr,
      const int width,
      const int height,
      const T rangeMin,
      const T rangeMax,
      unsigned int seed=123)
  {
    if(!arrayPtr)
    {
      std::cerr << "Cannot fill array. NULL pointer.\n";
      return 1;
    }
    if(!seed)
    {
      seed = (unsigned int)time(NULL);
    }
    srand(seed);
    double range = double(rangeMax - rangeMin) + 1.0;
    /* random initialisation of input */
    for(int i = 0; i < height; i++)
      for(int j = 0; j < width; j++)
      {
        int index = i*width + j;
        arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0));
      }
    return 0;
  }


  void rotate(unsigned char * word)
  {
    unsigned char c = word[0];
    for(unsigned int i=0; i<3; ++i)
      word[i] = word[i+1];
    word[3] = c;
  }
  
  unsigned char getRconValue(unsigned int num)
  {
    return Rcon[num];
  }

  unsigned char getSBoxValue(unsigned int num)
  {
    return sbox[num];
  }

  unsigned char getSBoxInvert(unsigned int num)
  {
    return rsbox[num];
  }

  void core(unsigned char * word, unsigned int iter)
  {
    rotate(word);

    for(unsigned int i=0; i < 4; ++i)
    {
      word[i] = getSBoxValue(word[i]);
    }    

    word[0] = word[0]^getRconValue(iter);
  }
  
  void keyExpansion(unsigned char * key, unsigned char * expandedKey,
                  unsigned int keySize, unsigned int explandedKeySize)
  {
    unsigned int currentSize    = 0;
    unsigned int rConIteration = 1;
    unsigned char temp[4]      = {0};

    for(unsigned int i=0; i < keySize; ++i)
    {
      expandedKey[i] = key[i];
    }

    currentSize += keySize;

    while(currentSize < explandedKeySize)
    {
      for(unsigned int i=0; i < 4; ++i)
      {
        temp[i] = expandedKey[(currentSize - 4) + i];
      }

      if(currentSize%keySize == 0)
      {
        core(temp, rConIteration++);
      }

      //XXX: add extra SBOX here if the keySize is 32 Bytes

      for(unsigned int i=0; i < 4; ++i)
      {
        expandedKey[currentSize] = expandedKey[currentSize - keySize]^temp[i];
        currentSize++;
      }
    }
  }

  void createRoundKey(unsigned char * eKey, unsigned char * rKey)
  {
    for(unsigned int i=0; i < 4; ++i)
      for(unsigned int j=0; j < 4; ++j)
      {
        rKey[i+ j*4] = eKey[i*4 + j];
      }
  }

public:
  AesEncrypt(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
   
    size = args.problem_size; // input size defined by the user
    local_size = args.local_size; // set local work_group size
    check_result = args.verification.enabled;

    key.resize(keySize);
    fillRandom<unsigned char>(key.data(), keySize, 1, 0, 255, seed);

    // declare some variables for intializing data 
    explandedKeySize = (rounds+1)*keySize;
    expandedKey.resize(explandedKeySize);
    roundKey.resize(explandedKeySize);
    input.resize(size*size);
    // init input
    for(int i=0; i < size*size; ++i) {
          s::uchar x = i;
          s::uchar y = i+1;
          s::uchar z = i+2;
          s::uchar w = i+3;
        // input[i] = {static_cast<s::uchar>(i), static_cast<s::uchar>(i+1), static_cast<s::uchar>(i+2), static_cast<s::uchar>(i+3)};
        input[i] = (s::uchar4){x,y, z,w};
    }

    output.resize(size*size);
    dec_output.resize(size*size);
   
    keyExpansion(key.data(), expandedKey.data(), keySize, explandedKeySize);
    for(unsigned int i = 0; i < rounds+1; ++i)
    {
      createRoundKey(expandedKey.data() + keySize*i, roundKey.data() + keySize*i);
    }

    // init buffer
    buf_input.initialize(args.device_queue, input.data(), s::range<1>(size*size));
    buf_roundKey.initialize(args.device_queue, roundKey.data(), s::range<1>(explandedKeySize));
    // Works only with local size 256 
    buf_SBox.initialize(args.device_queue, sbox, s::range<1>(local_size));
    buf_RSBox.initialize(args.device_queue, rsbox, s::range<1>(local_size));
    buf_output.initialize(args.device_queue, output.data(), s::range<1>(size*size));
    buf_dec_output.initialize(args.device_queue, dec_output.data(), s::range<1>(size*size));

  }


  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
    
    auto input_acc = buf_input.get_access<s::access::mode::read>(cgh);
    auto output_acc = buf_output.get_access<s::access::mode::read_write>(cgh);
    auto roundKey_acc = buf_roundKey.get_access<s::access::mode::read>(cgh);
    auto sBox_acc = buf_SBox.get_access<s::access::mode::read>(cgh);
    s::local_accessor<s::uchar4, 1> block0 {s::range<1>{keySize/4}, cgh};
    s::local_accessor<s::uchar4, 1> block1 {s::range<1>{keySize/4}, cgh};

    
    s::range<2> gws (size, size);
    s::range<2> lws (4, 1);

    cgh.parallel_for<class AesEncryptKernel>(s::nd_range<2>{gws,lws}, [input_acc, roundKey_acc, sBox_acc, output_acc, block0, block1, size_ = size, local_size_ = local_size, rounds_=rounds, this](s::nd_item<2> item) {
        unsigned int blockIdx = item.get_group(1);
        unsigned int blockIdy = item.get_group(0);
    
        //unsigned int localIdx = item.get_local_id(1);
        unsigned int localIdy = item.get_local_id(0);
        
        unsigned int globalIndex = (((blockIdy * size_) + blockIdx) * 4 )+ (localIdy);
        unsigned int localIndex  = localIdy;

        s::uchar4 galiosCoeff[4];
        galiosCoeff[0] = {2, 0, 0, 0};
        galiosCoeff[1] = {3, 0, 0, 0};
        galiosCoeff[2] = {1, 0, 0, 0};
        galiosCoeff[3] = {1, 0, 0, 0};

        block0[localIndex]  = input_acc[globalIndex];
        
        block0[localIndex] ^= roundKey_acc[localIndex];

        for(unsigned int r=1; r < rounds_; ++r)
        {
            block0[localIndex] = sboxRead(sBox_acc.get_pointer(), block0[localIndex]);

            block0[localIndex] = shiftRows(block0[localIndex], localIndex); 
          
            s::group_barrier(item.get_group());
            block1[localIndex]  = mixColumns(block0.get_pointer(), galiosCoeff, localIndex); 
            
            s::group_barrier(item.get_group());
            block0[localIndex] = block1[localIndex]^roundKey_acc[r*4 + localIndex];
        }  
        block0[localIndex] = sboxRead(sBox_acc.get_pointer(), block0[localIndex]);
      
        block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

        output_acc[globalIndex] =  block0[localIndex]^roundKey_acc[(rounds_)*4 + localIndex];
      });
    }));

    if(check_result){
      args.device_queue.wait();
      
      events.push_back(args.device_queue.submit([&](s::handler& cgh){
        auto input_acc = buf_output.get_access<s::access::mode::read>(cgh);
        auto output_acc = buf_dec_output.get_access<s::access::mode::read_write>(cgh);
        auto roundKey_acc = buf_roundKey.get_access<s::access::mode::read>(cgh);
        auto sBox_acc = buf_RSBox.get_access<s::access::mode::read>(cgh);
        s::local_accessor<s::uchar4, 1> block0 {s::range<1>{keySize/4}, cgh};
        s::local_accessor<s::uchar4, 1> block1 {s::range<1>{keySize/4}, cgh};
        
        s::range<2> gws (size, size);
        s::range<2> lws (4, 1);
        cgh.parallel_for<class AesDecKernel>(s::nd_range<2>{gws,lws}, [input_acc, roundKey_acc, sBox_acc, output_acc, block0, block1, size_ = size, local_size_ = local_size, rounds_=rounds, this](s::nd_item<2> item) {
            unsigned int blockIdx = item.get_group(1);
            unsigned int blockIdy = item.get_group(0);
        
            // unsigned int localIdx = item.get_local_id(1);
            unsigned int localIdy = item.get_local_id(0);
            
            unsigned int globalIndex = (((blockIdy * size_) + blockIdx) * 4 )+ (localIdy);
            unsigned int localIndex  = localIdy;

            s::uchar4 galiosCoeff[4];
            galiosCoeff[0] = {14, 0, 0, 0};
            galiosCoeff[1] = {11, 0, 0, 0};
            galiosCoeff[2] = {13, 0, 0, 0};
            galiosCoeff[3] = { 9, 0, 0, 0};

            block0[localIndex]  = input_acc[globalIndex];
            
            block0[localIndex] ^= roundKey_acc[4*rounds_ + localIndex];

            for(unsigned int r=rounds_ -1 ; r > 0; --r)
            {
                block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 
            
                block0[localIndex] = sboxRead(sBox_acc.get_pointer(), block0[localIndex]);
                
                s::group_barrier(item.get_group());
                block1[localIndex] = block0[localIndex]^roundKey_acc[r*4 + localIndex];
                s::group_barrier(item.get_group());

                block0[localIndex]  = mixColumns(block1.get_pointer(), galiosCoeff, localIndex); 
            }  

            block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

            block0[localIndex] = sboxRead(sBox_acc.get_pointer(), block0[localIndex]);

            output_acc[globalIndex] =  block0[localIndex]^roundKey_acc[localIndex];

        });
   
  
      })//end submit
      );//end push back
    }
   }
    
  bool verify(VerificationSetting& ver) {
    unsigned int check = 1;
		buf_output.reset();
    buf_dec_output.reset();
		
   for(int i = 0; i< size*size; ++i) {
      if(dec_output[i].x()-input[i].x() != 0)
        return false;
      else if(dec_output[i].y()-input[i].y() != 0)
        return false;
      else if(dec_output[i].z()-input[i].z() != 0)
        return false;
      else if(dec_output[i].w()-input[i].w() != 0)
        return false;
		}
    return true;    
  }


  static std::string getBenchmarkName() { return "Aes Encrypt"; }

}; // AesEncrypt class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<AesEncrypt>();  
  return 0;
}


