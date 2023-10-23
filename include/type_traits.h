#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

template<class T>
struct ReadableTypename
{};

#define MAKE_READABLE_TYPENAME(T, str) \
template<> \
struct ReadableTypename<T> \
{ static const char* name; }; const char* ReadableTypename<T>::name = str;

MAKE_READABLE_TYPENAME(char, "int8")
MAKE_READABLE_TYPENAME(unsigned char, "uint8")
MAKE_READABLE_TYPENAME(short, "int16")
MAKE_READABLE_TYPENAME(unsigned short, "uint16")
MAKE_READABLE_TYPENAME(int, "int32")
MAKE_READABLE_TYPENAME(unsigned int, "uint32")
MAKE_READABLE_TYPENAME(long long, "int64")
MAKE_READABLE_TYPENAME(unsigned long long, "uint64")
MAKE_READABLE_TYPENAME(float, "fp32")
MAKE_READABLE_TYPENAME(double, "fp64")

#endif
