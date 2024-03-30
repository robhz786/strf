#!/bin/bash

set -ex

mkdir cmake_build
cd cmake_build

CXXFLAGS="${CXXFLAGS} -Wall -Wextra -Werror -Wconversion -Wshadow -Wpedantic -fstrict-aliasing -Wstrict-aliasing=1"
#LDFLAGS="-lstdc++ -lm"

#find /usr/include -type f -name 'c++config.h'
#find /usr -type f -name 'libstdc++*'

if [ -z $CXX_STANDARD ]; then
    cmake -DSTRF_BUILD_TESTS=ON \
          -DSTRF_BUILD_EXAMPLES=ON \
          -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
          -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
          -DCMAKE_BUILD_TYPE=$CONFIG \
          -G "Ninja" .. \
    && cmake --build . \
    && ctest -V
else
    cmake -DSTRF_BUILD_TESTS=ON \
          -DSTRF_BUILD_EXAMPLES=ON \
          -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
          -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
          -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
          -DCMAKE_BUILD_TYPE=$CONFIG \
          -G "Ninja" .. \
    && cmake --build . \
    && ctest -V
fi

