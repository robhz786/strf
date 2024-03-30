#!/bin/bash

set -ex

mkdir cmake_build
cd cmake_build

CXX_FLAGS="-Wall -Wextra -Werror -Wconversion -Wshadow -Wpedantic -fstrict-aliasing -Wstrict-aliasing=1"

if [ -z $CXX_STANDARD ]; then
    cmake -DSTRF_BUILD_TESTS=ON \
          -DSTRF_BUILD_EXAMPLES=ON \
          -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
          -DCMAKE_BUILD_TYPE=$CONFIG \
          -G "Ninja" .. \
    && cmake --build . \
    && ctest -V
else
    cmake -DSTRF_BUILD_TESTS=ON \
          -DSTRF_BUILD_EXAMPLES=ON \
          -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
          -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
          -DCMAKE_BUILD_TYPE=$CONFIG \
          -G "Ninja" .. \
    && cmake --build . \
    && ctest -V
fi

