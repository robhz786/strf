#!/bin/bash

set -ex

mkdir cmake_build
cd cmake_build
cmake -DSTRF_BUILD_TESTS=ON \
      -DSTRF_BUILD_EXAMPLES=ON \
      -DCMAKE_CXX_FLAGS="-Wall -Wextra -Werror -Wconversion -Wshadow -Wpedantic -fstrict-aliasing -Wstrict-aliasing=1" \
      -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
      -DCMAKE_BUILD_TYPE=$CONFIG \
      -G "Ninja" .. \
&& cmake --build . \
&& ctest -V

