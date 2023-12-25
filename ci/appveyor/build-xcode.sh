#!/bin/bash

set -ex

mkdir cmake_build
cd cmake_build

CXX_FLAGS="-Wall -Wextra -Werror -Wconversion -Wshadow -Wpedantic -fstrict-aliasing -Wstrict-aliasing=1"

cmake -DSTRF_BUILD_TESTS=ON \
      -DSTRF_BUILD_EXAMPLES=ON \
      -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
      -DCMAKE_BUILD_TYPE=$CONFIG \
      -G "Xcode" .. \
      && cmake --build . --config $CONFIG \
      && ctest -V -C $CONFIG

