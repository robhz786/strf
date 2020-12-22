#!/bin/bash

set -ex

mkdir cmake_build
cd cmake_build
cmake -DSTRF_BUILD_TESTS=ON \
      -DSTRF_BUILD_EXAMPLES=ON \
      -DCMAKE_C_FLAGS="-Wall -Wextra" \
      -DCMAKE_CXX_FLAGS="-Wall -Wextra -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=all" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined -fno-sanitize-recover=all -fuse-ld=gold" \
      -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
      -DCMAKE_BUILD_TYPE=Debug \
      -G "Unix Makefiles" .. \
&& cmake --build . \
&& ctest -V

#      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined" \
