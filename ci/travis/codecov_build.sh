#!/bin/bash

set -ex

root_dir="$( pwd -P )"
src_dir="$( cd include; pwd -P )"

mkdir cmake_build
cd cmake_build

script_dir="$( pwd -P )"

cmake -DSTRF_BUILD_TESTS=ON \
      -DSTRF_BUILD_EXAMPLES=ON \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DCMAKE_C_COMPILER=gcc-7 \
      -DCMAKE_CXX_FLAGS="-Wall -Wextra -O0 -fprofile-arcs -ftest-coverage -DNDEBUG" \
      -DCMAKE_C_FLAGS="-Wall -Wextra -O0 -fprofile-arcs -ftest-coverage -DNDEBUG" \
      -DCMAKE_EXE_LINKER_FLAGS="-ftest-coverage -fprofile-arcs" \
      -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
      -DCMAKE_BUILD_TYPE=Debug \
      -G "Unix Makefiles" ..

make test-static-lib

lcov --gcov-tool=$GCOV --initial --base-directory $script_dir \
     --directory=$root_dir --capture --output-file all.info

./tests/static-lib

lcov --gcov-tool=$GCOV --rc lcov_branch_coverage=1 \
     --base-directory $script_dir --directory=$root_dir \
     --capture --output-file all.info
lcov --gcov-tool=$GCOV --rc lcov_branch_coverage=1 \
     --extract all.info "*include/strf/*" --output-file coverage.info

curl -s https://codecov.io/bash > .codecov
chmod +x .codecov
./.codecov -f coverage.info -X gcov -x "$GCOV"

