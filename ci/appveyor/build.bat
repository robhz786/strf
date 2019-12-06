@ECHO ON

mkdir cmake_build
dir
cd cmake_build
cmake -A %ARCH% ^
    -DCMAKE_CXX_FLAGS=" /W4 /WX /EHsc " ^
    -DCMAKE_CXX_STANDARD=%CXX_STANDARD% ^
    -DSTRF_BUILD_TESTS=ON ^
    -DSTRF_BUILD_EXAMPLES=ON ^
    -G "%GENERATOR%"  .. ^
  && cmake --build . --config %CONFIG%





    
