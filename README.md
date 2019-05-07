# Boost.Stringify

Boost.Stringify is not currently part of the [Boost libraries](www.boost.org), but the plan is to propose there some day. At this moment it is still in the initial stage of development and not ready for production use.

This is a C++ locale-independent format library that:

* supports encoding conversion and sanitization
* supports [input ranges](https://robhz786.github.io/stringify/doc/html/special_input_types/special_input_types.html#ranges)
* is able to align many arguments as they were one ( see [joins](https://robhz786.github.io/stringify/doc/html/special_input_types/special_input_types.html#special_input_types.special_input_types.joins) )
* is highly extensible, both for input and output types. For example, you can easily extend the library to write into QString (from Qt).  
* supports char, wchar_t, char32_t and char16_t
* is suitable to be used together with translation tools like [gettext](https://en.wikipedia.org/wiki/Gettext).
* has good performance ( not the best in all scenarios, but still good )

**Documentation:** http://robhz786.github.io/stringify/doc/html/

**Version:** 0.8.2

Branch   | Travis | Appveyor | codecov.io
---------|--------|----------|----------- 
develop  | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=develop)](https://travis-ci.org/robhz786/stringify)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=develop&svg=true)](https://ci.appveyor.com/project/robhz786/stringify/branch/develop)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/develop/graph/badge.svg)](https://codecov.io/gh/robhz786/stringify/branch/develop)
master   | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=master)](https://travis-ci.org/robhz786/stringify)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=master&svg=true)](https://ci.appveyor.com/project/robhz786/stringify/branch/master)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/master/graph/badge.svg)](https://codecov.io/gh/robhz786/stringify/branch/master)



# Installation
You can either use it as a header-only library or as a static library. In any case, you first need to have the Boost installed. It's enough to simply download and unpack the [tarball](https://www.boost.org/users/download/). After this, in the command line interface, inside the unpacked directory &#x2014; which is commonly referead as the _boost root directory_ &#x2014; run the following commands according to your operating system:

On Windows:
```
cd libs
git clone https://github.com/robhz786/stringify/
cd ..
.\bootstrap
.\b2 headers
```

On POSIX-like operating system:
```
cd libs
git clone https://github.com/robhz786/stringify/
cd ..
./bootstrap.sh
./b2 headers
```

Now just specify the _boost root directory_ as an include path to your compiler, and you are ready to use Stringify as a header only library.

But if you want to use it as a static library instead &#x2014; which is a good idea since it dramatically reduces the code bloat &#x2014; the code that uses the library must have the macro `BOOST_STRINGIFY_NOT_HEADER_ONLY` defined. And in order to build the library, there are three ways:

##### Option 1: Using Boost.Build system
In the command line interface, in the _boost root directory_, run the command:
On Windows:
```
.\b2 libs\stringify\build
```

On Posix-like operating systems:
```
./b2 libs/stringify/build
```
This will generate the library file somewhere inside the `bin.v2` sub-directory of the _boost root directory_, as indicated by the `b2` command output

##### Option 2: Using CMake
In the command line interface, in the _boost root directory_, run the command:

On Windows:

```
cd libs\stringify
mkdir cmake_build
cd cmake_build
cmake -G "Visual Studio 15 2017" ^
  -DBOOST_INSTALL_PREFIX=<output-dir> ^
  -DBOOST_INSTALL_SUBDIR=<subdir> ^
  -DCMAKE_CXX_STANDARD=14 ..
cmake --build . --config Release --target INSTALL
```

On Posix-like operating systems:
```
cd libs/stringify
mkdir cmake_build
cd cmake_build
cmake -G "Unix Makefiles" \
   -DCMAKE_BUILD_TYPE=Release \
   -DBOOST_INSTALL_PREFIX=<output-dir> \
   -DBOOST_INSTALL_SUBDIR=<subdir>  \
   -DCMAKE_CXX_STANDARD=14 ..
cmake --build . && make install
```
Where `<output-dir>` and `<subdir>` are paths of your choice. `<subdir>` must be a relative path. By default, `<output-dir>` is this _boost root directory_, and `<subdir>` is the concatenation of two CMake variables: `${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}`.

The last command install Boost.Stringify into `<output-dir>` including the binaries (inside `<subdir>` ) and headers. It also creates the file `<output-dir>/<subdir>/cmake/boost_stringify.cmake` that you can include in your CMake project and that defines the target `boost::stringify` as an imported static library.

If you additionaly want to check whether the unit tests pass in your environment, then add the option `-DBOOST_BUILD_TESTS=ON` on the `cmake -G ...` command. And run the command `ctest -C Release`.

If you additionaly want to run the benchmarks in your environment, then add the option `-DBOOST_BUILD_BENCHMARKS=ON` on the `cmake -G ...` command. This will cause the install process to create a directory `benchmark` inside the `<subdir>` directory with the benchmark programs.

##### Option 3: Do it in your own way
Using the build tool of your choice, simply generate a static library from the source file `libs/stringify/build/stringify.cpp`.

# Compilers

In its current state, Boost.Stringify is known to work with the following compilers:

* Clang 3.8 (with `--std=c++14` option )
* GCC 7
* Visual Studio 2017 15.8

However, more recent compilers may be necessary as the library evolves.

