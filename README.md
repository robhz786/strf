# Boost.Stringify

Branch   | Travis | Appveyor | codecov.io
---------|--------|----------|-----------
develop  | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=develop)](https://travis-ci.org/robhz786/stringify)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=develop&svg=true)](https://ci.appveyor.com/project/robhz786/stringify/branch/develop)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/develop/graph/badge.svg)](https://codecov.io/gh/robhz786/stringify/branch/develop)
master   | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=master)](https://travis-ci.org/robhz786/stringify)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=master&svg=true)](https://ci.appveyor.com/project/robhz786/stringify/branch/master)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/master/graph/badge.svg)](https://codecov.io/gh/robhz786/stringify/branch/master)

**Version:** 0.9.
**Full documentation:** http://robhz786.github.io/stringify/doc/html/index.html

Boost.Stringify C++ formatting library that

* is highly extensible.
* is highly customizable (see [facets](http://robhz786.github.io/stringify/doc/html/index.html#boost_stringify.overview.tour.facets) ). 
* is fast ( see the [benchmarks](http://robhz786.github.io/stringify/doc/html/benchmarks/benchmarks.html) ).
* is locale independent. ( Not aways an advantange, but usually ).
* supports encoding conversion.

It is not currently part of the [Boost libraries](www.boost.org), but the plan is to propose it there some day.

```c++
#include <cassert>
#include <iostream>
#include <boost/stringify.hpp> // The whole library is included in this header

void samples()
{
    namespace strf = boost::stringify::v0; // Everything is inside this namespace.
                                           // ( v0 is an inline namespace ).

    // basic example:
    int value = 255;
    std::string s = strf::to_string(value, " in hexadecimal is ", strf::hex(value));
    assert(s == "255 in hexadecimal is ff");


    // more formatting:  operator>(int width) : align to rigth
    //                   operator~()          : show base
    //                   p(int)               : set precision
    s = strf::to_string( "---"
                       , ~strf::hex(255).p(4).fill(U'.') > 10
                       , "---" );
    assert(s == "---....0x00ff---");

    // ranges
    int array[] = {20, 30, 40};
    const char* separator = " / ";
    s = strf::to_string( "--[", strf::range(array, separator), "]--");
    assert(s == "--[20 / 30 / 40]--");

    // range with formatting
    s = strf::to_string( "--["
                       , ~strf::hex(strf::range(array, separator)).p(4)
                       , "]--");
    assert(s == "--[0x0014 / 0x001e / 0x0028]--");

    // join: align a group of argument as one:
    s = strf::to_string( "---"
                       , strf::join_center(30, U'.')( value
                                                    , " in hexadecimal is "
                                                    , strf::hex(value) )
                       , "---" );
    assert(s == "---...255 in hexadecimal is ff...---");

    // encoding conversion
    auto s_utf8 = strf::to_u8string( strf::cv(u"aaa-")
                                   , strf::cv(U"bbb-")
                                   , strf::cv( "\x80\xA4"
                                             , strf::windows_1252<char>() ) );
    assert(s_utf8 == u8"aaa-bbb-\u20AC\u00A4");

    // string append
    strf::assign(s) ("aaa", "bbb");
    strf::append(s) ("ccc", "ddd");
    assert(s == "aaabbbcccddd");

    // other output types
    char buff[500];
    strf::write(buff) (value, " in hexadecimal is ", strf::hex(value));
    strf::write(stdout) ("Hello, ", "World", '!');
    strf::write(std::cout.rdbuf()) ("Hello, ", "World", '!');
    std::u16string s16 = strf::to_u16string( value
                                           , u" in hexadecimal is "
                                           , strf::hex(value) );

    // alternative syntax:
    s = strf::to_string.tr("{} in hexadecimal is {}", value, strf::hex(value));
    assert(s == "255 in hexadecimal is ff");
}
```

# Installation

Stringify depends on [Boost](https://www.boost.org) and on [Outbuf](https://github.com/robhz786/outbuf/) libraries.
In order to install Boost, it's enough to simply download and unpack the [tarball](https://www.boost.org/users/download/).
After this, in the command line interface, inside the unpacked directory &#x2014; which is commonly referead as the _boost root directory_ &#x2014; run the following commands according to your operating system:

On Windows:
```
cd libs
git clone https://github.com/robhz786/outbuf/
git clone https://github.com/robhz786/stringify/
cd ..
.\bootstrap
.\b2 headers
```

On POSIX-like operating system:
```
cd libs
git clone https://github.com/robhz786/outbuf/
git clone https://github.com/robhz786/stringify/
cd ..
./bootstrap.sh
./b2 headers
```
Now just specify the _boost root directory_ as an include path to your compiler, and you are ready to use Stringify as a header only library.

If you want to use it as a static library the code that uses the library must have the macro `BOOST_STRINGIFY_SEPARATE_COMPILATION` defined. And in order to build the library, there are three ways:

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

