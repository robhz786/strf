# Strf

Branch   | Travis | Appveyor | codecov.io
---------|--------|----------|-----------
develop  | [![Build Status](https://travis-ci.org/robhz786/strf.svg?branch=develop)](https://travis-ci.org/robhz786/strf)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/strf?branch=develop&svg=true)](https://ci.appveyor.com/project/robhz786/strf/branch/develop)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/develop/graph/badge.svg)](https://codecov.io/gh/robhz786/strf/branch/develop)
master   | [![Build Status](https://travis-ci.org/robhz786/strf.svg?branch=master)](https://travis-ci.org/robhz786/strf)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/strf?branch=master&svg=true)](https://ci.appveyor.com/project/robhz786/strf/branch/master)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/master/graph/badge.svg)](https://codecov.io/gh/robhz786/strf/branch/master)

**Version:** 0.10.5

**documentation:** http://robhz786.github.io/strf/doc/quick_reference.html

[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/cpp-strf/strf)


Strf is a C++ formatting library that

* is fast
* is locale independent. ( Not aways an advantange, but usually ).
* supports encoding conversion.

# Overview

```c++
#include <cassert>
#include <iostream>
#include <strf.hpp> // The whole library is included in this header

void samples()
{
    // basic example:
    int value = 255;
    std::string s = strf::to_string(value, " in hexadecimal is ", strf::hex(value));
    assert(s == "255 in hexadecimal is ff");


    // more formatting:  operator>(int width) : align to rigth
    //                   operator*()          : show base
    //                   p(int)               : precision
    s = strf::to_string( "---"
                       , *strf::hex(255).p(4).fill(U'.') > 10
                       , "---" );
    assert(s == "---....0x00ff---");


    // ranges
    int array[] = {20, 30, 40};
    s = strf::to_string( "--[", strf::separated_range(array, " / "), "]--");
    assert(s == "--[20 / 30 / 40]--");


    // range with formatting
    s = strf::to_string( "--["
                       , *strf::hex(strf::separated_range(array, ", ")).p(4)
                       , "]--");
    assert(s == "--[0x0014, 0x001e, 0x0028]--");


    // join: align a group of argument as one:
    s = strf::to_string( "---"
                       , strf::join_center(30, U'.')( value
                                                    , " in hexadecimal is "
                                                    , strf::hex(value) )
                       , "---" );
    assert(s == "---...255 in hexadecimal is ff...---");


    // numeric punctuation ( using facets )
    auto punct_facet = strf::numpunct<base>{3}.thousands_sep(U'.');
    s = strf::to_string .with(punct_facet) (100000000000ll);
    assert(s == "100.000.000.000");


    // encoding conversion
    auto s_utf8 = strf::to_u8string( strf::conv(u"aaa-")
                                   , strf::conv(U"bbb-")
                                   , strf::conv( "\x80\xA4"
                                               , strf::windows_1252<char>() ) );
    assert(s_utf8 == u8"aaa-bbb-\u20AC\u00A4");


    // string append
    strf::assign(s) ("aaa", "bbb");
    strf::append(s) ("ccc", "ddd");
    assert(s == "aaabbbcccddd");


    // other output types
    char buff[500];
    strf::to(buff) (value, " in hexadecimal is ", strf::hex(value));
    strf::to(stdout) ("Hello, ", "World", '!');
    strf::to(std::cout.rdbuf()) ("Hello, ", "World", '!');
    std::u16string s16 = strf::to_u16string( value
                                           , u" in hexadecimal is "
                                           , strf::hex(value) );


    // alternative syntax:
    s = strf::to_string.tr("{} in hexadecimal is {}", value, strf::hex(value));
    assert(s == "255 in hexadecimal is ff");
}
```
# Acknowledgments

- This library uses [Ryu](https://github.com/ulfjack/ryu) to print floating-points. Thanks to Ulf Adams for creating such a great algorithm and providing a C implementation. It saved me a ton of work.
- Thanks to Eyal Rozenberg -- the author of [cuda-kat](https://github.com/eyalroz/cuda-kat) library -- for enabling strf to work on CUDA.

# Compilers

In its current state, Strf is known to work with the following compilers:

* Clang 3.8 (with `--std=c++14` option )
* GCC 6 (with `--std=c++14` option )
* Visual Studio 2017 15.8

# To-do

* Documentation
* Make it possible to use it as a static library.
* More input types:
  - Hexadecimal floating-points.
  - `bool` ( currently converted to `int` and printed as `0` or `1` )
  - `char32_t` ( converting from UTF-32 to the output encoding )
* More encodings:
  - [CESU-8](https://en.wikipedia.org/wiki/CESU-8),
  - [Modified UTF-8](https://en.wikipedia.org/wiki/UTF-8#Modified_UTF-8)
  - ISO 8859-x
  - Windows-125x
* Enable the user to customize how the width is calculated in input strings.
* Support encoding conversion in Tr-string.
