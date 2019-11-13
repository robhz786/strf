# Strf

Branch   | Travis | Appveyor | codecov.io
---------|--------|----------|-----------
develop  | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=develop)](https://travis-ci.org/robhz786/stringify)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=develop&svg=true)](https://ci.appveyor.com/project/robhz786/stringify/branch/develop)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/develop/graph/badge.svg)](https://codecov.io/gh/robhz786/stringify/branch/develop)
master   | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=master)](https://travis-ci.org/robhz786/stringify)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=master&svg=true)](https://ci.appveyor.com/project/robhz786/stringify/branch/master)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/master/graph/badge.svg)](https://codecov.io/gh/robhz786/stringify/branch/master)

**Version:** 0.10.
**documentation:** http://robhz786.github.io/stringify/doc/quick_reference.html

Strf is C++ formatting library that

* is highly extensible.
* is highly customizable
* is fast
* is locale independent. ( Not aways an advantange, but usually ).
* supports encoding conversion.

# Compilers

In its current state, Strf is known to work with the following compilers:

* Clang 3.8 (with `--std=c++14` option )
* GCC 7
* Visual Studio 2017 15.8

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

