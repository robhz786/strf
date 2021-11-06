# Strf

[![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/strf?branch=main&svg=true)](https://ci.appveyor.com/project/robhz786/strf/branch/main)
[![codecov](https://codecov.io/gh/robhz786/strf/branch/main/graph/badge.svg?token=d5DIZzYv5O)](https://codecov.io/gh/robhz786/strf)

Strf is a C++11 text formatting library that

* [is fast](http://robhz786.github.io/strf-benchmarks/v0.15.3/results.html)
* [is highly extensible](http://robhz786.github.io/strf/v0.15.3/versus_fmtlib.html#_extensibility)
* [can do thing that others can't](https://robhz786.github.io/strf/v0.15.3/versus_fmtlib.html#_strf)

__Attention__ : Branch `master` was renamed to `main` at the time of release 0.15.0.

## Documentation

* Overview
  * [Tutorial](http://robhz786.github.io/strf/v0.15.3/tutorial.html)
  * [Quick reference](http://robhz786.github.io/strf/v0.15.3/quick_reference.html)
  * [Strf versus {fmt}](http://robhz786.github.io/strf/v0.15.3/versus_fmtlib.html)
* How to extend strf:
  * [Adding destination](http://robhz786.github.io/strf/v0.15.3/howto_add_destination.html)
  * [Adding printable types](http://robhz786.github.io/strf/v0.15.3/howto_add_printable_types.html)
  * [Overriding printable types](http://robhz786.github.io/strf/v0.15.3/howto_override_printable_types.html)
* Header references:
  * [`<strf.hpp>`](http://robhz786.github.io/strf/v0.15.3/strf_hpp.html) is the main header. This document is big and covers many details you will probably never need to know. So it's not the best starting point.
  * [`<strf/destination.hpp>`](http://robhz786.github.io/strf/v0.15.3/destination_hpp.html) is a lightweight and freestanding header that defines the `destination` class template. All other headers depend on this one.
  * [`<strf/iterator.hpp>`](http://robhz786.github.io/strf/v0.15.3/iterator_hpp.html) defines an output iterator adapter for the `destination` class template.
  * [`<strf/to_string.hpp>`](http://robhz786.github.io/strf/v0.15.3/to_string_hpp.html) adds support for writting to `std::basic_string`. It includes `<strf.hpp>`.
  * [`<strf/to_cfile.hpp>`](http://robhz786.github.io/strf/v0.15.3/to_cfile_hpp.html)  adds support for writting to `FILE*`. It includes `<strf.hpp>`.
  * [`<strf/to_streambuf.hpp>`](http://robhz786.github.io/strf/v0.15.3/to_streambuf_hpp.html) adds support for writting to `std::basic_streambuf`. It includes `<strf.hpp>`.
* Miscellaneous
  * [How to use strf on CUDA devices](http://robhz786.github.io/strf/v0.15.3/cuda.html)
  * [Benchmarks](http://robhz786.github.io/strf-benchmarks/v0.15.3/results.html)

## Requirements

Strf has been tested in the following compilers:

* Clang 3.8.1
* GCC 6.3.0
* Visual Studio 2017 15.8
* NVCC 11.0

## A glance

```c++
#include <strf/to_string.hpp>
#include <assert>

constexpr int x = 255;

void samples()
{
    // Creating std::string
    auto str = strf::to_string(x, " in hexadecimal is ", *strf::hex(x), '.');
    assert("255 in hexadecimal is 0xff.");

    // Alternative syntax
    auto str_tr = strf::to_string.tr("{} in hexadecimal is {}.", x, *strf::hex(x));
    assert(str_tr == str);

    // Applying a facet
    auto to_string_mc = strf::to_string.with(strf::mixedcase);
    auto str_mc = to_string_mc(x, " in hexadecimal is ", *strf::hex(x), '.');
    assert(str_mc == "255 in hexadecimal is 0xFF.");

    // Achieving the same result, but in multiple steps:
    strf::string_maker str_maker;
    strf::to(str_maker) (x, " in hexadecimal is ");
    strf::to(str_maker).with(strf::mixedcase) (*strf::hex(x), '.');
    auto str_mc_2 = str_maker.finish();
    assert(str_mc_2 == str_mc);

    // Writing instead to char*
    char buff[200];
    strf::to(buff, sizeof(buff)) (x, " in hexadecimal is ", *strf::hex(x), '.');
    assert(str == buff);
}
```

