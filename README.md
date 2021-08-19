# Strf

[![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/strf?branch=main&svg=true)](https://ci.appveyor.com/project/robhz786/strf/branch/main)
[![codecov](https://codecov.io/gh/robhz786/strf/branch/main/graph/badge.svg?token=d5DIZzYv5O)](https://codecov.io/gh/robhz786/strf)

Yet another C++ formatting library.

__Attention__ : Branch `master` was renamed to `main` when version 0.15.0 was released.

## Documentation

* Overview
  * [Tutorial](http://robhz786.github.io/strf/v0.15.0/tutorial.html)
  * [Quick reference](http://robhz786.github.io/strf/v0.15.0/quick_reference.html)
  * [Strf versus {fmt}](http://robhz786.github.io/strf/v0.15.0/versus_fmtlib.html)  ( a quick comparison )
* How to extend strf:
  * [Adding destination](http://robhz786.github.io/strf/v0.15.0/howto_add_destination.html)
  * [Adding printable types](http://robhz786.github.io/strf/v0.15.0/howto_add_printable_types.html)
  * [Overriding printable types](http://robhz786.github.io/strf/v0.15.0/howto_override_printable_types.html)
* Header references:
  * [`<strf.hpp>`](http://robhz786.github.io/strf/v0.15.0/strf_hpp.html) is the main header. This document is big and covers many details you will probably never need to know. So it's not the best starting point.
  * [`<strf/outbuff.hpp>`](http://robhz786.github.io/strf/v0.15.0/outbuff_hpp.html) is a lightweight and freestanding header that defines the `basic_outbuff` class template. All other headers depend on this one.
  * [`<strf/to_string.hpp>`](http://robhz786.github.io/strf/v0.15.0/to_string_hpp.html) adds support for writting to `std::basic_string`. It includes `<strf.hpp>`.
  * [`<strf/to_cfile.hpp>`](http://robhz786.github.io/strf/v0.15.0/to_cfile_hpp.html)  adds support for writting to `FILE*`. It includes `<strf.hpp>`.
  * [`<strf/to_streambuf.hpp>`](http://robhz786.github.io/strf/v0.15.0/to_streambuf_hpp.html) adds support for writting to `std::basic_streambuf`. It includes `<strf.hpp>`.
* Miscellaneous
  * [How to use strf on CUDA devices](http://robhz786.github.io/strf/v0.15.0/cuda.html)
  * [Benchmarks](http://robhz786.github.io/strf-benchmarks/v0.15.0/results.html)

## Requirements

Strf has been tested in the following compilers:

* Clang 3.8.1
* GCC 6.3.0
* Visual Studio 2017 15.8
* NVCC 11.0

