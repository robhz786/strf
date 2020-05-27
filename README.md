# Strf
Version: 0.11

Branch   | Travis | Appveyor | codecov.io
---------|--------|----------|-----------
develop  | [![Build Status](https://travis-ci.org/robhz786/strf.svg?branch=develop)](https://travis-ci.org/robhz786/strf)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/strf?branch=develop&svg=true)](https://ci.appveyor.com/project/robhz786/strf/branch/develop)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/develop/graph/badge.svg)](https://codecov.io/gh/robhz786/strf/branch/develop)
master   | [![Build Status](https://travis-ci.org/robhz786/strf.svg?branch=master)](https://travis-ci.org/robhz786/strf)| [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/strf?branch=master&svg=true)](https://ci.appveyor.com/project/robhz786/strf/branch/master)| [![codecov](https://codecov.io/gh/robhz786/robhz786/branch/master/graph/badge.svg)](https://codecov.io/gh/robhz786/strf/branch/master)

[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/cpp-strf/strf)

Strf is a C++ formatting library that

* is fast
* is locale independent.
* supports encoding conversion.

## Documentation

The [**introduction**](http://robhz786.github.io/strf/v0.11/introduction.html) is what you should
read if this is your first time here.

After that, the [**quick reference**](http://robhz786.github.io/strf/v0.11/quick_reference.html) provides a nice overview of the library's capabilities.
It is the document that people are supposed to visit more often.

At last, there are the header references, which aim to be a more accurate and complete.
* [`<strf.hpp>`](http://robhz786.github.io/strf/v0.11/strf_hpp.html) is the main header.
* [`<strf/outbuf.hpp>`](http://robhz786.github.io/strf/v0.11/outbuf_hpp.html) is a lightweight and freestanding header that defines the `basic_outbuf` class template.
                       All other headers includes this one.
* [`<strf/to_string.hpp>`](http://robhz786.github.io/strf/v0.11/to_string_hpp.html) adds support for writting to `std::basic_string`. It includes `<strf.hpp>.
* [`<strf/to_cfile.hpp>`](http://robhz786.github.io/strf/v0.11/to_cfile_hpp.html)  adds support for writting to `FILE*`. It includes `<strf.hpp>.
* [`<strf/to_streambuf.hpp>`](http://robhz786.github.io/strf/v0.11/to_streambuf_hpp.html) adds support for writting to `std::basic_streambuf`. It includes `<strf.hpp>.

## Acknowledgments

- This library uses [Ryu](https://github.com/ulfjack/ryu) to print floating-points. Thanks to Ulf Adams for creating such a great algorithm and providing a C implementation. It saved me a ton of work.
- Thanks to Eyal Rozenberg -- the author of [cuda-kat](https://github.com/eyalroz/cuda-kat) library -- for enabling strf to work on CUDA.

## Requirements

Strf demands C++14 features. In the current state, it is known to work with the following compilers:

* Clang 3.8 (with `--std=c++14` option )
* GCC 6 (with `--std=c++14` option )
* Visual Studio 2017 15.8

