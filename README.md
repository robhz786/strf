# Boost.Stringify
Boost.Stringify is a C++ locale-independent format library. It is fast, highly extensible and provides some unique features.

It does not make part of the [Boost libraries](www.boost.org), but the plan is to propose there some day. At this moment it is still in the initial stage of development and not ready for production use.

- Documentation: http://robhz786.github.io/stringify/doc/html/
- Version: 0.6

Build status:

Branch   | Travis | Appveyor
---------|--------|---------
Develop  | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=develop)](https://travis-ci.org/robhz786/stringify) | [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=develop&svg=true)](https://ci.appveyor.com/project/robhz786/stringify)
Master   | [![Build Status](https://travis-ci.org/robhz786/stringify.svg?branch=master)](https://travis-ci.org/robhz786/stringify) | [![Build Status](https://ci.appveyor.com/api/projects/status/github/robhz786/stringify?branch=master&svg=true)](https://ci.appveyor.com/project/robhz786/stringify)

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
cmake -G "Visual Studio 15 2017" -DCMAKE_INSTALL_PREFIX=output-dir ..
cmake --build . --config Release --target INSTALL
```

On Posix-like operating systems:
```
cd libs/stringify
mkdir cmake_build
cd cmake_build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=output-dir ..
cmake --build . && make install
```

, where `output-dir` is a directory of your choice. Inside this directory, CMake will create a subdirectory name `cmake` containing a file named `BoostStringify.cmake` that you can include in your CMake project. This file defines the target `boost::stringify` as an imported static library, and this target defines the macro `BOOST_STRINGIFY_NOT_HEADER_ONLY` for the consumers targets. Also, a directory named `include` containing a copy of all the headers of Boost.Stringify is created in `output-dir`, and added as an include directory for the consumer targes. This does not eliminate, however, the need of adding the _boost root directory_ as an include directory, since Boost.Stringify depends on others Boost headers.

##### Option 3: Do it in your own way
Using the build tool of your choice, simply generate a static library from the source file `libs/stringify/build/stringify.cpp`.

# Compilers

In its current state, Boost.Stringify is known to work with the following compilers:

* Clang 3.8 (with `--std=c++14` option )
* GCC 6 (with `--std=c++14` option ) and 7
* Visual Studio 2017 15.8

However, more recent compilers may be necessary as the library evolves.

