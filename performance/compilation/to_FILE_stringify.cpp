#define BOOST_STRINGIFY_NOT_HEADER_ONLY
#include <boost/stringify.hpp>
#include <cstdio>
#include "args.hpp"

namespace strf = boost::stringify;

void FUNCTION_NAME (std::FILE* out)
{
    strf::format(out) ("blah blah blah {} {} {} blah {} {} {}\n").exception
           ( arg_c0
           , strf::right(arg_c1, 10)
           , arg_c2
           , +strf::fmt(arg_c3) > 5
           , arg_c4
           , ~strf::oct(arg_c5) > 6
           , strf::hex(arg_c6)
           , arg_c7
           );

    strf::format(out) ("blah blah {} {}{} {} {} blah {} {} {}\n").exception
           ( arg_b0
           , strf::right(arg_b1, 9)
           , arg_b2
           , arg_b3
           , +strf::fmt(arg_b4) > 5
           , ~strf::oct(arg_b5) > 6
           , strf::hex(arg_b6)
           , arg_b7
           );

    strf::format(out) ("blah blah {} {:>10} {} {} {} {} {} {}\n").exception
           ( arg_c0
           , strf::right(arg_c1, 10)
           , arg_c2
           , +strf::fmt(arg_c3) > 5
           , arg_c4
           , ~strf::oct(arg_c5) > 6
           , strf::hex(arg_c6)
           , arg_c7
           );
}
