#define BOOST_STRINGIFY_NOT_HEADER_ONLY
#include <boost/stringify.hpp>
#include "args.hpp"

namespace strf = boost::stringify;

void FUNCTION_NAME (int)
{
    constexpr std::size_t buff_len = 1000;
    char buff[buff_len];
    char *end = buff + buff_len;
    char* out = buff;
    std::size_t len = 0;

    strf::write_to(out, end, &len) ["blah blah blah {} {} {} blah {} {} {}\n"]
        &= { {arg_a0, {10, ">"}}
           , arg_a1
           , {arg_a2, {5, ">+"}}
           , {arg_a3, {6, ">#o"}}
           , {arg_a4, "x"}
           , arg_a5
           };

    out += len;
    strf::write_to(out, end, &len) ["blah blah {} {}{} {} {} blah {} {} {}\n"]
        &= { arg_b0
           , {arg_b1, {9, ">"}}
           , arg_b2
           , arg_b3
           , {arg_b4, {5, ">+"}}
           , {arg_b5, {6, ">#o"}}
           , {arg_b6, "x"}
           , arg_b7
           };

    out += len;
    strf::write_to(out, end) ["blah blah {} {:>10} {} {} {} {} {} {}\n"]
        &= { arg_c0
           , {arg_c1, {10,">"}}
           , arg_c2
           , {arg_c3, {5,">+"}}
           , arg_c4
           , {arg_c5, {6,">#o"}}
           , {arg_c6, "x"}
           , arg_c7
           };

    std::puts(buff);
}
