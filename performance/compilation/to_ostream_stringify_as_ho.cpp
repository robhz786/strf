#include <boost/stringify.hpp>
#include <ostream>
#include "args.hpp"

namespace strf = boost::stringify;

void FUNCTION_NAME (std::ostream& out_)
{
    auto* out = out_.rdbuf();

    (void)strf::write(out) .as
           ( "blah blah blah {} {} {} blah {} {} {}\n" )
           ( strf::right(arg_a0, 10)
           , arg_a1
           , +strf::fmt(arg_a2) > 5
           , ~strf::oct(arg_a3) > 6
           , strf::hex(arg_a4)
           , arg_a5 );

    (void)strf::write(out) .as
           ( "blah blah {} {}{} {} {} blah {} {} {}\n" )
           ( arg_b0
           , strf::right(arg_b1, 9)
           , arg_b2
           , arg_b3
           , +strf::fmt(arg_b4) > 5
           , ~strf::oct(arg_b5) > 6
           , strf::hex(arg_b6)
           , arg_b7 );

    (void)strf::write(out) .as
           ( "blah blah {} {:>10} {} {} {} {} {} {}\n" )
           ( arg_c0
           , strf::right(arg_c1, 10)
           , arg_c2
           , +strf::fmt(arg_c3) > 5
           , arg_c4
           , ~strf::oct(arg_c5) > 6
           , strf::hex(arg_c6)
           , arg_c7 );
}
