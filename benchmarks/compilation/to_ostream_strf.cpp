#include <strf/to_streambuf.hpp>
#include <ostream>
#include "strf_args_shuffle.hpp"

void FUNCTION_NAME (std::ostream& out_)
{
    auto* out = out_.rdbuf();
    (void) strf::to(out)("blah ", ARG(0));
    (void) strf::to(out)("blah ", ARG(1), ' ', ARG(2));
    (void) strf::to(out)("blah ", ARG(3), ' ', ARG(4), " ", ARG(5));
    (void) strf::to(out)("blah ", ARG(6), ' ', ARG(7), " ", ARG(8), " ", ARG(9));
    (void) strf::to(out)("blah ", ARG(10), ' ', ARG(11), " ", ARG(12), " ", ARG(13), " ", ARG(14));
    (void) strf::to(out)
        ("blah ", ARG(15), ' ', ARG(16), " ", ARG(17), " ", ARG(18), " ", ARG(19), " ", ARG(20) );
}
