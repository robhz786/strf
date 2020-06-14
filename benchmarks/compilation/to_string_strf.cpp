#include <strf/to_string.hpp>
#include "strf_args_shuffle.hpp"

void FUNCTION_NAME (std::string& out)
{
    out += strf::to_string("blah ", ARG(0));
    out += strf::to_string("blah ", ARG(1), ' ', ARG(2));
    out += strf::to_string("blah ", ARG(3), ' ', ARG(4), " ", ARG(5));
    out += strf::to_string("blah ", ARG(6), ' ', ARG(7), " ", ARG(8), " ", ARG(9));
    out += strf::to_string("blah ", ARG(10), ' ', ARG(11), " ", ARG(12), " ", ARG(13), " ", ARG(14));
    out += strf::to_string
        ("blah ", ARG(15), ' ', ARG(16), " ", ARG(17), " ", ARG(18), " ", ARG(19), " ", ARG(20) );
}
