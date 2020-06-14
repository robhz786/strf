#include <strf.hpp>
#include "strf_args_shuffle.hpp"

void FUNCTION_NAME (int)
{
    constexpr std::size_t buff_len = 1000;
    char buff[buff_len];
    char *end = buff + buff_len;
    char* out = buff;

    out = strf::to(out, end)
        ("blah ", ARG(0))
        .ptr;
    out = strf::to(out, end)
        ("blah ", ARG(1), ' ', ARG(2))
        .ptr;
    out = strf::to(out, end)
        ("blah ", ARG(3), ' ', ARG(4), " ", ARG(5))
        .ptr;
    out = strf::to(out, end)
        ("blah ", ARG(6), ' ', ARG(7), " ", ARG(8), " ", ARG(9))
        .ptr;
    out = strf::to(out, end)
        ("blah ", ARG(10), ' ', ARG(11), " ", ARG(12), " ", ARG(13), " ", ARG(14))
        .ptr;
    out = strf::to(out, end)
        ("blah ", ARG(15), ' ', ARG(16), " ", ARG(17), " ", ARG(18), " ", ARG(19), " ", ARG(20))
        .ptr;

    std::puts(buff);
}
