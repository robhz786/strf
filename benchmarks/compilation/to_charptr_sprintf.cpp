#include <cstdio>
#include "args_shuffle.hpp"

void FUNCTION_NAME (int)
{
    constexpr std::size_t buff_len = 1000;
    char buff[buff_len];
    char* out = buff;

    out += std::sprintf(out, format_string(), ARG(0));
    out += std::sprintf(out, format_string(), ARG(1), ARG(2));
    out += std::sprintf(out, format_string(), ARG(3), ARG(4), ARG(5));
    out += std::sprintf(out, format_string(), ARG(6), ARG(7), ARG(8), ARG(9));
    out += std::sprintf(out, format_string(), ARG(10), ARG(11), ARG(12), ARG(13), ARG(14));
    std::sprintf(out, format_string(), ARG(15), ARG(16), ARG(17), ARG(18), ARG(19), ARG(20));

    std::puts(buff);
}
