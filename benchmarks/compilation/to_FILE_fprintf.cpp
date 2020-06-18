#include <cstdio>
#include "args_shuffle.hpp"

void FUNCTION_NAME (std::FILE* out)
{
    std::fprintf(out, format_string(), ARG(0));
    std::fprintf(out, format_string(), ARG(1), ARG(2));
    std::fprintf(out, format_string(), ARG(3), ARG(4), ARG(5));
    std::fprintf(out, format_string(), ARG(6), ARG(7), ARG(8), ARG(9));
    std::fprintf(out, format_string(), ARG(10), ARG(11), ARG(12), ARG(13), ARG(14));
    std::fprintf(out, format_string(), ARG(15), ARG(16), ARG(17), ARG(18), ARG(19), ARG(20));
}
