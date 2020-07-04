#if defined(FMT_HEADER_ONLY)
#  include <fmt/format.h>
#else
#  include <fmt/core.h>
#endif
#include "fmt/compile.h"
#include "fmt_compile_args_shuffle.hpp"

void FUNCTION_NAME (std::string& out)
{
    out += fmt::format(FMT_COMPILE(FMT1), ARG(0));
    out += fmt::format(FMT_COMPILE(FMT2), ARG(1), ARG(2));
    out += fmt::format(FMT_COMPILE(FMT3), ARG(3), ARG(4), ARG(5));
    out += fmt::format(FMT_COMPILE(FMT4), ARG(6), ARG(7), ARG(8), ARG(9));
    out += fmt::format(FMT_COMPILE(FMT5), ARG(10), ARG(11), ARG(12), ARG(13), ARG(14));
    out += fmt::format(FMT_COMPILE(FMT6), ARG(15), ARG(16), ARG(17), ARG(18), ARG(19), ARG(20));
}
