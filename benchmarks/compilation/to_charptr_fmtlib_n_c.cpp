#include "fmt/format.h"
#include "fmt/compile.h"
#include <cstdio>
#include "fmt_compile_args_shuffle.hpp"

void FUNCTION_NAME (int)
{
    constexpr std::size_t buff_size = 1000;
    char buff[buff_size];
    char* end = buff + buff_size;
    char* dest = buff;

    dest = fmt::format_to_n(dest, end - dest, FMT_COMPILE(FMT1), ARG(0)).out;
    dest = fmt::format_to_n(dest, end - dest, FMT_COMPILE(FMT2), ARG(1), ARG(2)).out;
    dest = fmt::format_to_n(dest, end - dest, FMT_COMPILE(FMT3), ARG(3), ARG(4), ARG(5)).out;
    dest = fmt::format_to_n(dest, end - dest, FMT_COMPILE(FMT4), ARG(6), ARG(7), ARG(8), ARG(9)).out;
    dest = fmt::format_to_n(dest, end - dest, FMT_COMPILE(FMT5), ARG(10), ARG(11), ARG(12), ARG(13), ARG(14)).out;
    dest = fmt::format_to_n(dest, end - dest, FMT_COMPILE(FMT6), ARG(15), ARG(16), ARG(17), ARG(18), ARG(19), ARG(20)).out;
    *dest = 0;
    std::puts(buff);
}

