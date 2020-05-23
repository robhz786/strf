#include <cstdio>
#include "args.hpp"

void FUNCTION_NAME (std::FILE* out)
{
    std::fprintf(out, "blah blah blah %10s %u %+5d blah %#6o %x %d\n",
                 arg_a0, arg_a1, arg_a2, arg_a3, arg_a4, arg_a5);

    std::fprintf(out, "blah blah %s %9s%c %lu %+5d blah %#6llo %x %c\n",
                 arg_b0, arg_b1, arg_b2, arg_b3, arg_b4, arg_b5, arg_b6, arg_b7);

    std::fprintf(out, "blah blah %s %10s %d %+5d %s %#6lo %lx %llu\n",
                 arg_c0, arg_c1, arg_c2, arg_c3, arg_c4, arg_c5, arg_c6, arg_c7);
}
