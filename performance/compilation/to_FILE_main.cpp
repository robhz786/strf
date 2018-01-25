#include <cstdio>

using output_type = std::FILE*;

#include "tmp/functions_declations.hpp"

int main()
{
    FILE* destination = stdout;

#include "tmp/functions_calls.cpp"

    return 0;
}


