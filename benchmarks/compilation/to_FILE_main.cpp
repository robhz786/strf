#include <cstdio>

using output_type = std::FILE*;

#include "tmp/functions_declations.hpp"

const char* format_string()
{
    return "";
}

int main()
{
    FILE* destination = stdout;

#include "tmp/functions_calls.cpp"

    return 0;
}


