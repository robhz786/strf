#include <cstdio>

using output_type = int; // dummy type

#include "tmp/functions_declations.hpp"

const char* format_string()
{
    return "";
}

int main()
{
    int destination = 0;

#include "tmp/functions_calls.cpp"

    return 0;
}


