#include <cstdio>

using output_type = int; // dummy type

#include "tmp/functions_declations.hpp"

int main()
{
    int destination = 0;

#include "tmp/functions_calls.cpp"

    return 0;
}


