#include <cstdio>
#include <string>

using output_type = std::string&;

#include "tmp/functions_declations.hpp"

int main()
{
    std::string destination;
    destination.reserve(10000);

#include "tmp/functions_calls.cpp"

    puts(destination.c_str());
    return 0;
}


