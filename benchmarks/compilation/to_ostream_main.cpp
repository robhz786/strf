#include <iostream>

using output_type = std::ostream&;

#include "tmp/functions_declations.hpp"

const char* format_string()
{
    return "";
}

int main()
{
    std::ostream & destination = std::cout;

#include "tmp/functions_calls.cpp"

    return 0;
}


