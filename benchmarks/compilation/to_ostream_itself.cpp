#include <ostream>
#include <iomanip>
#include "args.hpp"

void FUNCTION_NAME (std::ostream& out)
{
    out << "blah blah blah "
        << std::right << std::setw(10) << arg_a0
        << ' ' << std::dec << arg_a1
        << ' ' << std::showpos << std::setw(5) << arg_a2
        << " blah "
        << std::showbase << std::right << std::oct << std::setw(6) << arg_a3
        << ' ' << std::noshowbase << std::hex << arg_a4
        << ' ' << std::noshowpos << arg_a5
        << '\n';

    out << "blah blah "
        << arg_b0
        << ' ' << std::right << std::setw(9) << arg_b1 << arg_b2
        << ' ' << std::noshowpos << std::dec << arg_b3
        << ' ' << std::showpos << std::setw(5) << arg_b4
        << " blah "
        << std::showbase << std::right << std::oct << std::setw(6) << arg_b5
        << ' ' << std::noshowbase << std::hex << arg_b6
        << ' ' << arg_b7
        << '\n';

    out << "blah blah "
        << arg_c0
        << ' ' << std::right << std::setw(10) << arg_c1
        << ' ' << std::noshowpos << std::dec << arg_c2
        << ' ' << std::showpos << std::setw(5) << arg_c3
        << ' ' << arg_c4
        << ' ' << std::showbase << std::right << std::oct << std::setw(6) << arg_c5
        << ' ' << std::noshowbase << std::hex << arg_c6
        << ' ' << arg_c7
        << '\n';
}
