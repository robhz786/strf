#include "fmt/format.h"
#include "args_shuffle.hpp"

void FUNCTION_NAME (std::string& out)
{
    out += fmt::format("{}", ARG(0));
    out += fmt::format("{}{}", ARG(1), ARG(2));
    out += fmt::format("{}{}{}", ARG(3), ARG(4), ARG(5));
    out += fmt::format("{}{}{}{}", ARG(6), ARG(7), ARG(8), ARG(9));
    out += fmt::format("{}{}{}{}{}", ARG(10), ARG(11), ARG(12), ARG(13), ARG(14));
    out += fmt::format("{}{}{}{}{}{}", ARG(15), ARG(16), ARG(17), ARG(18), ARG(19), ARG(20));
}
