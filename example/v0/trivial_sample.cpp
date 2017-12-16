/*=============================================================================
    Use, modification and distribution is subject to the Boost Software
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

//[ trivial_sample
#include <boost/stringify.hpp>
#include <boost/assert.hpp>
#include <cstring>

int main()
{
    namespace strf = boost::stringify::v0;

    char output[80];
    auto leading_expression = strf::write_to(output);
    const auto name = "Anna";
    const auto age  = int{22};
    const auto assembly_string = "{} is {} years old.";
    const auto expected_result = "Anna is 22 years old.";
    
    // with assembly string:
    std::error_code err1 = leading_expression (assembly_string) = {name, age};
    BOOST_ASSERT(!err1 && strcmp(expected_result, output) == 0);
    
    // without assembly string:
    std::error_code err2 = leading_expression = {name, " is ", age, " years old."};
    BOOST_ASSERT(!err2 && strcmp(expected_result, output) == 0);

    // when the leading is not assignable, you can make:
    std::error_code err3 = leading_expression() = {name, " is ", age, " years old."};
    BOOST_ASSERT(!err3 && strcmp(expected_result, output) == 0);
    
    // or:
    std::error_code err4 = leading_expression [{name, " is ", age, " years old."}];
    BOOST_ASSERT(!err4 && strcmp(expected_result, output) == 0);
    
    return 0;
}
//]

