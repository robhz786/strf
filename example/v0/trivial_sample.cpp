/*=============================================================================
    Use, modification and distribution is subject to the Boost Software
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

//[ trivial_sample
#include <boost/stringify.hpp>
#include <boost/assert.hpp>

int main()
{
    namespace strf = boost::stringify::v0; // v0 is an inline namespace

    char output[80];
    const auto name = "Anna";
    const auto age  = 22;
    const auto assembly_string = "{} is {} years old.";
    const auto expected_result = std::string{"Anna is 22 years old."};
    
    // with assembly string:
    std::error_code err1 = strf::write_to(output) [assembly_string] = {name, age};
    BOOST_ASSERT(!err1 && expected_result == output);
    
    // without assembly string:
    std::error_code err2 = strf::write_to(output) = {name, " is ", age, " years old."};
    BOOST_ASSERT(!err2 && expected_result == output);

    return 0;
}
//]

