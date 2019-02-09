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

    const auto name = "Anna";
    const auto age  = 22;

    // without assembly string:
    auto s1 = strf::to_string(name, " is ", age, " years old.");
    BOOST_ASSERT(s1 == "Anna is 22 years old.");

    // with assembly string:
    auto s2 = strf::to_string .as("{} is {} years old.", name, age);
    BOOST_ASSERT(s2 == "Anna is 22 years old.");

    return 0;
}
//]

