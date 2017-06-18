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
    namespace strf = boost::stringify::v1; // v1 is an inline namespace

    std::string result = strf::make_string("ten = ", 10, " and twenty = ", 20);
    BOOST_ASSERT(result == "ten = 10 and twenty = 20");
    return 0;
}
//]

