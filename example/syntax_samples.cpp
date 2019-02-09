//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include <boost/assert.hpp>

void reserve()
{
    //[ syntax_reserve
    namespace strf = boost::stringify::v0;  // v0 is an inline namespace

    // reserving a bigger size because there are some further appends:
    auto str = strf::to_string.reserve(500)("blah", "blah");

    // by the way, note that in order to avoid repetition
    // you can store part of the syntax in a variable:
    auto append_to_str = strf::append(str).no_reserve();

    (void) append_to_str("_bleh", "_bleh");
    (void) append_to_str.as("--{}--{}--", "blih", "blih");
    (void) append_to_str("bloh", "bloh");
    (void) append_to_str.reserve_calc()("bluh", "bluh");

    BOOST_ASSERT(str == "blahblah_bleh_bleh--blih--blih--blohblohbluhbluh");
    //]
}


int main()
{
    reserve();

    return 0;
}
