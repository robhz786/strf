/*=============================================================================
    Use, modification and distribution is subject to the Boost Software
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

void fff();


#include <boost/stringify.hpp>
#include <boost/assert.hpp>

int main()
{
    namespace strf = boost::stringify::v0; // v0 is an inline namespace

//[ trivial_sample
    int age = 22;
    std::string name = "Anna";
    const std::string expected = "Anna is 22 years old.";
    char buff[80];

    // example with assembly string
    std::error_code err1 = strf::write_to(buff) ("{} is {} years old.") = {name, age};
    BOOST_ASSERT(!err1 && expected == buff);
    
    // and without assembly string
    buff[0] = '\0';
    std::error_code err2 = strf::write_to(buff) = {name, " is ", 22, " years old."};
    BOOST_ASSERT(!err2 && expected == buff);
//]

    {
       
//[ trivial_make_string_sample
        strf::expected_string xstr = strf::make_string("ten = {}, and twenty = {}") = {10, 20};

        BOOST_ASSERT(xstr && *xstr == "ten = 10, and twenty = 20");
//]
    }

    
    {
//[ make_string_is_not_assignable
        /*
          auto xstr1 = strf::make_string = {"blah", "blah", "blah"}; // compilation error
        */
        auto xstr2 = strf::make_string() = {"blah", "blah", "blah"}; // ok

        auto xstr3 = strf::make_string [{"blah", "blah", "blah"}]; // ok
//]
    }

    return 0;
}


