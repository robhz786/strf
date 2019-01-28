//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

int main()
{
    {
        //[ asmstr_escape_sample
        auto str = strf::to_string.as("} {{ } {}", "aaa");
        BOOST_ASSERT(str.value() == "} { } aaa");
        //]
    }

    {
        //[ asmstr_comment_sample
        auto str = strf::to_string.as
            ( "You can learn more about python{-the programming language, not the reptile} at {}"
            , "www.python.org" );
        BOOST_ASSERT(str.value() == "You can learn more about python at www.python.org");
        //]
    }


    {
        //[ asmstr_positional_arg
        auto str = strf::to_string.as("{1 a person} likes {0 a food type}.", "sandwich", "Paul");
        BOOST_ASSERT(str.value() == "Paul likes sandwich.");
        //]
    }

    {
        //[ asmstr_non_positional_arg
        auto str = strf::to_string.as("{a person} likes {a food type}.", "Paul", "sandwich");
        BOOST_ASSERT(str.value() == "Paul likes sandwich.");
        //]
    }

    {
        //[ asmstr_replace
        auto str = strf::to_string
            .as("{} are {}. {} are {}.")
            ("Roses", "red", "Violets");
        BOOST_ASSERT(str.value() == u8"Roses are red. Violets are �.");
        //]
   }
   {
       //[ asmstr_omit
       auto str = strf::to_string
           .facets(strf::asm_invalid_arg::ignore)
           .as("{} are {}. {} are {}.")
           ("Roses", "red", "Violets");
       BOOST_ASSERT(str.value() == u8"Roses are red. Violets are .");
       //]
   }
   {
       //[ asmstr_stop
       auto str = strf::to_string
           .facets(strf::asm_invalid_arg::stop)
           .as("{} are {}. {} are {}.")
           ("Roses", "red", "Violets");
       BOOST_ASSERT(!str);
       BOOST_ASSERT(str.error() == std::make_error_code(std::errc::invalid_argument));
       //]
   }

   return 0;
};
