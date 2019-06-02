//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/stringify.hpp>

#if ! defined(__cpp_char8_t)

namespace boost{ namespace stringify{ inline namespace v0{
constexpr auto to_u8string = to_string;
}}}

#endif

namespace strf = boost::stringify::v0;

int main()
{
    {
        //[ trstr_escape_sample
        auto str = strf::to_string.tr("} {{ } {}", "aaa");
        BOOST_ASSERT(str == "} { } aaa");
        //]
    }

    {
        //[ trstr_comment_sample
        auto str = strf::to_string.tr
            ( "You can learn more about python{-the programming language, not the reptile} at {}"
            , "www.python.org" );
        BOOST_ASSERT(str == "You can learn more about python at www.python.org");
        //]
    }


    {
        //[ trstr_positional_arg
        auto str = strf::to_string.tr("{1 a person} likes {0 a food type}.", "sandwich", "Paul");
        BOOST_ASSERT(str == "Paul likes sandwich.");
        //]
    }

    {
        //[ trstr_non_positional_arg
        auto str = strf::to_string.tr("{a person} likes {a food type}.", "Paul", "sandwich");
        BOOST_ASSERT(str == "Paul likes sandwich.");
        //]
    }

    {
        //[ trstr_replace
        auto str = strf::to_u8string
            .tr(u8"{} are {}. {} are {}.", u8"Roses", u8"red", u8"Violets");
        BOOST_ASSERT(str == u8"Roses are red. Violets are \uFFFD.");
        //]
   }
   {
       //[ trstr_omit
       auto str = strf::to_string
           .facets(strf::tr_invalid_arg::ignore)
           .tr("{} are {}. {} are {}.", "Roses", "red", "Violets");
       BOOST_ASSERT(str == "Roses are red. Violets are .");
       //]
   }
   {
       //[ trstr_stop
       bool exception_thrown = false;
       try
       {
           auto str = strf::to_string
               .facets(strf::tr_invalid_arg::stop)
               .tr("{} are {}. {} are {}.", "Roses", "red", "Violets");
       }
       catch(strf::tr_string_syntax_error& x)
       {
            exception_thrown = true;
       }

       BOOST_ASSERT(exception_thrown);
       //]
   }

   return 0;
};
