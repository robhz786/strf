//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>

#if ! defined(__cpp_char8_t)

namespace strf{
constexpr auto to_u8string = to_string;
}

#endif

int main()
{
    {
        //[ trstr_escape_sample
        auto str = strf::to_string.tr("} {{x} {{{} {{{}}", "aaa", "bbb");
        assert(str == "} {x} {aaa {bbb}");
        //]
    }

    {
        //[ trstr_comment_sample
        auto str = strf::to_string.tr
            ( "You can learn more about python{-the programming language, not the reptile} at {}"
            , "www.python.org" );
        assert(str == "You can learn more about python at www.python.org");
        //]
    }

    {
        //[ trstr_positional_arg
        auto str = strf::to_string.tr("{1 a person} likes {0 a food type}.", "sandwich", "Paul");
        assert(str == "Paul likes sandwich.");
        //]
    }

    {
        //[ trstr_non_positional_arg
        auto str = strf::to_string.tr("{a person} likes {a food type}.", "Paul", "sandwich");
        assert(str == "Paul likes sandwich.");
        //]
    }

    {
        //[ trstr_replace
        auto str = strf::to_u8string
            .tr(u8"{} are {}. {} are {}.", u8"Roses", u8"red", u8"Violets");
        assert(str == u8"Roses are red. Violets are \uFFFD.");
        //]
   }

   return 0;
};
