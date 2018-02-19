//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>


void sample1()
{
    //[ ftuple_input
    namespace strf = boost::stringify::v0;

    auto ft = strf::make_ftuple
        ( strf::monotonic_grouping<10>(3)
        , strf::monotonic_grouping<16>(4).thousands_sep('\'')
        );

    auto result = strf::make_string.with(strf::monotonic_grouping<10>(1)) &=
    {   10000
    ,   "  "
    ,   strf::hex(0x10000)
    ,   {   ft
        ,   {   "  { "
            ,   10000
            ,   "  "
            ,   strf::hex(0x10000)
            ,   " }"
            }
        }
    };

    BOOST_ASSERT(result == "1,0,0,0,0  10000  { 10,000  1'0000 }");
    //]

}

void sample2()
{
    //[ ftuple_input_2
    namespace strf = boost::stringify::v0;

    auto ft = strf::make_ftuple
        ( strf::monotonic_grouping<10>(3)
        , strf::monotonic_grouping<16>(4).thousands_sep('\'')
        );

    auto result = strf::make_string.with(strf::monotonic_grouping<10>(1)) &=
    {   10000
    ,   "  "
    ,   strf::hex(0x10000)
    ,   {   ft
        ,   {   "  { "
            ,   10000
            ,   "  "
            ,   strf::hex(0x10000)
            ,   {   strf::make_ftuple(strf::monotonic_grouping<10>(2).thousands_sep('.'))
                ,   { "  { "
                    , 10000
                    , " }"
                    }
                }
            ,   " }"
            }
        }
    };

    BOOST_ASSERT(result == "1,0,0,0,0  10000  { 10,000  1'0000  { 1.00.00 } }");
    //]
}

void sample3()
{
    //[ ftuple_input_in_assembly_string
    namespace strf = boost::stringify::v0;
    auto result = strf::make_string["{} -- {} -- {}"] &=
    {   "aaa"
    ,   {strf::make_ftuple(), {"bbb", "ccc", "ddd"}}
    ,   "eee"    
    };

    BOOST_ASSERT(result == "aaa -- bbbcccddd -- eee");
    //]
}




int main()
{
    sample1();
    sample2();
    sample3();
}
