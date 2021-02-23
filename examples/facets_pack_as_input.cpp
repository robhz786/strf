//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>


void sample1()
{
    //[ facets_pack_input
    auto str = strf::to_string .with(strf::numpunct<10>(1))
        ( !strf::dec(10000)
        , "  "
        , !strf::hex(0x10000)
        , strf::with( strf::numpunct<10>(3)
                    , strf::numpunct<16>(4).thousands_sep('\'') )
            ( "  { "
            , !strf::dec(10000)
            , "  "
            , !strf::hex(0x10000)
            , " }" ) );

    assert(str == "1,0,0,0,0  10000  { 10,000  1'0000 }");
    //]

}

void sample2()
{
    //[ facets_pack_input_2
    auto fp = strf::pack
        ( strf::numpunct<10>(3)
        , strf::numpunct<16>(4).thousands_sep('\'') );

    auto str = strf::to_string.with(strf::numpunct<10>(1))
        ( !strf::dec(10000)
        , "  "
        , !strf::hex(0x10000)
        , strf::with(fp)
            ( "  { "
            , !strf::dec(10000)
            , "  "
            , !strf::hex(0x10000)
            , strf::with
                (strf::numpunct<10>(2).thousands_sep('.'))
                  ("  { ", !strf::dec(10000), " }")
            , " }" ) );

    assert(str == "1,0,0,0,0  10000  { 10,000  1'0000  { 1.00.00 } }");
    //]
}

void sample3()
{
    //[ facets_pack_input_in_tr_string
    auto str = strf::to_string
        .tr( "{} -- {} -- {}"
           , "aaa"
           , strf::with()("bbb", "ccc", "ddd")
           , "eee" );

    assert(str == "aaa -- bbbcccddd -- eee");
    //]
}




int main()
{
    sample1();
    sample2();
    sample3();
}
