//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>

void sample_hex()
{
    //[ trivial_hex_sample
    auto str = strf::to_string(255, "  ", strf::hex(255));

    assert(str == "255  ff");
    //]
}


void samples()
{
    //[ formatting_samples
    auto str = strf::to_string
        ( strf::hex(255) > 5
        , '/', strf::center(255, 7, '.').hex()
        , '/', ~strf::hex(255) % 7
        , '/', strf::multi('a', 3) ^ 7
        , '/', +strf::fmt(255) );

    assert(str == "   ff/..ff.../0x   ff/  aaa  /+255");
    //]
}


int main()
{
    sample_hex();
    samples();

    return 0;
}
