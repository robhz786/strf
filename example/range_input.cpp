//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <vector>

void sample()
{
    //[ range_sample
    int array[] = { 11, 22, 33 };

    auto str = strf::to_string("[", strf::range(array, ", "), "]");

    assert(str == "[11, 22, 33]");
    //]
}


void sample2()
{
    //[ range_sample_2
    int array[] = { 250, 251, 252 };

    auto str = strf::to_string("[", ~strf::hex(strf::range(array, ", ")), "]");

    assert(str == "[0xfa, 0xfb, 0xfc]");
    //]
}

void sample3()
{
    //[ range_sample_3
    int array[] = { 11, 22, 33 };

    auto str = strf::to_string("[", +strf::fmt_range(array, " ;") > 4, "]");

    assert(str == "[ +11 ; +22 ; +33]");
    //]
}

void sample4()
{
    
std::vector<int> vec = { 11, 22, 33 };
auto str1 = strf::to_string("[", +strf::fmt_range(vec, " ;") > 4, "]");
assert(str1 == "[ +11 ; +22 ; +33]");

auto str2 = strf::to_string
    ( "["
    , ~strf::fmt_range(vec, " / ").fill('.').hex() > 6,
    " ]");
assert(str2 == "[..0xfa / ..0xfb / ..0xfc]");
}


int main()
{
    sample();
    sample2();
    sample3();
    return 0;
}
