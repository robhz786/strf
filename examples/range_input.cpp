//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>
#include <strf/to_cfile.hpp>
#include <vector>
#include <map>

void sample()
{
    //[ range_sample
    int array[] = { 11, 22, 33 };

    auto str = strf::to_string("[", strf::separated_range(array, ", "), "]");

    assert(str == "[11, 22, 33]");
    //]
}


void sample2()
{
    //[ range_sample_2
    int array[] = { 250, 251, 252 };

    auto str = strf::to_string("[", *strf::hex(strf::separated_range(array, ", ")), "]");

    assert(str == "[0xfa, 0xfb, 0xfc]");
    //]
}

void sample3()
{
    //[ range_sample_3
    int array[] = { 11, 22, 33 };

    auto str = strf::to_string("[", +strf::fmt_separated_range(array, " ;") > 4, "]");

    assert(str == "[ +11 ; +22 ; +33]");
    //]
}

void sample4()
{
    int array[] = { 250, 251, 252 };
    auto str = strf::to_string
        ( "["
        , *strf::fmt_separated_range(array, " / ").fill('.').hex() > 6
        , "]");
    assert(str == "[..0xfa / ..0xfb / ..0xfc]");
}

void sample5()
{
    const std::map<int, const char*> m = {
        {1, "one"},
        {2, "two"},
        {1000, "a thousand"}
    };
    auto f = [](auto p) {
        return strf::join(p.first, " -> '", p.second, '\'');
    };
    auto str = strf::to_string('[', strf::separated_range(m, "; ", f), ']');
    assert(str == "[1 -> 'one'; 2 -> 'two'; 1000 -> 'a thousand']");
}


int main()
{
    sample();
    sample2();
    sample3();
    sample4();
    sample5();
    return 0;
}
