//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include <limits>
#include <locale>
#include <sstream>
#include <iostream>
#include <string.h>

template <typename T> struct is_long :  public std::is_same<long, T>
{
};

#define SAMPLE


int main()
{
    namespace strf = boost::stringify;

   
    {
        char buff[200] = "";

        boost::stringify::writef(buff)
            .with(strf::noshowpos_if<is_long>(), strf::showpos)
            [{"abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"}}];

        BOOST_TEST(std::string(buff) == "abcd0+1+23+4");
    }
    
    {
        auto str = strf::make_string
            .with(strf::noshowpos_if<is_long>(), strf::showpos)
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "abcd0+1+23+4");
    }

    {
        std::string str = "qwert";
        strf::appendf(str)
            .with(strf::noshowpos_if<is_long>(), strf::showpos)
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "qwertabcd0+1+23+4");
    }


    {
        std::string str = "qwert";
        strf::assignf(str)
            .with(strf::noshowpos_if<is_long>(), strf::showpos)
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "abcd0+1+23+4");
    }


    return  boost::report_errors();
}


