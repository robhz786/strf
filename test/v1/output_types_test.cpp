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
    namespace strf = boost::stringify::v1;

   
    {
        char buff[200] = "";

        strf::write_to(buff)
            .with(strf::showpos, strf::noshowpos_if<is_long>())
            [{"abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"}}];

        BOOST_TEST(std::string(buff) == "abcd0+1+23+4");
    }
    
    {
        auto str = strf::make_string
            .with(strf::showpos, strf::noshowpos_if<is_long>())
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "abcd0+1+23+4");
    }

    {
        std::string str = "qwert";
        strf::append_to(str)
            .with(strf::showpos, strf::noshowpos_if<is_long>())
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "qwertabcd0+1+23+4");
    }


    {
        std::string str = "qwert";
        strf::assign_to(str)
            .with(strf::showpos)
            .with(strf::noshowpos_if<is_long>())
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

        BOOST_TEST(str == "abcd0+1+23+4");
    }

    int rc = boost::report_errors();
    return rc;
}


