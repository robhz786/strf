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
        boost::stringify::writef(buff)()(1243);
        BOOST_TEST(std::string(buff) == "1243");
    }

    
    {
        char buff[200] = "";
        boost::stringify::writef(buff)
            (strf::noshowpos<is_long>, strf::showpos<>)
            ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});
        BOOST_TEST(std::string(buff) == "abcd0+1+23+4");
    }
    
    // {
    //     auto str = strf::make_string
    //         (strf::noshowpos<is_long>, strf::showpos<>)
    //         ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

    //     BOOST_TEST(str == "abcd0+1+23+4");
    // }
    
    // {
    //     std::string str = "qwert";
    //     strf::appendf(str)
    //         (strf::noshowpos<is_long>, strf::showpos<>)
    //         ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

    //     BOOST_TEST(str == "qwertabcd0+1+23+4");
    // }


    // {
    //     std::string str = "qwert";
    //     strf::assignf(str)
    //         (strf::noshowpos<is_long>, strf::showpos<>)
    //         ("abcd", (long)0, 1, 2, {3, "-"}, {(long)4, "+"});

    //     BOOST_TEST(str == "abcd0+1+23+4");
    // }

    
    return  boost::report_errors();
}


