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


int main()
{
    namespace strf = boost::stringify;

    {
        auto str = strf::make_string
            (strf::noshowpos<is_long>, strf::showpos<>)
            ((long)0, 1, 2, {3, "-"}, {(long)4, "+"});
        BOOST_TEST(str == "0+1+23+4");
    }
    
    return  boost::report_errors();
}


