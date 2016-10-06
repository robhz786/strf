#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"
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
        char buff[200] = "";
        boost::stringify::writef(buff) (strf::showpos<std::is_signed>) (5, 6, 7, (unsigned)8);
        BOOST_TEST(std::string(buff) == "+5+6+78");
    }
    {
        char buff[200] = "";
        boost::stringify::writef(buff)
            (strf::noshowpos<is_long>, strf::showpos<>)
            ((long)0, 1, 2, {3, "-"}, {(long)4, "+"});
        BOOST_TEST(std::string(buff) == "0+1+23+4");
    }
    {     
        char buff[200] = "";
        boost::stringify::writef<char, to_upper_char_traits<char> >(buff)()("aa", "bb", 12, 34);
        BOOST_TEST(std::string(buff) == "AABB1234");
    }
    {
        char buff[200] = "";
        boost::stringify::writef<char, to_upper_char_traits<char> >(buff)()[{"aa", "bb", 12, 34}];
        BOOST_TEST(std::string(buff) == "AABB1234");
    }
    {
        char buff[200] = "";
        boost::stringify::writef<char, to_upper_char_traits<char> >(buff)[{"aa", "bb", 12, 34}];
        BOOST_TEST(std::string(buff) == "AABB1234");
    }
        
    // BOOST_TEST(strf::lengthf<wchar_t>({}, -12) == 3);
    // std::cout << "---" << strf::lengthf<wchar_t>({}, -12) << "----" << std::endl;
    
    return  boost::report_errors();
}


