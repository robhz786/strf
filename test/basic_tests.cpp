#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"
#include <limits>
#include <locale>
#include <sstream>
#include <iostream>
#include <string.h>

namespace strf = boost::stringify;

void f
    ( char* out
    , const strf::input_base_ref<char, void, strf::formater_tuple<> >& i
    )
{
    i.write(out, {});
}
    


int main()
{
    {
        auto ft = strf::make_formating
             ( strf::noshowpos<std::is_signed>()
             , strf::showpos()
             );
        
        auto sp = ft.get_fmt<int>(strf::ftype_showpos());
        static_assert( ! sp.show(), "");

        auto sp2 = ft.get_fmt<unsigned>(strf::ftype_showpos());
        static_assert( sp2.show(), "");

        //std::cout << "sizeof(ft) ==" << sizeof(ft) << std::endl;
        
    }
    

    {
        char buff[200] = "";
        //boost::stringify::basic_writef<char, std::char_traits<char> >(buff, {}, 12, 34);

        boost::stringify::writef(buff) (strf::showpos<std::is_signed>()) (5, 6, 7, (unsigned)8);
        BOOST_TEST(std::string(buff) == "+5+6+78");

        boost::stringify::writef(buff)
            (strf::noshowpos<std::is_unsigned>(), strf::showpos())
            ((unsigned)0, 1, 2, 3);
        
        BOOST_TEST(std::string(buff) == "0+1+2+3");
        
        boost::stringify::writef<char, to_upper_char_traits<char> >(buff)()("aa", "bb", 12, 34);
        BOOST_TEST(std::string(buff) == "AABB1234");

        strcpy(buff, "--------------------");
        
        boost::stringify::writef<char, to_upper_char_traits<char> >(buff)()[{"aa", "bb", 12, 34}];
        BOOST_TEST(std::string(buff) == "AABB1234");

        strcpy(buff, "--------------------");


        boost::stringify::writef<char, to_upper_char_traits<char> >(buff)[{"aa", "bb", 12, 34}];
        BOOST_TEST(std::string(buff) == "AABB1234");

        
        // BOOST_TEST(strf::lengthf<wchar_t>({}, -12) == 3);
        // std::cout << "---" << strf::lengthf<wchar_t>({}, -12) << "----" << std::endl;
    }
    
    return  boost::report_errors();
}


