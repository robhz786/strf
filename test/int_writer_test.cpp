#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"
#include <limits>

#define TEST testf<__LINE__>

int main()
{
    namespace strf = boost::stringify;
    
    TEST ( "0") () (0);
    TEST (u"0") () (0);
    TEST (U"0") () (0);
    TEST (L"0") () (0);
    
    TEST ( "0") () ((unsigned)0);
    TEST (u"0") () ((unsigned)0);
    TEST (U"0") () ((unsigned)0);
    TEST (L"0") () ((unsigned)0);
   
    TEST ( "123") () (123);
    TEST (u"123") () (123);
    TEST (U"123") () (123);
    TEST (L"123") () (123);

    TEST ( "-123") () (-123);
    TEST (u"-123") () (-123);
    TEST (U"-123") () (-123);
    TEST (L"-123") () (-123);

    TEST ( std::to_string(INT32_MAX).c_str()) () (INT32_MAX);
    TEST (std::to_wstring(INT32_MAX).c_str()) () (INT32_MAX);
    
    TEST ( std::to_string(INT32_MIN).c_str()) () (INT32_MIN);
    TEST (std::to_wstring(INT32_MIN).c_str()) () (INT32_MIN);

    TEST ( std::to_string(UINT32_MAX).c_str()) () (UINT32_MAX);
    TEST (std::to_wstring(UINT32_MAX).c_str()) () (UINT32_MAX);

    {
//        auto f_showpos = strf::make_formating(strf::showpos<>);

        TEST ("+123") (strf::showpos<>) (123);
        TEST ("123")  (strf::showpos<>) ({123, "-"});
        TEST ("+123") ()                ({123, "+"});
        TEST ("-123") (strf::showpos<>) ( -123);
        TEST ("+0")   (strf::showpos<>) (0);
        TEST ("123")  (strf::showpos<>) ((unsigned)123);
        TEST ("123")  (strf::showpos<>) ({(unsigned)123, "+"});
    }
    
    
    return  boost::report_errors();
}






