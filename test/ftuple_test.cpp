#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include <limits>
#include <locale>
#include <sstream>
#include <iostream>
#include <string.h>


template <typename T> struct is_long_or_longlong
    : public std::integral_constant
        < bool
        , std::is_same<long, T>::value || std::is_same<long long, T>::value
        >
{
};

template <typename T> struct is_long :  public std::is_same<long, T>
{
};

template <typename T> struct is_short :  public std::is_same<short, T>
{
};


int main()
{
    namespace strf = boost::stringify;

    
    {
        //first match wins. Hence, more specific formatters shall come first.
        auto fmt_good = strf::make_ftuple(strf::noshowpos<is_long>, strf::showpos<>);
        auto fmt_bad  = strf::make_ftuple(strf::showpos<>, strf::noshowpos<is_long>);
        
        auto str_good = strf::make_string(fmt_good) ((long)1, 2);
        auto str_bad  = strf::make_string(fmt_bad)  ((long)1, 2);
        
        BOOST_TEST(str_good ==  "1+2");
        BOOST_TEST(str_bad  == "+1+2");
    }


    {
        // merge ftuples
        auto fmt1 = strf::make_ftuple
            ( strf::noshowpos<is_long_or_longlong>
            , strf::showpos<>
            );
      
        auto fmt_merged  = strf::make_ftuple
            ( strf::showpos<is_long>
            , fmt1
            , strf::noshowpos<>
            );

        static_assert(is_short<short>::value, "");
        
        auto result = strf::make_string(fmt_merged) ((long)1, (long long)2, 3);

        std::cout << result << std::endl;
        BOOST_TEST(result == "+12+3");
    }

    
    return  boost::report_errors();
}


