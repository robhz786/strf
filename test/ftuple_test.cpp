//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>

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
        auto fmt_good = strf::make_ftuple(strf::noshowpos_if<is_long>, strf::showpos);
        auto fmt_bad  = strf::make_ftuple(strf::showpos, strf::noshowpos/*_if<is_long>*/);

        auto str_good = strf::make_string(fmt_good) ((long)1, 2);
        auto str_bad  = strf::make_string(fmt_bad)  ((long)1, 2);

        BOOST_TEST(str_good ==  "1+2");
        BOOST_TEST(str_bad  == "+1+2");
    }

    {
        // merge ftuple

        auto fmt1 = strf::make_ftuple
            ( strf::noshowpos_if<is_long_or_longlong>
            , strf::noshowpos_if<is_short>
            );

        auto result = strf::make_string
            (strf::showpos_if<is_long>, fmt1, strf::showpos)
            ((long)1, (long long)2, (short)3, (int)4);

        std::cout << result << std::endl;
        BOOST_TEST(result == "+123+4");
    }

    {
        // merge 2 ftuple
        auto result = strf::make_string
            ( strf::make_ftuple
                ( strf::showpos_if<is_long>
                , strf::noshowpos_if<is_short>
                )
            , strf::make_ftuple
                ( strf::noshowpos_if<is_long_or_longlong>
                , strf::showpos
                )
            )
            ((long)1, (short)2, (long long)3, (int)4);

        std::cout << result << std::endl;
        BOOST_TEST(result == "+123+4");
    }

    return  boost::report_errors();
}


