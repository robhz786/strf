//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

template <typename CharT>
void test_successfull_append()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();
    auto expected_content = tiny_str + double_str;

    std::basic_string<CharT> str;
    strf::basic_string_appender<CharT> ob(str);
    write(ob, tiny_str.c_str(), tiny_str.size());
    write(ob, double_str.c_str(), double_str.size());
    ob.finish();
    BOOST_TEST(str == expected_content);
}

template <typename CharT>
void test_successfull_make()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();
    auto expected_content = tiny_str + double_str;

    strf::basic_string_maker<CharT> ob;
    write(ob, tiny_str.c_str(), tiny_str.size());
    write(ob, double_str.c_str(), double_str.size());
    BOOST_TEST(ob.finish() == expected_content);
}

template <typename CharT>
void test_dispatchers()
{
    auto double_str = test_utils::make_double_string<CharT>();
    auto half_str = test_utils::make_half_string<CharT>();
    {
        auto s = strf::to_basic_string<CharT>(half_str);
        BOOST_TEST(s == half_str);
    }
    {
        auto s = strf::to_basic_string<CharT>.reserve(20) (half_str);
        BOOST_TEST(s == half_str);
    }
    {
        auto s = strf::to_basic_string<CharT>.reserve_calc() (half_str);
        BOOST_TEST(s == half_str);
    }
    {
        auto s = double_str;
        strf::append(s) (half_str);
        BOOST_TEST(s == double_str + half_str);
    }
    {
        auto s = double_str;
        strf::append(s).reserve(20) (half_str);
        BOOST_TEST(s == double_str + half_str);
    }
    {
        auto s = double_str;
        strf::append(s).reserve_calc() (half_str);
        BOOST_TEST(s == double_str + half_str);
    }
}


int main()
{
    test_dispatchers<char>();
    test_dispatchers<char16_t>();

    test_successfull_append<char>();
    test_successfull_append<char16_t>();
    test_successfull_append<char>();
    test_successfull_append<char16_t>();

    test_successfull_make<char>();
    test_successfull_make<char16_t>();
    test_successfull_make<char>();
    test_successfull_make<char16_t>();

    return boost::report_errors();
}
