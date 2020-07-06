//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/to_string.hpp>

template <typename CharT>
static void test_successfull_append()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    std::basic_string<CharT> str;
    strf::basic_string_appender<CharT> ob(str);
    write(ob, tiny_str.begin(), tiny_str.size());
    write(ob, double_str.begin(), double_str.size());
    ob.finish();

    TEST_EQ(str.size(), tiny_str.size() + double_str.size());
    TEST_TRUE(0 == str.compare( 0, tiny_str.size()
                              , tiny_str.begin()
                              , tiny_str.size() ));
    TEST_TRUE(0 == str.compare( tiny_str.size()
                              , double_str.size()
                              , double_str.begin()
                              , double_str.size() ));
}

template <typename CharT>
static void test_successfull_make()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    strf::basic_string_maker<CharT> ob;
    write(ob, tiny_str.begin(), tiny_str.size());
    write(ob, double_str.begin(), double_str.size());
    auto result = ob.finish();

    TEST_EQ(result.size(), tiny_str.size() + double_str.size());
    TEST_TRUE(0 == result.compare( 0, tiny_str.size()
                                 , tiny_str.begin()
                                 , tiny_str.size() ));
    TEST_TRUE(0 == result.compare( tiny_str.size()
                                 , double_str.size()
                                 , double_str.begin()
                                 , double_str.size() ));
}

template <typename CharT>
static void test_destinations()
{
    auto double_str = test_utils::make_double_string<CharT>();
    auto half_str = test_utils::make_half_string<CharT>();
    {
        auto s = strf::to_basic_string<CharT>(half_str);
        TEST_EQ(s.size(), half_str.size());
        TEST_EQ(0, s.compare( 0, half_str.size()
                            , half_str.begin(), half_str.size() ));
    }
    {
        auto s = strf::to_basic_string<CharT>.reserve(20) (half_str);
        TEST_EQ(s.size(), half_str.size());
        TEST_EQ(0, s.compare( 0, half_str.size()
                              , half_str.begin(), half_str.size() ));
    }
    {
        auto s = strf::to_basic_string<CharT>.reserve_calc() (half_str);
        TEST_EQ(s.size(), half_str.size());
        TEST_EQ(0, s.compare( 0, half_str.size()
                            , half_str.begin(), half_str.size() ));
    }
    {
        std::basic_string<CharT> s(double_str.begin(), double_str.size());
        strf::append(s) (half_str);
        TEST_EQ(s.size(), double_str.size() + half_str.size());
        TEST_EQ(0, s.compare( 0, double_str.size()
                            , double_str.begin()
                            , double_str.size() ));
        TEST_EQ(0, s.compare( double_str.size()
                            , half_str.size()
                            , half_str.begin()
                            , half_str.size() ));
    }
    {
        std::basic_string<CharT> s(double_str.begin(), double_str.size());
        strf::append(s).reserve(20) (half_str);
        TEST_EQ(s.size(), double_str.size() + half_str.size());
        TEST_EQ(0, s.compare( 0, double_str.size()
                            , double_str.begin()
                            , double_str.size() ));
        TEST_EQ(0, s.compare( double_str.size()
                            , half_str.size()
                            , half_str.begin()
                            , half_str.size() ));
    }
    {
        std::basic_string<CharT> s(double_str.begin(), double_str.size());
        strf::append(s).reserve_calc() (half_str);
        TEST_EQ(s.size(), double_str.size() + half_str.size());
        TEST_EQ(0, s.compare( 0, double_str.size()
                            , double_str.begin()
                            , double_str.size() ));
        TEST_EQ(0, s.compare( double_str.size()
                            , half_str.size()
                            , half_str.begin()
                            , half_str.size() ));
    }
}


void test_string_writer()
{
    test_destinations<char>();
    test_destinations<char16_t>();

    test_successfull_append<char>();
    test_successfull_append<char16_t>();
    test_successfull_append<char>();
    test_successfull_append<char16_t>();

    test_successfull_make<char>();
    test_successfull_make<char16_t>();
    test_successfull_make<char>();
    test_successfull_make<char16_t>();
}
