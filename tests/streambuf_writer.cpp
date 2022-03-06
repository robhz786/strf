//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_streambuf.hpp>
#include "test_utils.hpp"
#include <sstream>

namespace {

template <typename CharT>
void test_successfull_writing()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    std::basic_ostringstream<CharT> dest;
    strf::basic_streambuf_writer<CharT> writer(*dest.rdbuf());

    writer.write(tiny_str.begin(), tiny_str.size());
    writer.write(double_str.begin(), double_str.size());
    auto status = writer.finish();
    dest.rdbuf()->pubsync();

    auto obtained_content = dest.str();

    TEST_TRUE(status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, tiny_str.size() + double_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, tiny_str.size()
                                           , tiny_str.begin()
                                           , tiny_str.size() ));
    TEST_TRUE(0 == obtained_content.compare( tiny_str.size(), double_str.size()
                                           , double_str.begin()
                                           , double_str.size() ));
}

template <typename CharT>
void when_finish_is_not_called()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();

    std::basic_ostringstream<CharT> oss;
    {
        strf::basic_streambuf_writer<CharT> writer(oss.rdbuf());
        strf::detail::copy_n<CharT>(tiny_str.begin(), tiny_str.size(), writer.buffer_ptr());
        writer.advance(tiny_str.size());
    }
    auto obtained_content = oss.str();
    TEST_TRUE(0 == obtained_content.compare( 0, tiny_str.size()
                                           , tiny_str.begin()
                                           , tiny_str.size() ));
}

template <typename CharT>
void when_finish_is_not_called_but_state_is_bad_anyway()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();

    std::basic_ostringstream<CharT> oss;
    {
        strf::basic_streambuf_writer<CharT> writer(*oss.rdbuf());
        strf::detail::copy_n(tiny_str.begin(), tiny_str.size(), writer.buffer_ptr());
        writer.advance(tiny_str.size());
        test_utils::turn_into_bad(writer);
    }
    TEST_TRUE(oss.str().empty());
}

template <typename CharT>
void test_failing_to_recycle_buffer()
{
    auto half_str = test_utils::make_half_string<CharT>();

    std::basic_ostringstream<CharT> dest;
    strf::basic_streambuf_writer<CharT> writer(*dest.rdbuf());

    writer.write(half_str.begin(), half_str.size());
    writer.flush(); // first flush() works
    test_utils::turn_into_bad(writer);

    strf::to(writer)(strf::multi(static_cast<CharT>('x'), 10));
    writer.flush(); // this fails

    auto status = writer.finish();
    dest.rdbuf()->pubsync();

    auto obtained_content = dest.str();

    TEST_TRUE(! status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, half_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                           , half_str.begin()
                                           , half_str.size() ));
}

template <typename CharT>
void test_failing_to_call_do_write()
{
    auto half_str = test_utils::make_half_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    std::basic_ostringstream<CharT> dest;
    strf::basic_streambuf_writer<CharT> writer(*dest.rdbuf());

    writer.write(half_str.begin(), half_str.size());
    writer.flush(); // first flush() works
    writer.write(double_str.begin(), double_str.size());
    auto status = writer.finish();
    dest.rdbuf()->pubsync();

    auto obtained_content = dest.str();

    TEST_TRUE(! status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, half_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                           , half_str.begin()
                                           , half_str.size() ));
}



template <typename CharT>
void test_failing_to_finish()
{
    auto double_str = test_utils::make_double_string<CharT>();
    auto half_str = test_utils::make_half_string<CharT>();

    std::basic_ostringstream<CharT> dest;
    strf::basic_streambuf_writer<CharT> writer(*dest.rdbuf());

    writer.write(double_str.begin(), double_str.size());
    writer.flush();
    writer.write(half_str.begin(), half_str.size());
    test_utils::turn_into_bad(writer);

    auto status = writer.finish();
    dest.rdbuf()->pubsync();

    auto obtained_content = dest.str();

    TEST_TRUE(! status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, double_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, double_str.size()
                                           , double_str.begin()
                                           , double_str.size() ));
}

template <typename CharT>
void basic_tests()
{
    auto half_str = test_utils::make_half_string<CharT>();
    auto full_str = test_utils::make_full_string<CharT>();

    {
        std::basic_ostringstream<CharT> dest;
        strf::to(dest.rdbuf()) (half_str, full_str);
        auto obtained_content = dest.str();
        TEST_EQ(obtained_content.size(), half_str.size() + full_str.size());
        TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                               , half_str.begin()
                                               , half_str.size() ));
        TEST_TRUE(0 == obtained_content.compare( half_str.size()
                                               , full_str.size()
                                               , full_str.begin()
                                               , full_str.size() ));
    }
    {
        std::basic_ostringstream<CharT> dest;
        strf::to(*dest.rdbuf()) (half_str, full_str);
        auto obtained_content = dest.str();
        TEST_EQ(obtained_content.size(), half_str.size() + full_str.size());
        TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                               , half_str.begin()
                                               , half_str.size() ));
        TEST_TRUE(0 == obtained_content.compare( half_str.size()
                                               , full_str.size()
                                               , full_str.begin()
                                               , full_str.size() ));
    }
}

} // unnamed namespace

void test_streambuf_writer()
{
    basic_tests<char>();
    basic_tests<char16_t>();
    basic_tests<char32_t>();
    basic_tests<wchar_t>();

    test_successfull_writing<char>();
    test_successfull_writing<char16_t>();
    test_successfull_writing<char32_t>();
    test_successfull_writing<wchar_t>();

    when_finish_is_not_called<char>();;
    when_finish_is_not_called<char16_t>();
    when_finish_is_not_called<char32_t>();
    when_finish_is_not_called<wchar_t>();

    when_finish_is_not_called_but_state_is_bad_anyway<char>();
    when_finish_is_not_called_but_state_is_bad_anyway<char16_t>();
    when_finish_is_not_called_but_state_is_bad_anyway<char32_t>();
    when_finish_is_not_called_but_state_is_bad_anyway<wchar_t>();

    test_failing_to_recycle_buffer<char>();
    test_failing_to_recycle_buffer<char16_t>();
    test_failing_to_recycle_buffer<char32_t>();
    test_failing_to_recycle_buffer<wchar_t>();

    test_failing_to_finish<char>();
    test_failing_to_finish<char16_t>();
    test_failing_to_finish<char32_t>();
    test_failing_to_finish<wchar_t>();
}
