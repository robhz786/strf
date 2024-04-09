//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/to_string.hpp>

namespace {

template <typename CharT>
void test_string_appender()
{
    const auto tiny_str   = test_utils::make_random_std_string<CharT>(5);
    const auto tiny_str2  = test_utils::make_random_std_string<CharT>(6);
    constexpr std::size_t half_size = strf::min_destination_buffer_size / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when nothing is actually appended
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dst(str);
        dst.finish();
        TEST_TRUE(str == tiny_str);
    }

    {   // when nor recycle() neither do_write() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dst(str);
        dst.write(&tiny_str2[0], tiny_str2.size());
        dst.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }

    {   // when recycle() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dst(str);

        const std::size_t count0 = dst.buffer_space() > 5 ? dst.buffer_space() - 5 : 0;
        auto first_append = test_utils::make_random_std_string<CharT>(count0);
        dst.write(&first_append[0], count0);
        dst.flush();
        dst.write(&half_str[0], half_str.size());
        dst.flush();
        dst.write(&half_str2[0], half_str2.size());
        dst.finish();

        TEST_TRUE(str == tiny_str + first_append + half_str + half_str2);
    }

    {   // when do_write() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dst(str);

        const std::size_t count0 = dst.buffer_space() > 5 ? dst.buffer_space() - 5 : 0;
        auto first_append = test_utils::make_random_std_string<CharT>(count0);
        dst.write(&first_append[0], count0);
        dst.write(&half_str[0], half_str.size());
        dst.finish();

        TEST_TRUE(str == tiny_str + first_append + half_str);
    }
    {   // writing stuff after finish() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dst(str);

        to(dst) (half_str);
        dst.finish();
        TEST_FALSE(dst.good());
        to(dst) (half_str2); // should have no effect
        dst.finish();
        TEST_TRUE(str == tiny_str + half_str);
    }
    {   // testing dtor when finish() has not been called
        std::basic_string<CharT> str = tiny_str;
        {
            strf::basic_string_appender<CharT> dst(str);
            dst.write(&tiny_str2[0], tiny_str2.size());
        }
        TEST_TRUE(str == tiny_str + tiny_str2);
    }
    {   // testing dtor when finish() has been called
        std::basic_string<CharT> str = tiny_str;
        {
            strf::basic_string_appender<CharT> dst(str);
            to(dst) (tiny_str2);
            dst.finish();
            to(dst) (tiny_str); // should have no effect
        }
        TEST_TRUE(str == tiny_str + tiny_str2);
    }

    {   // testing strf::append
        std::basic_string<CharT> str = tiny_str;
        strf::append(str) (tiny_str2, half_str);
        const auto last_str = test_utils::make_random_std_string<CharT>
            (strf::min_destination_buffer_size - 10);
        strf::append(str) (half_str2, last_str);
        TEST_TRUE(str == tiny_str + tiny_str2 + half_str + half_str2 + last_str);
    }

    {   // testing strf::append(...).reserve(...)
        const auto last_str = test_utils::make_random_std_string<CharT>
            (strf::min_destination_buffer_size - 10);

        std::basic_string<CharT> str = half_str;
        strf::append(str)
            .reserve(half_str2.size() + last_str.size() + 100)
            (half_str2, last_str);
        TEST_TRUE(str == half_str + half_str2 + last_str);
        TEST_TRUE(str.capacity() >= half_str.size() + last_str.size() + 100);
    }
}

template <typename CharT>
void test_string_maker()
{
    const auto tiny_str   = test_utils::make_random_std_string<CharT>(5);
    const auto tiny_str2  = test_utils::make_random_std_string<CharT>(6);
    constexpr std::size_t half_size = strf::min_destination_buffer_size / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when nothing is actually written
        strf::basic_string_maker<CharT> dst;
        auto str = dst.finish();
        TEST_TRUE(str.empty());
    }
    {   // when nor recycle() neither do_write() is called
        strf::basic_string_maker<CharT> dst;
        dst.write(&tiny_str[0], tiny_str.size());
        dst.write(&tiny_str2[0], tiny_str2.size());
        auto str = dst.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }
    {   // when recycle() is called
        strf::basic_string_maker<CharT> dst;
        const std::size_t count0 = dst.buffer_space() > 5 ? dst.buffer_space() - 5 : 0;
        auto part0 = test_utils::make_random_std_string<CharT>(count0);
        dst.write(&part0[0], count0);
        dst.flush();
        dst.write(&half_str[0], half_str.size());
        dst.flush();
        dst.write(&half_str2[0], half_str2.size());
        auto str = dst.finish();
        TEST_TRUE(str == part0 + half_str + half_str2);
    }
    {   // when do_write() is called
        strf::basic_string_maker<CharT> dst;
        const std::size_t count0 = dst.buffer_space() > 5 ? dst.buffer_space() - 5 : 0;
        auto part0 = test_utils::make_random_std_string<CharT>(count0);
        dst.write(&part0[0], count0);
        dst.write(&half_str[0], half_str.size());
        auto str = dst.finish();
        TEST_TRUE(str == part0 + half_str);
    }
    {   // when recyle() and do_write() is called
        strf::basic_string_maker<CharT> dst;
        auto str0 = test_utils::make_random_std_string<CharT>(15);
        dst.write(&str0[0], str0.size());
        dst.flush();
        auto str1 = test_utils::make_random_std_string<CharT>(dst.buffer_space() + 50);
        dst.write(&str1[0], str1.size());
        auto str = dst.finish();
        TEST_TRUE(str == str0 + str1);
    }


    {   // test strf::to_basic_string
#if defined(STRF_HAS_VARIABLE_TEMPLATES)

        auto str = strf::to_basic_string<CharT>(tiny_str, tiny_str2, half_str, half_str2);
        TEST_TRUE(str == tiny_str + tiny_str2 + half_str + half_str2);

#endif // defined(STRF_HAS_VARIABLE_TEMPLATES)
    }
}

template <typename CharT>
void test_sized_string_maker()
{
    const auto tiny_str   = test_utils::make_random_std_string<CharT>(5);
    const auto tiny_str2  = test_utils::make_random_std_string<CharT>(6);
    constexpr std::size_t half_size = strf::min_destination_buffer_size / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when reserved size is zero and nothing is written
        strf::basic_sized_string_maker<CharT> dst(0);
        auto str = dst.finish();
        TEST_TRUE(str.empty());
    }
    {   // when reserved size is zero but something is written
        strf::basic_sized_string_maker<CharT> dst(0);
        dst.write(&tiny_str[0], tiny_str.size());
        auto str = dst.finish();
        TEST_TRUE(str == tiny_str);
    }
    {   // when nor recycle() neither do_write() is called
        strf::basic_sized_string_maker<CharT> dst(tiny_str.size() + tiny_str2.size());
        dst.write(&tiny_str[0], tiny_str.size());
        dst.write(&tiny_str2[0], tiny_str2.size());
        auto str = dst.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }
    {   // when nor recycle() neither do_write() is called
        // and the content is smaller than the reserved size
        strf::basic_sized_string_maker<CharT> dst(tiny_str.size() + tiny_str2.size() + 20);
        dst.write(&tiny_str[0], tiny_str.size());
        dst.write(&tiny_str2[0], tiny_str2.size());
        auto str = dst.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
        TEST_TRUE(str.capacity() >= str.size() + 20);
    }
    {   // when recyle() is called
        strf::basic_sized_string_maker<CharT> dst(20);
        auto str0 = test_utils::make_random_std_string<CharT>(15);
        dst.write(&str0[0], str0.size());
        dst.flush();
        dst.write(&half_str[0], half_str.size());
        dst.flush();
        auto str = dst.finish();
        TEST_TRUE(str == str0 + half_str);
    }
    {   // when do_write() is called
        strf::basic_sized_string_maker<CharT> dst(20);
        auto str0 = test_utils::make_random_std_string<CharT>(100);
        dst.write(&str0[0], str0.size());
        dst.write(&half_str[0], half_str.size());
        auto str = dst.finish();
        TEST_TRUE(str == str0 + half_str);
    }
    {   // test strf::to_basic_string.reserve(...)
#if defined(STRF_HAS_VARIABLE_TEMPLATES)

        auto str = strf::to_basic_string<CharT>.reserve(1000)(tiny_str, tiny_str2);
        TEST_TRUE(str == tiny_str + tiny_str2);
        TEST_TRUE(str.capacity() >= 1000);

#endif // defined(STRF_HAS_VARIABLE_TEMPLATES)
    }
}


void test_string_writer()
{
    test_string_appender<char>();
    test_string_appender<char16_t>();

    test_string_maker<char>();
    test_string_maker<char16_t>();

    test_sized_string_maker<char>();
    test_sized_string_maker<char16_t>();
}

} // namespace

REGISTER_STRF_TEST(test_string_writer)

