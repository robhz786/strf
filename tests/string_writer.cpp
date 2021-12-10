//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/to_string.hpp>

template <typename CharT>
static void test_string_appender()
{
    const auto tiny_str   = test_utils::make_random_std_string<CharT>(5);
    const auto tiny_str2  = test_utils::make_random_std_string<CharT>(6);
    constexpr std::size_t half_size = strf::destination_space_after_flush / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when nothing is actually appended
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dest(str);
        dest.finish();
        TEST_TRUE(str == tiny_str);
    }

    {   // when nor recycle_buffer() neither do_write() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dest(str);
        dest.write(&tiny_str2[0], tiny_str2.size());
        dest.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }

    {   // when recycle_buffer() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dest(str);

        const std::size_t count0 = dest.buffer_space() > 5 ? dest.buffer_space() - 5 : 0;
        auto first_append = test_utils::make_random_std_string<CharT>(count0);
        dest.write(&first_append[0], count0);
        dest.flush();
        dest.write(&half_str[0], half_str.size());
        dest.flush();
        dest.write(&half_str2[0], half_str2.size());
        dest.finish();

        TEST_TRUE(str == tiny_str + first_append + half_str + half_str2);
    }

    {   // when do_write() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> dest(str);

        const std::size_t count0 = dest.buffer_space() > 5 ? dest.buffer_space() - 5 : 0;
        auto first_append = test_utils::make_random_std_string<CharT>(count0);
        dest.write(&first_append[0], count0);
        dest.write(&half_str[0], half_str.size());
        dest.finish();

        TEST_TRUE(str == tiny_str + first_append + half_str);
    }

    {   // testing strf::append
        std::basic_string<CharT> str = tiny_str;
        strf::append(str) (tiny_str2, half_str);
        const auto last_str = test_utils::make_random_std_string<CharT>
            (strf::destination_space_after_flush - 10);
        strf::append(str) (half_str2, last_str);
        TEST_TRUE(str == tiny_str + tiny_str2 + half_str + half_str2 + last_str);
    }

    {   // testing strf::append(...).reserve(...)
        const auto last_str = test_utils::make_random_std_string<CharT>
            (strf::destination_space_after_flush - 10);

        std::basic_string<CharT> str = half_str;
        strf::append(str)
            .reserve(half_str2.size() + last_str.size() + 100)
            (half_str2, last_str);
        TEST_TRUE(str == half_str + half_str2 + last_str);
        TEST_TRUE(str.capacity() >= half_str.size() + last_str.size() + 100);
    }
}

template <typename CharT>
static void test_string_maker()
{
    const auto tiny_str   = test_utils::make_random_std_string<CharT>(5);
    const auto tiny_str2  = test_utils::make_random_std_string<CharT>(6);
    constexpr std::size_t half_size = strf::destination_space_after_flush / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when nothing is actually written
        strf::basic_string_maker<CharT> dest;
        auto str = dest.finish();
        TEST_TRUE(str.empty());
    }
    {   // when nor recycle_buffer() neither do_write() is called
        strf::basic_string_maker<CharT> dest;
        dest.write(&tiny_str[0], tiny_str.size());
        dest.write(&tiny_str2[0], tiny_str2.size());
        auto str = dest.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }
    {   // when recycle_buffer() is called
        strf::basic_string_maker<CharT> dest;
        const std::size_t count0 = dest.buffer_space() > 5 ? dest.buffer_space() - 5 : 0;
        auto part0 = test_utils::make_random_std_string<CharT>(count0);
        dest.write(&part0[0], count0);
        dest.flush();
        dest.write(&half_str[0], half_str.size());
        dest.flush();
        dest.write(&half_str2[0], half_str2.size());
        auto str = dest.finish();
        TEST_TRUE(str == part0 + half_str + half_str2);
    }
    {   // when do_write() is called
        strf::basic_string_maker<CharT> dest;
        const std::size_t count0 = dest.buffer_space() > 5 ? dest.buffer_space() - 5 : 0;
        auto part0 = test_utils::make_random_std_string<CharT>(count0);
        dest.write(&part0[0], count0);
        dest.write(&half_str[0], half_str.size());
        auto str = dest.finish();
        TEST_TRUE(str == part0 + half_str);
    }
    {   // when recyle() and do_write() is called
        strf::basic_string_maker<CharT> dest;
        auto str0 = test_utils::make_random_std_string<CharT>(15);
        dest.write(&str0[0], str0.size());
        dest.flush();
        auto str1 = test_utils::make_random_std_string<CharT>(dest.buffer_space() + 50);
        dest.write(&str1[0], str1.size());
        auto str = dest.finish();
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
static void test_sized_string_maker()
{
    const auto tiny_str   = test_utils::make_random_std_string<CharT>(5);
    const auto tiny_str2  = test_utils::make_random_std_string<CharT>(6);
    constexpr std::size_t half_size = strf::destination_space_after_flush / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when reserved size is zero and nothing is written
        strf::basic_sized_string_maker<CharT> dest(0);
        auto str = dest.finish();
        TEST_TRUE(str.empty());
    }
    {   // when reserved size is zero but something is written
        strf::basic_sized_string_maker<CharT> dest(0);
        dest.write(&tiny_str[0], tiny_str.size());
        auto str = dest.finish();
        TEST_TRUE(str == tiny_str);
    }
    {   // when nor recycle_buffer() neither do_write() is called
        strf::basic_sized_string_maker<CharT> dest(tiny_str.size() + tiny_str2.size());
        dest.write(&tiny_str[0], tiny_str.size());
        dest.write(&tiny_str2[0], tiny_str2.size());
        auto str = dest.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }
    {   // when nor recycle_buffer() neither do_write() is called
        // and the content is smaller than the reserved size
        strf::basic_sized_string_maker<CharT> dest(tiny_str.size() + tiny_str2.size() + 20);
        dest.write(&tiny_str[0], tiny_str.size());
        dest.write(&tiny_str2[0], tiny_str2.size());
        auto str = dest.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
        TEST_TRUE(str.capacity() >= str.size() + 20);
    }
    {   // when recyle() is called
        strf::basic_sized_string_maker<CharT> dest(20);
        auto str0 = test_utils::make_random_std_string<CharT>(15);
        dest.write(&str0[0], str0.size());
        dest.flush();
        dest.write(&half_str[0], half_str.size());
        dest.flush();
        auto str = dest.finish();
        TEST_TRUE(str == str0 + half_str);
    }
    {   // when do_write() is called
        strf::basic_sized_string_maker<CharT> dest(20);
        auto str0 = test_utils::make_random_std_string<CharT>(100);
        dest.write(&str0[0], str0.size());
        dest.write(&half_str[0], half_str.size());
        auto str = dest.finish();
        TEST_TRUE(str == str0 + half_str);
    }
    {   // test strf::to_basic_string.reseve(...)
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
