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
    constexpr std::size_t half_size = strf::min_space_after_recycle<CharT>() / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when nothing is actually appended
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> ob(str);
        ob.finish();
        TEST_TRUE(str == tiny_str);
    }

    {   // when nor recycle() neither do_write() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> ob(str);
        ob.write(&tiny_str2[0], tiny_str2.size());
        ob.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }

    {   // when recycle() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> ob(str);

        const std::size_t count0 = ob.space() > 5 ? ob.space() - 5 : 0;
        auto first_append = test_utils::make_random_std_string<CharT>(count0);
        ob.write(&first_append[0], count0);
        ob.recycle();
        ob.write(&half_str[0], half_str.size());
        ob.recycle();
        ob.write(&half_str2[0], half_str2.size());
        ob.finish();

        TEST_TRUE(str == tiny_str + first_append + half_str + half_str2);
    }

    {   // when do_write() is called
        std::basic_string<CharT> str = tiny_str;
        strf::basic_string_appender<CharT> ob(str);

        const std::size_t count0 = ob.space() > 5 ? ob.space() - 5 : 0;
        auto first_append = test_utils::make_random_std_string<CharT>(count0);
        ob.write(&first_append[0], count0);
        ob.write(&half_str[0], half_str.size());
        ob.finish();

        TEST_TRUE(str == tiny_str + first_append + half_str);
    }

    {   // testing strf::append
        std::basic_string<CharT> str = tiny_str;
        strf::append(str) (tiny_str2, half_str);
        const auto last_str = test_utils::make_random_std_string<CharT>
            (strf::min_space_after_recycle<CharT>() - 10);
        strf::append(str) (half_str2, last_str);
        TEST_TRUE(str == tiny_str + tiny_str2 + half_str + half_str2 + last_str);
    }

    {   // testing strf::append(...).reserve(...)
        const auto last_str = test_utils::make_random_std_string<CharT>
            (strf::min_space_after_recycle<CharT>() - 10);

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
    constexpr std::size_t half_size = strf::min_space_after_recycle<CharT>() / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when nothing is actually written
        strf::basic_string_maker<CharT> ob;
        auto str = ob.finish();
        TEST_TRUE(str.empty());
    }
    {   // when nor recycle() neither do_write() is called
        strf::basic_string_maker<CharT> ob;
        ob.write(&tiny_str[0], tiny_str.size());
        ob.write(&tiny_str2[0], tiny_str2.size());
        auto str = ob.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }
    {   // when recycle() is called
        strf::basic_string_maker<CharT> ob;
        const std::size_t count0 = ob.space() > 5 ? ob.space() - 5 : 0;
        auto part0 = test_utils::make_random_std_string<CharT>(count0);
        ob.write(&part0[0], count0);
        ob.recycle();
        ob.write(&half_str[0], half_str.size());
        ob.recycle();
        ob.write(&half_str2[0], half_str2.size());
        auto str = ob.finish();
        TEST_TRUE(str == part0 + half_str + half_str2);
    }
    {   // when do_write() is called
        strf::basic_string_maker<CharT> ob;
        const std::size_t count0 = ob.space() > 5 ? ob.space() - 5 : 0;
        auto part0 = test_utils::make_random_std_string<CharT>(count0);
        ob.write(&part0[0], count0);
        ob.write(&half_str[0], half_str.size());
        auto str = ob.finish();
        TEST_TRUE(str == part0 + half_str);
    }
    {   // when recyle() and do_write() is called
        strf::basic_string_maker<CharT> ob;
        auto str0 = test_utils::make_random_std_string<CharT>(15);
        ob.write(&str0[0], str0.size());
        ob.recycle();
        auto str1 = test_utils::make_random_std_string<CharT>(ob.space() + 50);
        ob.write(&str1[0], str1.size());
        auto str = ob.finish();
        TEST_TRUE(str == str0 + str1);
    }
    {   // test strf::to_basic_string
        auto str = strf::to_basic_string<CharT>(tiny_str, tiny_str2, half_str, half_str2);
        TEST_TRUE(str == tiny_str + tiny_str2 + half_str + half_str2);
    }
}

template <typename CharT>
static void test_sized_string_maker()
{
    const auto tiny_str   = test_utils::make_random_std_string<CharT>(5);
    const auto tiny_str2  = test_utils::make_random_std_string<CharT>(6);
    constexpr std::size_t half_size = strf::min_space_after_recycle<CharT>() / 2;
    const auto half_str   = test_utils::make_random_std_string<CharT>(half_size);
    const auto half_str2  = test_utils::make_random_std_string<CharT>(half_size);

    {   // when reserved size is zero and nothing is written
        strf::basic_sized_string_maker<CharT> ob(0);
        auto str = ob.finish();
        TEST_TRUE(str.empty());
    }
    {   // when reserved size is zero but something is written
        strf::basic_sized_string_maker<CharT> ob(0);
        ob.write(&tiny_str[0], tiny_str.size());
        auto str = ob.finish();
        TEST_TRUE(str == tiny_str);
    }
    {   // when nor recycle() neither do_write() is called
        strf::basic_sized_string_maker<CharT> ob(tiny_str.size() + tiny_str2.size());
        ob.write(&tiny_str[0], tiny_str.size());
        ob.write(&tiny_str2[0], tiny_str2.size());
        auto str = ob.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
    }
    {   // when nor recycle() neither do_write() is called
        // and the content is smaller than the reserved size
        strf::basic_sized_string_maker<CharT> ob(tiny_str.size() + tiny_str2.size() + 20);
        ob.write(&tiny_str[0], tiny_str.size());
        ob.write(&tiny_str2[0], tiny_str2.size());
        auto str = ob.finish();
        TEST_TRUE(str == tiny_str + tiny_str2);
        TEST_TRUE(str.capacity() >= str.size() + 20);
    }
    {   // when recyle() is called
        strf::basic_sized_string_maker<CharT> ob(20);
        auto str0 = test_utils::make_random_std_string<CharT>(15);
        ob.write(&str0[0], str0.size());
        ob.recycle();
        ob.write(&half_str[0], half_str.size());
        ob.recycle();
        auto str = ob.finish();
        TEST_TRUE(str == str0 + half_str);
    }
    {   // when do_write() is called
        strf::basic_sized_string_maker<CharT> ob(20);
        auto str0 = test_utils::make_random_std_string<CharT>(100);
        ob.write(&str0[0], str0.size());
        ob.write(&half_str[0], half_str.size());
        auto str = ob.finish();
        TEST_TRUE(str == str0 + half_str);
    }
    {   // test strf::to_basic_string.reseve(...)
        auto str = strf::to_basic_string<CharT>.reserve(1000)(tiny_str, tiny_str2);
        TEST_TRUE(str == tiny_str + tiny_str2);
        TEST_TRUE(str.capacity() >= 1000);
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
