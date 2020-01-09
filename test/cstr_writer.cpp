//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

void test_cstr_writer_destination_too_small()
{
    char buff[8];
    strf::basic_cstr_writer<char> sw(buff);
    write(sw, "Hello");
    write(sw, " World");
    write(sw, "blah blah blah");
    auto r = sw.finish();

    TEST_TRUE(r.truncated);
    TEST_EQ(*r.ptr, '\0');
    TEST_EQ(r.ptr, &buff[7]);
    TEST_CSTR_EQ(buff, "Hello W");
}

void test_write_into_cstr_writer_after_finish()
{
    const char s1a[] = "Hello";
    const char s1b[] = " World";
    const char s2[] = "Second string content";

    const char expected[]
        = "Hello World\0Second string content\0Third string cont";

    char buff[sizeof(expected)];

    strf::basic_cstr_writer<char> sw(buff);
    strf::write(sw, s1a);
    strf::write(sw, s1b);
    auto r1 = sw.finish();

    // after finish

    TEST_TRUE(! r1.truncated);
    TEST_EQ(*r1.ptr, '\0');
    TEST_EQ(r1.ptr, &buff[11]);
    TEST_CSTR_EQ(buff, "Hello World");
    TEST_TRUE(! sw.good());

    // write after finish

    strf::write(sw, s2);
    auto r2 = sw.finish();
    TEST_TRUE(! sw.good());
    TEST_TRUE(r2.truncated);
    TEST_EQ(*r2.ptr, '\0');
    TEST_EQ(r2.ptr, r1.ptr);
}

template <typename CharT>
void test_destinations()
{

    const auto half_str = test_utils::make_half_string<CharT>();
    const auto full_str = test_utils::make_full_string<CharT>();

    {
        constexpr std::size_t buff_size
            = test_utils::full_string_size<CharT>
            + test_utils::half_string_size<CharT> + 1;

        CharT buff[buff_size];
        auto res = strf::to(buff) (full_str,  half_str);

        TEST_TRUE(!res.truncated);
        TEST_TRUE(res.ptr == buff + buff_size - 1);
        TEST_TRUE(*res.ptr == CharT())
        TEST_TRUE(strf::detail::str_equal( full_str.begin()
                                         , buff
                                         , test_utils::full_string_size<CharT>) );
        TEST_TRUE(strf::detail::str_equal( half_str.begin()
                                         , buff + test_utils::full_string_size<CharT>
                                         , test_utils::half_string_size<CharT>) );
    }
    {
        constexpr std::size_t buff_size
            = test_utils::full_string_size<CharT>
            + test_utils::half_string_size<CharT> + 1;

        CharT buff[buff_size];
        auto res = strf::to(buff, buff_size) (full_str,  half_str);

        TEST_TRUE(!res.truncated);
        TEST_TRUE(res.ptr == buff + buff_size - 1);
        TEST_TRUE(strf::detail::str_equal( full_str.begin()
                                         , buff
                                         , test_utils::full_string_size<CharT>) );
        TEST_TRUE(strf::detail::str_equal( half_str.begin()
                                         , buff + test_utils::full_string_size<CharT>
                                         , test_utils::half_string_size<CharT>) );
    }
    {
        constexpr std::size_t buff_size
            = test_utils::full_string_size<CharT>
            + test_utils::half_string_size<CharT> +1;

        CharT buff[buff_size];
        auto res = strf::to(buff, buff + buff_size) (full_str,  half_str);

        TEST_TRUE(!res.truncated);
        TEST_TRUE(res.ptr == buff + buff_size - 1);
        TEST_TRUE(strf::detail::str_equal( full_str.begin()
                                         , buff
                                         , test_utils::full_string_size<CharT>) );
        TEST_TRUE(strf::detail::str_equal( half_str.begin()
                                         , buff + test_utils::full_string_size<CharT>
                                         , test_utils::half_string_size<CharT>) );
    }
}


int main()
{
    test_cstr_writer_destination_too_small();
    test_write_into_cstr_writer_after_finish();

    test_destinations<char>();
    test_destinations<char16_t>();
    test_destinations<char32_t>();
    test_destinations<wchar_t>();

    return test_finish();
}
