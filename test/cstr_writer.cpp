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

    BOOST_TEST(r.truncated);
    BOOST_TEST_EQ(*r.ptr, '\0');
    BOOST_TEST_EQ(r.ptr, &buff[7]);
    BOOST_TEST_CSTR_EQ(buff, "Hello W");
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

    BOOST_TEST(! r1.truncated);
    BOOST_TEST_EQ(*r1.ptr, '\0');
    BOOST_TEST_EQ(r1.ptr, &buff[11]);
    BOOST_TEST_CSTR_EQ(buff, "Hello World");
    BOOST_TEST(! sw.good());

    // write after finish

    strf::write(sw, s2);
    auto r2 = sw.finish();
    BOOST_TEST(! sw.good());
    BOOST_TEST(r2.truncated);
    BOOST_TEST_EQ(*r2.ptr, '\0');
    BOOST_TEST_EQ(r2.ptr, r1.ptr);
}

template <typename CharT>
void test_dispatchers()
{

    const auto half_str = test_utils::make_half_string<CharT>();
    const auto full_str = test_utils::make_full_string<CharT>();
    const auto double_str = test_utils::make_double_string<CharT>();

    {
        constexpr std::size_t buff_size
            = test_utils::full_string_size<CharT>
            + test_utils::half_string_size<CharT> +1;

        CharT buff[buff_size];
        auto res = strf::write(buff) (full_str,  half_str);

        BOOST_TEST(!res.truncated);
        BOOST_TEST(res.ptr == buff + buff_size - 1);
        BOOST_TEST(full_str + half_str == buff);
    }
    {
        constexpr std::size_t buff_size
            = test_utils::full_string_size<CharT>
            + test_utils::half_string_size<CharT> +1;

        CharT buff[buff_size];
        auto res = strf::write(buff, buff_size) (full_str,  half_str);

        BOOST_TEST(!res.truncated);
        BOOST_TEST(res.ptr == buff + buff_size - 1);
        BOOST_TEST(full_str + half_str == buff);
    }
    {
        constexpr std::size_t buff_size
            = test_utils::full_string_size<CharT>
            + test_utils::half_string_size<CharT> +1;

        CharT buff[buff_size];
        auto res = strf::write(buff, buff + buff_size) (full_str,  half_str);

        BOOST_TEST(!res.truncated);
        BOOST_TEST(res.ptr == buff + buff_size - 1);
        BOOST_TEST(full_str + half_str == buff);
    }
}


int main()
{
    test_cstr_writer_destination_too_small();
    test_write_into_cstr_writer_after_finish();

    test_dispatchers<char>();
    test_dispatchers<char16_t>();
    test_dispatchers<char32_t>();
    test_dispatchers<wchar_t>();

    return boost::report_errors();
}
