//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _CRT_SECURE_NO_WARNINGS // NOLINT(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)

#include <ctime>
#include <cstdlib>
#if defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#include <strf/to_cfile.hpp>
#include "test_utils.hpp"

template <typename CharT>
void test_narrow_successfull_writing()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    std::FILE* file = std::tmpfile();
    strf::narrow_cfile_writer<CharT, strf::min_destination_buffer_size> writer(file);

    writer.write(tiny_str.begin(), tiny_str.size());
    writer.write(double_str.begin(), double_str.size());
    auto status = writer.finish();
    std::fflush(file);
    std::rewind(file);
    auto obtained_content = test_utils::read_file<CharT>(file);
    std::fclose(file);

    TEST_TRUE(status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, tiny_str.size() + double_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, tiny_str.size()
                                           , tiny_str.begin()
                                           , tiny_str.size() ));
    TEST_TRUE(0 == obtained_content.compare( tiny_str.size()
                                           , double_str.size()
                                           , double_str.begin()
                                           , double_str.size() ));
}

void test_wide_successfull_writing()
{
    auto tiny_str = test_utils::make_tiny_string<wchar_t>();
    auto double_str = test_utils::make_double_string<wchar_t>();

    std::FILE* file = std::tmpfile();
    strf::wide_cfile_writer writer(file);

    writer.write(tiny_str.begin(), tiny_str.size());
    writer.write(double_str.begin(), double_str.size());
    auto status = writer.finish();
    std::fflush(file);
    std::rewind(file);
    auto obtained_content = test_utils::read_wfile(file);
    std::fclose(file);

    TEST_TRUE(status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, tiny_str.size() + double_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, tiny_str.size()
                                           , tiny_str.begin()
                                           , tiny_str.size() ));
    TEST_TRUE(0 == obtained_content.compare( tiny_str.size()
                                           , double_str.size()
                                           , double_str.begin()
                                           , double_str.size() ));
}

struct traits_that_fails {
    STRF_HD traits_that_fails(char* dest, std::size_t dest_size)
        : dest_(dest)
        , dest_end_(dest + dest_size)
    {
    }

    STRF_HD std::size_t write(const char* ptr, std::size_t count) noexcept {
        std::size_t size = dest_end_ - dest_;
        std::size_t c = count <= size ? count : size;
        memcpy(dest_, ptr, c);
        dest_ += c;
        return c;
    }

    char* dest_;
    char* dest_end_;
};

void test_cfile_writer_base()
{
    char buff[strf::min_destination_buffer_size];
    char result_buff[strf::min_destination_buffer_size + 50];

    using tester_t = strf::detail::cfile_writer_base<char, traits_that_fails>;

    {   // fails on flush();
        memset(result_buff, 0, sizeof(result_buff));
        tester_t tester{buff, sizeof(buff), result_buff, 10};

        memcpy(tester.buffer_ptr(), "0123456789abcdef", 16);
        tester.advance(16);
        tester.flush();
        TEST_FALSE(tester.good());

        memcpy(tester.buffer_ptr(), "ABCDEF", 6);
        tester.advance(6);
        tester.flush();

        TEST_STRVIEW_EQ(result_buff, "0123456789", 10);
        TEST_FALSE(tester.good());

        auto r = tester.finish();
        TEST_FALSE(r.success);
        TEST_EQ(r.count, 10);
        TEST_STRVIEW_EQ(result_buff, "0123456789", 10);
    }
    {   // fails on do_write(), in its first call to traits_.write
        memset(result_buff, 0, sizeof(result_buff));
        tester_t tester{buff, sizeof(buff), result_buff, 10};

        strf::to(tester) (strf::multi('x', tester.buffer_space()));
        tester.write("0123456789abcdef", 16);
        TEST_STRVIEW_EQ(result_buff, "xxxxxxxxxx", 10);
        TEST_FALSE(tester.good());

        auto r = tester.finish();
        TEST_FALSE(r.success);
        TEST_EQ(r.count, 10);
        TEST_STRVIEW_EQ(result_buff, "xxxxxxxxxx", 10);
    }
    {   // fails on do_write(), in its second call to traits_.write
        memset(result_buff, 0, sizeof(result_buff));
        tester_t tester{buff, sizeof(buff), result_buff, sizeof(buff) + 10};

        strf::to(tester) (strf::multi('x', sizeof(buff)));
        tester.write("0123456789abcdef", 16);
        TEST_FALSE(tester.good());

        auto r = tester.finish();

        TEST_FALSE(r.success);
        TEST_EQ(r.count, sizeof(buff) + 10);
        TEST_CSTR_EQ(result_buff + sizeof(buff), "0123456789");
    }
    {   // fails on finish()
        memset(result_buff, 0, sizeof(result_buff));
        tester_t tester{buff, sizeof(buff), result_buff, 10};

        memcpy(tester.buffer_ptr(), "0123456789abcdef", 16);
        tester.advance(16);
        auto r = tester.finish();

        TEST_FALSE(r.success);
        TEST_EQ(r.count, 10);
        TEST_STRVIEW_EQ(result_buff, "0123456789", 10);
    }
    {   // succeeds in everything
        memset(result_buff, 0, sizeof(result_buff));
        tester_t tester{buff, sizeof(buff), result_buff, sizeof(result_buff)};

        memcpy(tester.buffer_ptr(), "ABCD", 4);
        tester.advance(4);
        tester.flush();
        strf::to(tester) (strf::multi('x', sizeof(buff)));
        tester.write("0123456789abcdef", 16);
        auto r = tester.finish();

        TEST_TRUE(r.success);

        {
            char expected[sizeof(result_buff)];
            auto r2 = strf::to(expected)
                ("ABCD", strf::multi('x', sizeof(buff)), "0123456789abcdef");
            TEST_EQ(r.count, size_t(r2.ptr - expected));
            TEST_STRVIEW_EQ(result_buff, expected, r.count);
        }
    }
    {   // when finish() is not called
        memset(result_buff, 0, sizeof(result_buff));
        {
            tester_t tester{buff, sizeof(buff), result_buff, sizeof(result_buff)};
            memcpy(tester.buffer_ptr(), "ABCD", 4);
            tester.advance(4);
        }
        // the destructor shall flush the content left in the buffer
        TEST_CSTR_EQ(result_buff, "ABCD");
    }
}


template <typename CharT>
void test_narrow_failing_to_flush()
{
    auto half_str = test_utils::make_half_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "we");
    strf::narrow_cfile_writer<CharT, strf::min_destination_buffer_size> writer(file);

    writer.write(half_str.begin(), half_str.size());
    writer.flush(); // first flush shall work
    test_utils::turn_into_bad(writer);
    writer.write(double_str.begin(), double_str.size());

    auto status = writer.finish();
    std::fclose(file);
    auto obtained_content = test_utils::read_file<CharT>(path.c_str());
    std::remove(path.c_str());

    TEST_TRUE(! status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, half_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                           , half_str.begin()
                                           , half_str.size() ));
}

void test_wide_failing_to_flush()
{
    auto half_str = test_utils::make_half_string<wchar_t>();
    auto double_str = test_utils::make_double_string<wchar_t>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "we");
    strf::wide_cfile_writer writer(file);

    writer.write(half_str.begin(), half_str.size());
    writer.flush();
    test_utils::turn_into_bad(writer);
    writer.write(double_str.begin(), double_str.size());

    auto status = writer.finish();
    std::fclose(file);
    auto obtained_content = test_utils::read_wfile(path.c_str());
    std::remove(path.c_str());

    TEST_TRUE(! status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, half_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                           , half_str.begin()
                                           , half_str.size() ));
}


template <typename CharT>
void test_narrow_failing_to_finish()
{
    auto double_str = test_utils::make_double_string<CharT>();
    auto half_str = test_utils::make_half_string<CharT>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "we");
    strf::narrow_cfile_writer<CharT, strf::min_destination_buffer_size> writer(file);

    writer.write(double_str.begin(), double_str.size());
    writer.flush();
    writer.write(half_str.begin(), half_str.size());
    test_utils::turn_into_bad(writer);

    auto status = writer.finish();
    std::fclose(file);
    auto obtained_content = test_utils::read_file<CharT>(path.c_str());
    std::remove(path.c_str());

    TEST_TRUE(! status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, double_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, double_str.size()
                                           , double_str.begin()
                                           , double_str.size() ));
}

void test_wide_failing_to_finish()
{
    auto double_str = test_utils::make_double_string<wchar_t>();
    auto half_str = test_utils::make_half_string<wchar_t>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "we");
    strf::wide_cfile_writer writer(file);

    writer.write(double_str.begin(), double_str.size());
    writer.flush();
    writer.write(half_str.begin(), half_str.size());
    test_utils::turn_into_bad(writer);

    auto status = writer.finish();
    std::fclose(file);
    auto obtained_content = test_utils::read_file<char>(path.c_str());
    std::remove(path.c_str());

    TEST_TRUE(! status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, double_str.size());
    auto narrow_double_str = test_utils::make_double_string<char>();
    TEST_TRUE(0 == obtained_content.compare( 0, narrow_double_str.size()
                                           , narrow_double_str.begin()
                                           , narrow_double_str.size() ));
}

template <typename CharT>
void test_narrow_cfile_writer_creator()
{
    auto half_str = test_utils::make_half_string<CharT>();
    auto full_str = test_utils::make_full_string<CharT>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "we");

    auto status = strf::to<CharT>(file)(half_str, full_str);
    std::fclose(file);
    auto obtained_content = test_utils::read_file<CharT>(path.c_str());
    std::remove(path.c_str());

    TEST_TRUE(status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, half_str.size() + full_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                           , half_str.begin()
                                           , half_str.size() ));
    TEST_TRUE(0 == obtained_content.compare( half_str.size()
                                           , full_str.size()
                                           , full_str.begin()
                                           , full_str.size() ));

}

void test_wide_cfile_writer_creator()
{
    auto half_str = test_utils::make_half_string<wchar_t>();
    auto full_str = test_utils::make_full_string<wchar_t>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "we");

    auto status = strf::wto(file)(half_str, full_str);
    std::fclose(file);

    auto obtained_content = test_utils::read_wfile(path.c_str());
    std::remove(path.c_str());

    TEST_TRUE(status.success);
    TEST_EQ(status.count, obtained_content.size());
    TEST_EQ(status.count, half_str.size() + full_str.size());
    TEST_TRUE(0 == obtained_content.compare( 0, half_str.size()
                                           , half_str.begin()
                                           , half_str.size() ));
    TEST_TRUE(0 == obtained_content.compare( half_str.size()
                                           , full_str.size()
                                           , full_str.begin()
                                           , full_str.size() ));
}

void test_cfile_writer()
{
    // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    test_cfile_writer_base();

    test_narrow_cfile_writer_creator<char>();
    test_narrow_cfile_writer_creator<char16_t>();
    test_narrow_cfile_writer_creator<char32_t>();
    test_narrow_cfile_writer_creator<wchar_t>();
    test_wide_cfile_writer_creator();

    test_narrow_successfull_writing<char>();
    test_narrow_successfull_writing<char16_t>();
    test_narrow_successfull_writing<char32_t>();
    test_narrow_successfull_writing<wchar_t>();

    test_narrow_failing_to_flush<char>();
    test_narrow_failing_to_flush<char16_t>();
    test_narrow_failing_to_flush<char32_t>();
    test_narrow_failing_to_flush<wchar_t>();

    test_narrow_failing_to_finish<char>();
    test_narrow_failing_to_finish<char16_t>();
    test_narrow_failing_to_finish<char32_t>();
    test_narrow_failing_to_finish<wchar_t>();

    test_wide_successfull_writing();
    test_wide_failing_to_flush();
    test_wide_failing_to_finish();
}
