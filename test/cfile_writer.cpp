//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _CRT_SECURE_NO_WARNINGS

#include <ctime>
#include <cstdlib>
#include "test_utils.hpp"

template <typename CharT>
void test_narrow_successfull_writing()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    std::FILE* file = std::tmpfile();
    strf::narrow_cfile_writer<CharT> writer(file);

    write(writer, tiny_str.begin(), tiny_str.size());
    write(writer, double_str.begin(), double_str.size());
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

    write(writer, tiny_str.begin(), tiny_str.size());
    write(writer, double_str.begin(), double_str.size());
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

template <typename CharT>
void test_narrow_failing_to_recycle()
{
    auto half_str = test_utils::make_half_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "w");
    strf::narrow_cfile_writer<CharT> writer(file);

    write(writer, half_str.begin(), half_str.size());
    writer.recycle(); // first recycle shall work
    test_utils::turn_into_bad(writer);
    write(writer, double_str.begin(), double_str.size());

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

void test_wide_failing_to_recycle()
{
    auto half_str = test_utils::make_half_string<wchar_t>();
    auto double_str = test_utils::make_double_string<wchar_t>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "w");
    strf::wide_cfile_writer writer(file);

    write(writer, half_str.begin(), half_str.size());
    writer.recycle();
    test_utils::turn_into_bad(writer);
    write(writer, double_str.begin(), double_str.size());

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
    std::FILE* file = std::fopen(path.c_str(), "w");
    strf::narrow_cfile_writer<CharT> writer(file);

    write(writer, double_str.begin(), double_str.size());
    writer.recycle();
    write(writer, half_str.begin(), half_str.size());
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
    std::FILE* file = std::fopen(path.c_str(), "w");
    strf::wide_cfile_writer writer(file);

    write(writer, double_str.begin(), double_str.size());
    writer.recycle();
    write(writer, half_str.begin(), half_str.size());
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
void test_destination()
{
    auto half_str = test_utils::make_half_string<CharT>();
    auto full_str = test_utils::make_full_string<CharT>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "w");

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

void test_wdestination()
{
    auto half_str = test_utils::make_half_string<wchar_t>();
    auto full_str = test_utils::make_full_string<wchar_t>();

    auto path = test_utils::unique_tmp_file_name();
    std::FILE* file = std::fopen(path.c_str(), "w");

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

int main()
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    test_destination<char>();
    test_destination<char16_t>();
    test_destination<char32_t>();
    test_destination<wchar_t>();

    test_wdestination();

    test_narrow_successfull_writing<char>();
    test_narrow_successfull_writing<char16_t>();
    test_narrow_successfull_writing<char32_t>();
    test_narrow_successfull_writing<wchar_t>();

    test_narrow_failing_to_recycle<char>();
    test_narrow_failing_to_recycle<char16_t>();
    test_narrow_failing_to_recycle<char32_t>();
    test_narrow_failing_to_recycle<wchar_t>();

    test_narrow_failing_to_finish<char>();
    test_narrow_failing_to_finish<char16_t>();
    test_narrow_failing_to_finish<char32_t>();
    test_narrow_failing_to_finish<wchar_t>();

    test_wide_successfull_writing();
    test_wide_failing_to_recycle();
    test_wide_failing_to_finish();

    return test_finish();
}
