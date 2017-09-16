//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include "exception_thrower_arg.hpp"
#include <boost/stringify.hpp>
#include <fstream>

namespace strf = boost::stringify::v0;


template <typename CharT>
std::basic_string<CharT> read_file(const char* filename)
{
    constexpr std::size_t buff_size = 500;
    CharT buff[buff_size];
    std::basic_string<CharT> result;
    {
        std::FILE* file = std::fopen(filename, "r");
        std::size_t read_size = 0;
        do
        {
            read_size = std::fread(buff, sizeof(buff[0]), buff_size, file);
            result.append(buff, read_size);
        }
        while(read_size == buff_size);
        fclose(file);
    }
    return result;
}

int main()
{
    
    {   // narrow char
        
        const char* filename = "stringify_tmptestfile_n.tmp";
        std::string expected;
        std::size_t result_length = 1000;
        {
            std::remove(filename);
            std::FILE* file = fopen(filename, "w");

            std::error_code err = use_all_writing_function_of_output_writer
                ( strf::write_to(file, &result_length)
                , expected );

            fclose(file);
            BOOST_TEST(!err);
        }
        std::string result = read_file<char>(filename);
        
        BOOST_TEST(expected.length() == result_length);
        BOOST_TEST(expected == result);
    }
    
    {   // wide char
        const char* filename = "stringify_tmptestfile_.tmp";
        std::string expected = "abcdyyyyyz";
        std::size_t result_length = 1000;
        {
            std::remove(filename);
            std::FILE* file = fopen(filename, "w");

            std::error_code err = strf::wwrite_to(file, &result_length)
            [{
                L"abcd", {L'x', {"", 0}}, {L'y', {"", 5}}, L'z'
            }];


            fclose(file);
            BOOST_TEST(!err);
        }

        
        std::string result = read_file<char>(filename);
        BOOST_TEST(expected.length() == result_length);
        BOOST_TEST(expected == result);
    }

    {   // Test set_error / narrow char

        const char* filename = "stringify_tmptestfile.tmp";
        std::size_t result_length = 1000;
        std::error_code err_code;
        {
            std::remove(filename);
            std::FILE* file = fopen(filename, "w");
            err_code = strf::write_to(file, &result_length)
                [{"abcd", error_code_emitter_arg, "lkjlj"}];
            fclose(file);
        }

        std::string result = read_file<char>(filename);
        BOOST_TEST(err_code == std::errc::invalid_argument);
        BOOST_TEST(result == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // Test set_error / wide char
       
        const char* filename = "stringify_tmptestfile.tmp";
        std::size_t result_length = 1000;
        std::error_code err_code;
        {
            std::remove(filename);
            std::FILE* file = fopen(filename, "w");

            err_code = strf::wwrite_to(file, &result_length)
                [{L"abcd", error_code_emitter_arg, L"lkjlj"}];
            fclose(file);
        }

        std::string result = read_file<char>(filename);
        BOOST_TEST(err_code == std::errc::invalid_argument);
        BOOST_TEST(result == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // When exception is thrown / narrow char

        const char* filename = "stringify_tmptestfile.tmp";
        std::size_t result_length = 1000;
        {
            std::remove(filename);
            std::FILE* file = fopen(filename, "w");
            try
            {
                strf::write_to(file, &result_length)
                    [{"abcd", exception_thrower_arg, "lkjlj"}];
            }
            catch(...)
            {
            }
            fclose(file);
        }

        std::string result = read_file<char>(filename);
        BOOST_TEST(result == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // When exception is thrown / wide char

        const char* filename = "stringify_tmptestfile.tmp";
        std::size_t result_length = 1000;
        {
            std::remove(filename);
            std::FILE* file = fopen(filename, "w");
            try
            {
                strf::wwrite_to(file, &result_length)
                    [{L"abcd", exception_thrower_arg, L"lkjlj"}];
            }
            catch(...)
            {
            }
            fclose(file);
        }

        std::string result = read_file<char>(filename);
        BOOST_TEST(result == "abcd");
        BOOST_TEST(result_length == 4);
    }

    return boost::report_errors();
}
