//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

namespace test_utils {

#if ! defined(STRF_FREESTANDING)

std::string unique_tmp_file_name()
{

#if defined(_WIN32)

    char dirname[MAX_PATH];
    GetTempPathA(MAX_PATH, dirname);
    char fullname[MAX_PATH];
    sprintf_s(fullname, MAX_PATH, "%s\\test_boost_outbuff_%x.txt", dirname, std::rand());
    return fullname;

#else // defined(_WIN32)

   char fullname[200];
   sprintf(fullname, "/tmp/test_boost_outbuff_%x.txt", std::rand());
   return fullname;

#endif  // defined(_WIN32)
}

std::wstring read_wfile(std::FILE* file)
{
    std::wstring result;
    wint_t ch = fgetwc(file);
    while(ch != WEOF) {
        result += static_cast<wchar_t>(ch);
        ch = fgetwc(file);
    };

    return result;
}

std::wstring read_wfile(const char* filename)
{
    std::wstring result;

#if defined(_WIN32)

    std::FILE* file = NULL;
    (void) fopen_s(&file, filename, "r");

#else // defined(_WIN32)

    std::FILE* file = std::fopen(filename, "r");

#endif  // defined(_WIN32)

    if(file != nullptr) {
        result = read_wfile(file);
        fclose(file);
    }
    return result;
}

#endif // ! defined(STRF_FREESTANDING)


int& STRF_HD test_err_count()
{
    static int count = 0;
    return count;
}

void STRF_HD print_test_message_header(const char* filename, int line)
{
    test_scope::print_stack(test_outbuff());
    to(test_utils::test_outbuff()) (filename, ':', line, ": ");
}

void STRF_HD print_test_message_end(const char* funcname)
{
    to(test_utils::test_outbuff()) ("\n    In function '", funcname, "'\n");
}

} // namespace test_utils
