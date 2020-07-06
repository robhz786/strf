//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <strf/to_cfile.hpp>

namespace test_utils {

static strf::outbuff*& test_outbuff_ptr()
{
    static strf::outbuff* ptr = nullptr;
    return ptr;
}

void set_test_outbuff(strf::outbuff& ob)
{
    test_outbuff_ptr() = &ob;
}

strf::outbuff& test_outbuff()
{
    auto * ptr = test_outbuff_ptr();
    return *ptr;
}

} // namespace test_utils

void test_join();
void test_cfile_writer();
void test_cstr_writer();
void test_input_bool();
void test_input_char();
void test_input_facets_pack();
void test_input_float();
void test_input_int();
void test_input_range();
void test_input_string();

int main() {
    strf::narrow_cfile_writer<char> test_outbuff(stdout);
    test_utils::set_test_outbuff(test_outbuff);    

    test_join();
    test_cfile_writer();
    test_cstr_writer();
    test_input_bool();
    test_input_char();
    test_input_facets_pack();
    test_input_float();
    test_input_int();
    test_input_range();
    test_input_string();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_outbuff, "All test passed!\n");
    } else {
        strf::to(test_outbuff) (err_count, " tests failed!\n");
    }
    test_outbuff.finish();
    return err_count;
}
