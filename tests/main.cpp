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

void test_tr_string();
void test_dynamic_charset();
void test_encode_char();
void test_encode_fill();
void test_input_bool();
void test_input_char();
void test_input_char32();
void test_input_float();
void test_input_int();
void test_input_ptr();
void test_input_string();
void test_input_facets_pack();
void test_input_range();
void test_miscellaneous();
void test_numpunct();
void test_join();
void test_facets_pack_merge();
void test_facets_pack();
void test_reserve();
void test_width_t();
void test_width_calculator();
void test_utf();
void test_single_byte_encodings();
void test_cstr_writer();
void test_locale();
void test_cfile_writer();
void test_basic_outbuff();
void test_printable_overriding();
void test_streambuf_writer();
void test_string_writer();
void test_to_range();

int main() {
    strf::narrow_cfile_writer<char, 1024> test_outbuff(stdout);
    test_utils::set_test_outbuff(test_outbuff);

#if ! defined(STRF_FREESTANDING)

    test_locale();
    test_cfile_writer();
    test_streambuf_writer();
    test_string_writer();

#endif // ! defined(STRF_FREESTANDING)

    test_basic_outbuff();
    test_cstr_writer();
    test_to_range();
    test_dynamic_charset();
    test_encode_char();
    test_encode_fill();
    test_facets_pack();
    test_facets_pack_merge();
    test_input_bool();
    test_input_char();
    test_input_char32();
    test_input_facets_pack();
    test_input_float();
    test_input_int();
    test_input_ptr();
    test_input_range();
    test_input_string();
    test_join();
    test_miscellaneous();
    test_numpunct();
    test_printable_overriding();
    test_reserve();
    test_single_byte_encodings();
    test_tr_string();
    test_utf();
    test_width_calculator();
    test_width_t();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_outbuff, "All test passed!\n");
    } else {
        strf::to(test_outbuff) (err_count, " tests failed!\n");
    }
    test_outbuff.finish();
    return err_count;
}
