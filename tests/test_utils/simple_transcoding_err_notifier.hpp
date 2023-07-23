#ifndef TEST_UTILS_SIMPLE_TRANSCODING_ERR_NOTIFIER_HPP
#define TEST_UTILS_SIMPLE_TRANSCODING_ERR_NOTIFIER_HPP

#include "../test_utils.hpp"

namespace test_utils {

class simple_transcoding_err_notifier: public strf::transcoding_error_notifier
{
public:
    using strf::transcoding_error_notifier::transcoding_error_notifier;

    std::size_t invalid_sequence_calls_count = 0;
    int code_unit_size = 0;
    const char* src_charset_name = "";
    const void* invalid_seq = nullptr;
    std::ptrdiff_t invalid_seq_size = 0;

    std::size_t unsupported_codepoints_calls_count = 0;
    const char* dst_charset_name = "";
    unsigned unsupported_ch32 = 0;

    STRF_HD void invalid_sequence
        ( int p_code_unit_size
        , const char* p_charset_name
        , const void* p_invalid_seq
        , std::ptrdiff_t p_code_units_count ) override
    {
        ++invalid_sequence_calls_count;
        code_unit_size = p_code_unit_size;
        src_charset_name = p_charset_name;
        invalid_seq = p_invalid_seq;
        invalid_seq_size = p_code_units_count;
    }

    STRF_HD void unsupported_codepoint
        ( const char* charset_name
        , unsigned codepoint ) override
    {
        ++unsupported_codepoints_calls_count;
        dst_charset_name = charset_name;
        unsupported_ch32 = codepoint;
    }
};


} // namespace test_utils

#endif // TEST_UTILS_SIMPLE_TRANSCODING_ERR_NOTIFIER_HPP
