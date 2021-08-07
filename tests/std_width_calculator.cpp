//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

// https://unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries

namespace {


template <typename T, std::size_t N>
struct simple_array {
    T elements[N];
};

template <std::size_t N>
using size_array  = simple_array<std::size_t, N>;

template <typename T>
class span {
public:

    using const_iterator = T*;
    using iterator = T*;

    template <typename U, std::size_t N>
    STRF_HD span(simple_array<U, N>& arr)
        : begin_(&arr.elements[0])
        , size_(N)
    {
    }
    STRF_HD span(T* ptr, std::size_t s)
        : begin_(ptr)
        , size_(s)
    {
    }
    STRF_HD span(T* b, T* e)
        : begin_(b)
        , size_(e - b)
    {
    }

    span(const span&) = default;

    STRF_HD T* begin() const { return begin_; }
    STRF_HD T* end()   const { return begin_ + size_; }

    STRF_HD std::size_t size() const { return size_; }

private:
    T* begin_;
    std::size_t size_;
};

template <typename T, typename U>
STRF_HD bool operator==(const span<T>& l, const span<U>& r) noexcept
{
    if (l.size() != r.size())
        return false;

    for (std::size_t i = 0; i < l.size(); ++i) {
        if (l.begin()[i] != r.begin()[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, typename U>
STRF_HD bool operator!=(const span<T>& l, const span<U>& r) noexcept
{
    return ! (l == r);
}

using char32_range = strf::detail::simple_string_view<char32_t>;

class codepoint_samples {
public:
    STRF_HD codepoint_samples(const codepoint_samples&) = delete;

    STRF_HD codepoint_samples(char32_range r, const char* id)
        : range_(r)
        , id_(id)
    {
    }
    STRF_HD char32_t get_a_sample() {
        char32_t ch;
        if (index_ == range_.size()) {
            ch = range_.data()[0];
            index_ = 1;
        } else {
            ch = range_.data()[index_];
            ++index_;
        }
        return ch;
    }
    STRF_HD std::size_t count() const {
        return range_.size();
    }
    STRF_HD const char* id() const {
        return id_;
    };

private:
    std::size_t index_ = 0;
    char32_range range_;
    const char* id_;
};

STRF_HD auto hex_ch32(char32_t ch)
    -> decltype(strf::join("", strf::hex(0u).p(6)))
{
    return strf::join("U+", strf::hex((unsigned)ch).p(4));
}

STRF_HD void test_width_char_by_char
    ( const char* filename
    , int line
    , const char* funcname
    , int expected_width
    , std::size_t* size_buffer
    , span<const std::size_t> expected_grapheme_sizes
    , span<const char32_t> chars )
{
    const auto initial_width = (strf::width_t::max)();
    auto remaining_width = initial_width;
    unsigned state = 0;
    auto obtained_sizes_it = size_buffer;
    const char32_t* grapheme_begin = chars.begin();
    for(auto ptr = chars.begin(); ptr < chars.end(); ++ptr) {
        auto r = strf::detail::std_width_calc_func(ptr, ptr + 1, remaining_width, state, false);
        if (ptr == chars.begin()) {
            remaining_width = r.width;
        } else if (r.width != remaining_width) {
            *obtained_sizes_it++ = ptr - grapheme_begin;
            grapheme_begin = ptr;
            remaining_width = r.width;
        }
        state = r.state;
    }
    *obtained_sizes_it++ = chars.end() - grapheme_begin;

    auto obtained_width = (initial_width - remaining_width).round();
    span<const std::size_t> obtained_grapheme_sizes{size_buffer, obtained_sizes_it};

    if (obtained_width != expected_width) {
        test_utils::test_failure
            ( filename, line, funcname
            , "When calling std_width_calc_func for each sub-string:\n"
            , "    Obtained width = ", obtained_width
            , " (expected ", expected_width, ")\n sequence = "
            , strf::separated_range(chars, " ", hex_ch32) );
    }

    if (obtained_grapheme_sizes != expected_grapheme_sizes) {
        test_utils::test_failure
            ( filename, line, funcname
            , "When calling std_width_calc_func for each sub-string:\n"
            , "    Obtained grapheme sizes : "
            , strf::separated_range(obtained_grapheme_sizes, ", ")
            , "\n    Expected                : "
            , strf::separated_range(expected_grapheme_sizes, ", ") );
    }
}

STRF_HD void test_width_one_pass
    ( const char* filename
    , int line
    , const char* funcname
    , int expected_width
    , span<const std::size_t> // expected_grapheme_sizes
    , span<const char32_t> chars )
{
    const strf::width_t minuend = (strf::width_t::max)();
    auto r = strf::detail::std_width_calc_func(chars.begin(), chars.end(), minuend, 0, false);
    auto obtained_width = (minuend - r.width).round();

    if (obtained_width != expected_width) {
        test_utils::test_failure
            ( filename, line, funcname, "Obtained width = ", obtained_width
            , " (expected ", expected_width, ")\n sequence = "
            , strf::separated_range(chars, " ", hex_ch32) );
    }
}

STRF_HD void do_test_pos
    ( const char* filename
    , int line
    , const char* funcname
    , std::size_t expected_pos
    , strf::width_t max_width
    , span<const char32_t> chars )
{
    auto r = strf::detail::std_width_calc_func(chars.begin(), chars.end(), max_width, 0, true);
    std::size_t obtained_pos = r.ptr - chars.begin();
    if (obtained_pos != expected_pos) {
        test_utils::test_failure
            ( filename, line, funcname, "Obtained pos = ", obtained_pos
            , " (expected ", expected_pos, ")\n sequence = "
            , strf::separated_range(chars, " ", hex_ch32) );
    }
}


STRF_HD void generate_samples(char32_t*) {
}

template <typename CharSampleProvider0, typename... Others>
STRF_HD void generate_samples(char32_t* dest, CharSampleProvider0& arg0, Others&... args) {
    *dest = arg0.get_a_sample();
    generate_samples(dest + 1, args...);
}

template <typename... CharSampleProviders, std::size_t NumSizes>
STRF_HD void test_width
    ( const char* filename
    , int linenumber
    , const char* funcname
    , int expected_width
    , const size_array<NumSizes>& expected_grapheme_sizes_
    , CharSampleProviders&... sample_providers )
{
    char32_t buffer[sizeof...(sample_providers)];
    generate_samples(buffer, sample_providers...);
    span<const std::size_t> expected_grapheme_sizes{expected_grapheme_sizes_.elements, NumSizes};
    span<const char32_t> chars{buffer, sizeof...(sample_providers)};

    test_width_one_pass( filename, linenumber, funcname, expected_width
                       , expected_grapheme_sizes
                       , chars );
    std::size_t size_buff[sizeof...(sample_providers)];
    test_width_char_by_char( filename, linenumber, funcname, expected_width
                           , size_buff
                           , expected_grapheme_sizes
                           , chars );
}

template <typename... CharSampleProviders>
STRF_HD void test_pos
    ( const char* filename
    , int linenumber
    , const char* funcname
    , std::size_t expected_pos
    , strf::width_t max_width
    , CharSampleProviders&... sample_providers )
{
    char32_t buffer[sizeof...(sample_providers)];
    generate_samples(buffer, sample_providers...);
    span<const char32_t> chars{buffer, sizeof...(sample_providers)};
    do_test_pos(filename, linenumber, funcname, expected_pos, max_width, chars);
}

#define TEST_WIDTH(EXPECTED_WIDTH, EXPECTED_GRAPHEME_SIZES, ...)          \
    test_width( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
              , (EXPECTED_WIDTH)                                          \
              , EXPECTED_GRAPHEME_SIZES                                   \
              , __VA_ARGS__ );

#define TEST_POS(EXPECTED_POS, MAX_WIDTH, ...) \
    test_pos(__FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (EXPECTED_POS), (MAX_WIDTH), __VA_ARGS__);

STRF_HD char32_range range_of_control_w1() noexcept {
    static const char32_t samples[] = {
        0x00001F, 0x00009F, 0x0000AD, 0x00061C, 0x00180E, 0x00200B,
        0x00200F, 0x00202E, 0x00206F, 0x00FEFF, 0x00FFFB, 0x013438,
        0x01BCA3, 0x01D17A, 0x0E001F, 0x0E00FF, 0x0E0FFF
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_cr_w1() noexcept {
    static char32_t ch = 0x000D;
    return {&ch, 1};
}
STRF_HD char32_range range_of_lf_w1() noexcept {
    static char32_t ch = 0x000A;
    return {&ch, 1};
}
STRF_HD char32_range range_of_prepend_w1() noexcept {
    static const char32_t samples[] = {
        0x000605, 0x0006DD, 0x00070F, 0x0008E2, 0x000D4E, 0x0110BD,
        0x0110CD, 0x0111C3, 0x01193F, 0x011941, 0x011A3A, 0x011A89,
        0x011D46
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}


STRF_HD char32_range range_of_hangul_l_w1() noexcept {
    static const char32_t samples[] = {
        0x00A97C
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_hangul_l_w2() noexcept {
    static const char32_t samples[] = {
        0x00115F
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_hangul_v_w1() noexcept {
    static const char32_t samples[] = {
        0x0011A7, 0x00D7C6,
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_hangul_t_w1() noexcept {
    static const char32_t samples[] = {
        0x0011FF, 0x00D7FB
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_hangul_lv_w2() noexcept {
    static const char32_t samples[] = {
        0xAC00, 0xD788
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_hangul_lvt_w2() noexcept {
    static const char32_t samples[] = {
        0xD789, 0xAC01, 0xAC08
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}

STRF_HD char32_range range_of_ri_w1() noexcept {
    static const char32_t samples[] = {
        0x1F1E6, 0x1F1FF
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}

STRF_HD char32_range range_of_xpic_w1() noexcept {
    static const char32_t samples[] = {
        0x0000A9, 0x0000AE, 0x00203C, 0x002049, 0x002122, 0x002139,
        0x002199, 0x0021AA, 0x00231B, 0x002328, 0x002388, 0x0023CF,
        0x0023F3, 0x0023FA, 0x0024C2, 0x0025AB, 0x0025B6, 0x0025C0,
        0x0025FE, 0x002605, 0x002612, 0x002685, 0x002705, 0x002712,
        0x002714, 0x002716, 0x00271D, 0x002721, 0x002728, 0x002734,
        0x002744, 0x002747, 0x00274C, 0x00274E, 0x002755, 0x002757,
        0x002767, 0x002797, 0x0027A1, 0x0027B0, 0x0027BF, 0x002935,
        0x002B07, 0x002B1C, 0x002B50, 0x002B55, 0x01F0FF, 0x01F10F,
        0x01F12F, 0x01F171, 0x01F17F, 0x01F18E, 0x01F19A, 0x01F1E5,
        0x01F20F, 0x01F21A, 0x01F22F, 0x01F23A, 0x01F23F, 0x01F2FF,
        0x01F6FF, 0x01F77F, 0x01F7FF, 0x01F80F, 0x01F84F, 0x01F85F,
        0x01F88F, 0x01F8FF, 0x01FAFF, 0x01FFFD
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_xpic_w2() noexcept {
    static const char32_t samples[] = {
        0x003030, 0x00303D, 0x003297, 0x003299, 0x01F3FA, 0x01F53D,
        0x01F64F, 0x01F93A, 0x01F945, 0x01F9FF
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_zwj() noexcept {
    static const char32_t samples[] = {
        0x00200D
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_spacing_mark_w1() noexcept {
    static const char32_t samples[] = {
        0x000903, 0x00093B, 0x000940, 0x00094C, 0x00094F, 0x000983,
        0x0009C0, 0x0009C8, 0x0009CC, 0x000A03, 0x000A40, 0x000A83,
        0x000AC0, 0x000AC9, 0x000ACC, 0x000B03, 0x000B40, 0x000B48,
        0x000B4C, 0x000BBF, 0x000BC2, 0x000BC8, 0x000BCC, 0x000C03,
        0x000C44, 0x000C83, 0x000CBE, 0x000CC1, 0x000CC4, 0x000CC8,
        0x000CCB, 0x000D03, 0x000D40, 0x000D48, 0x000D4C, 0x000D83,
        0x000DD1, 0x000DDE, 0x000DF3, 0x000E33, 0x000EB3, 0x000F3F,
        0x000F7F, 0x001031, 0x00103C, 0x001057, 0x001084, 0x0017B6,
        0x0017C5, 0x0017C8, 0x001926, 0x00192B, 0x001931, 0x001938,
        0x001A1A, 0x001A55, 0x001A57, 0x001A72, 0x001B04, 0x001B3B,
        0x001B41, 0x001B44, 0x001B82, 0x001BA1, 0x001BA7, 0x001BAA,
        0x001BE7, 0x001BEC, 0x001BEE, 0x001BF3, 0x001C2B, 0x001C35,
        0x001CE1, 0x001CF7, 0x00A824, 0x00A827, 0x00A881, 0x00A8C3,
        0x00A953, 0x00A983, 0x00A9B5, 0x00A9BB, 0x00A9C0, 0x00AA30,
        0x00AA34, 0x00AA4D, 0x00AAEB, 0x00AAEF, 0x00AAF5, 0x00ABE4,
        0x00ABE7, 0x00ABEA, 0x00ABEC, 0x011000, 0x011002, 0x011082,
        0x0110B2, 0x0110B8, 0x01112C, 0x011146, 0x011182, 0x0111B5,
        0x0111C0, 0x0111CE, 0x01122E, 0x011233, 0x011235, 0x0112E2,
        0x011303, 0x01133F, 0x011344, 0x011348, 0x01134D, 0x011363,
        0x011437, 0x011441, 0x011445, 0x0114B2, 0x0114B9, 0x0114BC,
        0x0114BE, 0x0114C1, 0x0115B1, 0x0115BB, 0x0115BE, 0x011632,
        0x01163C, 0x01163E, 0x0116AC, 0x0116AF, 0x0116B6, 0x011726,
        0x01182E, 0x011838, 0x011935, 0x011938, 0x01193D, 0x011940,
        0x011942, 0x0119D3, 0x0119DF, 0x0119E4, 0x011A39, 0x011A58,
        0x011A97, 0x011C2F, 0x011C3E, 0x011CA9, 0x011CB1, 0x011CB4,
        0x011D8E, 0x011D94, 0x011D96, 0x011EF6, 0x016F87, 0x016FF1,
        0x01D166, 0x01D16D
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_extend_and_control_w1() noexcept {
    static const char32_t samples[] = {
        0x0E007F  // 0x0E007F is extend but also control
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_extend_w1() noexcept {
    static const char32_t samples[] = {
        0x00036F, 0x000489, 0x0005BD, 0x0005BF, 0x0005C2, 0x0005C5,
        0x0005C7, 0x00061A, 0x00065F, 0x000670, 0x0006DC, 0x0006E4,
        0x0006E8, 0x0006ED, 0x000711, 0x00074A, 0x0007B0, 0x0007F3,
        0x0007FD, 0x000819, 0x000823, 0x000827, 0x00082D, 0x00085B,
        0x0008E1, 0x000902, 0x00093A, 0x00093C, 0x000948, 0x00094D,
        0x000957, 0x000963, 0x000981, 0x0009BC, 0x0009BE, 0x0009C4,
        0x0009CD, 0x0009D7, 0x0009E3, 0x0009FE, 0x000A02, 0x000A3C,
        0x000A42, 0x000A48, 0x000A4D, 0x000A51, 0x000A71, 0x000A75,
        0x000A82, 0x000ABC, 0x000AC5, 0x000AC8, 0x000ACD, 0x000AE3,
        0x000AFF, 0x000B01, 0x000B3C, 0x000B3F, 0x000B44, 0x000B4D,
        0x000B57, 0x000B63, 0x000B82, 0x000BBE, 0x000BC0, 0x000BCD,
        0x000BD7, 0x000C00, 0x000C04, 0x000C40, 0x000C48, 0x000C4D,
        0x000C56, 0x000C63, 0x000C81, 0x000CBC, 0x000CBF, 0x000CC2,
        0x000CC6, 0x000CCD, 0x000CD6, 0x000CE3, 0x000D01, 0x000D3C,
        0x000D3E, 0x000D44, 0x000D4D, 0x000D57, 0x000D63, 0x000D81,
        0x000DCA, 0x000DCF, 0x000DD4, 0x000DD6, 0x000DDF, 0x000E31,
        0x000E3A, 0x000E4E, 0x000EB1, 0x000EBC, 0x000ECD, 0x000F19,
        0x000F35, 0x000F37, 0x000F39, 0x000F7E, 0x000F84, 0x000F87,
        0x000F97, 0x000FBC, 0x000FC6, 0x001030, 0x001037, 0x00103A,
        0x00103E, 0x001059, 0x001060, 0x001074, 0x001082, 0x001086,
        0x00108D, 0x00109D, 0x00135F, 0x001714, 0x001734, 0x001753,
        0x001773, 0x0017B5, 0x0017BD, 0x0017C6, 0x0017D3, 0x0017DD,
        0x00180D, 0x001886, 0x0018A9, 0x001922, 0x001928, 0x001932,
        0x00193B, 0x001A18, 0x001A1B, 0x001A56, 0x001A5E, 0x001A60,
        0x001A62, 0x001A6C, 0x001A7C, 0x001A7F, 0x001AC0, 0x001B03,
        0x001B3A, 0x001B3C, 0x001B42, 0x001B73, 0x001B81, 0x001BA5,
        0x001BA9, 0x001BAD, 0x001BE6, 0x001BE9, 0x001BED, 0x001BF1,
        0x001C33, 0x001C37, 0x001CD2, 0x001CE0, 0x001CE8, 0x001CED,
        0x001CF4, 0x001CF9, 0x001DF9, 0x001DFF, 0x00200C, 0x0020F0,
        0x002CF1, 0x002D7F, 0x002DFF, 0x00A672, 0x00A67D, 0x00A69F,
        0x00A6F1, 0x00A802, 0x00A806, 0x00A80B, 0x00A826, 0x00A82C,
        0x00A8C5, 0x00A8F1, 0x00A8FF, 0x00A92D, 0x00A951, 0x00A982,
        0x00A9B3, 0x00A9B9, 0x00A9BD, 0x00A9E5, 0x00AA2E, 0x00AA32,
        0x00AA36, 0x00AA43, 0x00AA4C, 0x00AA7C, 0x00AAB0, 0x00AAB4,
        0x00AAB8, 0x00AABF, 0x00AAC1, 0x00AAED, 0x00AAF6, 0x00ABE5,
        0x00ABE8, 0x00ABED, 0x00FB1E, 0x00FE0F, 0x00FE2F, 0x00FF9F,
        0x0101FD, 0x0102E0, 0x01037A, 0x010A03, 0x010A06, 0x010A0F,
        0x010A3A, 0x010A3F, 0x010AE6, 0x010D27, 0x010EAC, 0x010F50,
        0x011001, 0x011046, 0x011081, 0x0110B6, 0x0110BA, 0x011102,
        0x01112B, 0x011134, 0x011173, 0x011181, 0x0111BE, 0x0111CC,
        0x0111CF, 0x011231, 0x011234, 0x011237, 0x01123E, 0x0112DF,
        0x0112EA, 0x011301, 0x01133C, 0x01133E, 0x011340, 0x011357,
        0x01136C, 0x011374, 0x01143F, 0x011444, 0x011446, 0x01145E,
        0x0114B0, 0x0114B8, 0x0114BA, 0x0114BD, 0x0114C0, 0x0114C3,
        0x0115AF, 0x0115B5, 0x0115BD, 0x0115C0, 0x0115DD, 0x01163A,
        0x01163D, 0x011640, 0x0116AB, 0x0116AD, 0x0116B5, 0x0116B7,
        0x01171F, 0x011725, 0x01172B, 0x011837, 0x01183A, 0x011930,
        0x01193C, 0x01193E, 0x011943, 0x0119D7, 0x0119DB, 0x0119E0,
        0x011A0A, 0x011A38, 0x011A3E, 0x011A47, 0x011A56, 0x011A5B,
        0x011A96, 0x011A99, 0x011C36, 0x011C3D, 0x011C3F, 0x011CA7,
        0x011CB0, 0x011CB3, 0x011CB6, 0x011D36, 0x011D3A, 0x011D3D,
        0x011D45, 0x011D47, 0x011D91, 0x011D95, 0x011D97, 0x011EF4,
        0x016AF4, 0x016B36, 0x016F4F, 0x016F92, 0x016FE4, 0x01BC9E,
        0x01D165, 0x01D169, 0x01D172, 0x01D182, 0x01D18B, 0x01D1AD,
        0x01D244, 0x01DA36, 0x01DA6C, 0x01DA75, 0x01DA84, 0x01DA9F,
        0x01DAAF, 0x01E006, 0x01E018, 0x01E021, 0x01E024, 0x01E02A,
        0x01E136, 0x01E2EF, 0x01E8D6, 0x01E94A, 0x0E01EF
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_extend_w2() noexcept {
    static const char32_t samples[] = {
        0x00302F, 0x00309A, 0x01F3FF
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}

STRF_HD char32_range range_of_other_w1() noexcept {
    static const char32_t samples[] = {
        0x00007E, 0x0000A8, 0x0000AC, 0x0002FF, 0x000482, 0x000590,
        0x0005BE, 0x0005C0, 0x0005C3, 0x0005C6, 0x0005FF, 0x00060F,
        0x00061B, 0x00064A, 0x00066F, 0x0006D5, 0x0006DE, 0x0006E6,
        0x0006E9, 0x00070E, 0x000710, 0x00072F, 0x0007A5, 0x0007EA,
        0x0007FC, 0x000815, 0x00081A, 0x000824, 0x000828, 0x000858,
        0x0008D2, 0x000939, 0x00093D, 0x000950, 0x000961, 0x000980,
        0x0009BB, 0x0009BD, 0x0009C6, 0x0009CA, 0x0009D6, 0x0009E1,
        0x0009FD, 0x000A00, 0x000A3B, 0x000A3D, 0x000A46, 0x000A4A,
        0x000A50, 0x000A6F, 0x000A74, 0x000A80, 0x000ABB, 0x000ABD,
        0x000AC6, 0x000ACA, 0x000AE1, 0x000AF9, 0x000B00, 0x000B3B,
        0x000B3D, 0x000B46, 0x000B4A, 0x000B54, 0x000B61, 0x000B81,
        0x000BBD, 0x000BC5, 0x000BC9, 0x000BD6, 0x000BFF, 0x000C3D,
        0x000C45, 0x000C49, 0x000C54, 0x000C61, 0x000C80, 0x000CBB,
        0x000CBD, 0x000CC5, 0x000CC9, 0x000CD4, 0x000CE1, 0x000CFF,
        0x000D3A, 0x000D3D, 0x000D45, 0x000D49, 0x000D56, 0x000D61,
        0x000D80, 0x000DC9, 0x000DCE, 0x000DD5, 0x000DD7, 0x000DF1,
        0x000E30, 0x000E32, 0x000E46, 0x000EB0, 0x000EB2, 0x000EC7,
        0x000F17, 0x000F34, 0x000F36, 0x000F38, 0x000F3D, 0x000F70,
        0x000F85, 0x000F8C, 0x000F98, 0x000FC5, 0x00102C, 0x001038,
        0x001055, 0x00105D, 0x001070, 0x001081, 0x001083, 0x00108C,
        0x00109C, 0x0010FF, 0x00135C, 0x001711, 0x001731, 0x001751,
        0x001771, 0x0017B3, 0x0017DC, 0x00180A, 0x001884, 0x0018A8,
        0x00191F, 0x00192F, 0x001A16, 0x001A54, 0x001A5F, 0x001A61,
        0x001A64, 0x001A7E, 0x001AAF, 0x001AFF, 0x001B33, 0x001B6A,
        0x001B7F, 0x001BA0, 0x001BE5, 0x001C23, 0x001CCF, 0x001CD3,
        0x001CEC, 0x001CF3, 0x001CF6, 0x001DBF, 0x001DFA, 0x00200A,
        0x002027, 0x00203B, 0x002048, 0x00205F, 0x0020CF, 0x002121,
        0x002138, 0x002193, 0x0021A8, 0x002319, 0x002327, 0x002387,
        0x0023CE, 0x0023E8, 0x0023F7, 0x0024C1, 0x0025A9, 0x0025B5,
        0x0025BF, 0x0025FA, 0x0025FF, 0x002606, 0x002613, 0x00268F,
        0x002707, 0x002713, 0x002715, 0x00271C, 0x002720, 0x002727,
        0x002732, 0x002743, 0x002746, 0x00274B, 0x00274D, 0x002752,
        0x002756, 0x002762, 0x002794, 0x0027A0, 0x0027AF, 0x0027BE,
        0x002933, 0x002B04, 0x002B1A, 0x002B4F, 0x002B54, 0x002CEE,
        0x002D7E, 0x002DDF, 0x002E7F, 0x00303F, 0x00A66E, 0x00A673,
        0x00A69D, 0x00A6EF, 0x00A801, 0x00A805, 0x00A80A, 0x00A822,
        0x00A82B, 0x00A87F, 0x00A8B3, 0x00A8DF, 0x00A8FE, 0x00A925,
        0x00A946, 0x00A95F, 0x00A97F, 0x00A9B2, 0x00A9E4, 0x00AA28,
        0x00AA42, 0x00AA4B, 0x00AA7B, 0x00AAAF, 0x00AAB1, 0x00AAB6,
        0x00AABD, 0x00AAC0, 0x00AAEA, 0x00AAF4, 0x00ABE2, 0x00ABEB,
        0x00ABFF, 0x00D7AF, 0x00D7CA, 0x00F8FF, 0x00FB1D, 0x00FDFF,
        0x00FE1F, 0x00FEFE, 0x00FF9D, 0x00FFDF, 0x00FFEF, 0x0101FC,
        0x0102DF, 0x010375, 0x010A00, 0x010A04, 0x010A0B, 0x010A37,
        0x010A3E, 0x010AE4, 0x010D23, 0x010EAA, 0x010F45, 0x010FFF,
        0x011037, 0x01107E, 0x0110AF, 0x0110BC, 0x0110CC, 0x0110FF,
        0x011126, 0x011144, 0x011172, 0x01117F, 0x0111B2, 0x0111C1,
        0x0111C8, 0x0111CD, 0x01122B, 0x01123D, 0x0112DE, 0x0112FF,
        0x01133A, 0x01133D, 0x011346, 0x01134A, 0x011356, 0x011361,
        0x011365, 0x01136F, 0x011434, 0x01145D, 0x0114AF, 0x0115AE,
        0x0115B7, 0x0115DB, 0x01162F, 0x0116AA, 0x01171C, 0x011721,
        0x01182B, 0x01192F, 0x011936, 0x01193A, 0x0119D0, 0x0119D9,
        0x0119E3, 0x011A00, 0x011A32, 0x011A46, 0x011A50, 0x011A83,
        0x011C2E, 0x011C37, 0x011C91, 0x011CA8, 0x011D30, 0x011D39,
        0x011D3B, 0x011D3E, 0x011D89, 0x011D8F, 0x011D92, 0x011EF2,
        0x01342F, 0x016AEF, 0x016B2F, 0x016F4E, 0x016F50, 0x016F8E,
        0x016FE3, 0x016FEF, 0x01BC9C, 0x01BC9F, 0x01D164, 0x01D16C,
        0x01D184, 0x01D1A9, 0x01D241, 0x01D9FF, 0x01DA3A, 0x01DA74,
        0x01DA83, 0x01DA9A, 0x01DAA0, 0x01DFFF, 0x01E007, 0x01E01A,
        0x01E022, 0x01E025, 0x01E12F, 0x01E2EB, 0x01E8CF, 0x01E943,
        0x01EFFF, 0x01F10C, 0x01F12E, 0x01F16B, 0x01F17D, 0x01F18D,
        0x01F190, 0x01F1AC, 0x01F200, 0x01F219, 0x01F22E, 0x01F231,
        0x01F23B, 0x01F248, 0x01F67F, 0x01F773, 0x01F7D4, 0x01F80B,
        0x01F847, 0x01F859, 0x01F887, 0x01F8AD, 0x01FBFF, 0x01FFFF,
        0x02FFFF, 0x0DFFFF, 0x10FFFF
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}
STRF_HD char32_range range_of_other_w2() noexcept {
    static const char32_t samples[] = {
        0x00232A, 0x003029, 0x00303C, 0x00303E, 0x003098, 0x003296,
        0x003298, 0x00A4CF, 0x00FAFF, 0x00FE19, 0x00FE6F, 0x00FF60,
        0x00FFE6, 0x01F545, 0x01F90B, 0x01F93B, 0x01F946, 0x02FFFD,
        0x03FFFD
    };
    return char32_range{samples, sizeof(samples)/sizeof(samples[0])};
}

template <typename... Args>
STRF_HD size_array<sizeof...(Args)> sizes(Args... args) {
    return size_array<sizeof...(Args)>{{static_cast<std::size_t>(args)...}};
}

STRF_HD void test_many_sequences()
{
    codepoint_samples control{range_of_control_w1(), "control_w1"};
    codepoint_samples cr{range_of_cr_w1(), "cr_w1"};
    codepoint_samples lf{range_of_lf_w1(), "lf_w1"};
    codepoint_samples prepend{range_of_prepend_w1(), "prepend_w1"};
    codepoint_samples hangul_l{range_of_hangul_l_w1(), "hangul_l_w1"};
    codepoint_samples hangul_l_w2{range_of_hangul_l_w2(), "hangul_l_w2"};
    codepoint_samples hangul_v{range_of_hangul_v_w1(), "hangul_v_w1"};
    codepoint_samples hangul_t{range_of_hangul_t_w1(), "hangul_t_w1"};
    codepoint_samples hangul_lv_w2{range_of_hangul_lv_w2(), "hangul_lv_w2"};
    codepoint_samples hangul_lvt_w2{range_of_hangul_lvt_w2(), "hangul_lvt_w2"};
    codepoint_samples ri{range_of_ri_w1(), "ri_w1"};
    codepoint_samples ext_pict{range_of_xpic_w1(), "xpic_w1"};
    codepoint_samples ext_pict_w2{range_of_xpic_w2(), "xpic_w2"};
    codepoint_samples zwj{range_of_zwj(), "zwj"};
    codepoint_samples spacing_mark{range_of_spacing_mark_w1(), "spacing_mark_w1"};
    codepoint_samples extend_and_control{range_of_extend_and_control_w1(), "extend_and_control_w1"};
    codepoint_samples extend{range_of_extend_w1(), "extend_w1"};
    codepoint_samples extend_w2{range_of_extend_w2(), "extend_w2"};
    codepoint_samples other{range_of_other_w1(), "other_w1"};
    codepoint_samples other_w2{range_of_other_w2(), "other_w2"};

    // The following test cases are based on
    // http://www.unicode.org/Public/UCD/latest/ucd/auxiliary/GraphemeBreakTest.txt

    // usage:
    // TEST_WIDTH( expected width ,  expected grapheme sizes ,  codepoints sequence )

    TEST_WIDTH(2, sizes(1, 1),       other, other);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       other, cr);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       other, lf);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       other, control);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, control);
    TEST_WIDTH(1, sizes(2),          other, extend);
    TEST_WIDTH(1, sizes(3),          other, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       other, ri);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       other, prepend);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, prepend);
    TEST_WIDTH(1, sizes(2),          other, spacing_mark);
    TEST_WIDTH(1, sizes(3),          other, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       other, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       other, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       other, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       other, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       other, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       other, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       other, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       other, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          other, extend);
    TEST_WIDTH(1, sizes(3),          other, extend, extend);
    TEST_WIDTH(1, sizes(2),          other, zwj);
    TEST_WIDTH(1, sizes(3),          other, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       other, other);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       cr, other);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       cr, cr);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, cr);
    TEST_WIDTH(1, sizes(2),          cr, lf);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       cr, control);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, control);
    TEST_WIDTH(2, sizes(1, 1),       cr, extend);
    TEST_WIDTH(2, sizes(1, 2),       cr, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       cr, ri);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       cr, prepend);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, prepend);
    TEST_WIDTH(2, sizes(1, 1),       cr, spacing_mark);
    TEST_WIDTH(2, sizes(1, 2),       cr, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       cr, hangul_l);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       cr, hangul_v);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       cr, hangul_t);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       cr, hangul_lv_w2);
    TEST_WIDTH(4, sizes(1, 1, 1),    cr, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       cr, hangul_lvt_w2);
    TEST_WIDTH(4, sizes(1, 1, 1),    cr, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       cr, ext_pict);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, ext_pict);
    TEST_WIDTH(2, sizes(1, 1),       cr, extend);
    TEST_WIDTH(2, sizes(1, 2),       cr, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       cr, zwj);
    TEST_WIDTH(2, sizes(1, 2),       cr, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       cr, other);
    TEST_WIDTH(3, sizes(1, 1, 1),    cr, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       lf, other);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       lf, cr);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       lf, lf);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       lf, control);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, control);
    TEST_WIDTH(2, sizes(1, 1),       lf, extend);
    TEST_WIDTH(2, sizes(1, 2),       lf, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       lf, ri);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       lf, prepend);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, prepend);
    TEST_WIDTH(2, sizes(1, 1),       lf, spacing_mark);
    TEST_WIDTH(2, sizes(1, 2),       lf, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       lf, hangul_l);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       lf, hangul_v);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       lf, hangul_t);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       lf, hangul_lv_w2);
    TEST_WIDTH(4, sizes(1, 1, 1),    lf, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       lf, hangul_lvt_w2);
    TEST_WIDTH(4, sizes(1, 1, 1),    lf, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       lf, ext_pict);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, ext_pict);
    TEST_WIDTH(2, sizes(1, 1),       lf, extend);
    TEST_WIDTH(2, sizes(1, 2),       lf, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       lf, zwj);
    TEST_WIDTH(2, sizes(1, 2),       lf, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       lf, other);
    TEST_WIDTH(3, sizes(1, 1, 1),    lf, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       control, other);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       control, cr);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       control, lf);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       control, control);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, control);
    TEST_WIDTH(2, sizes(1, 1),       control, extend);
    TEST_WIDTH(2, sizes(1, 2),       control, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       control, ri);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       control, prepend);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, prepend);
    TEST_WIDTH(2, sizes(1, 1),       control, spacing_mark);
    TEST_WIDTH(2, sizes(1, 2),       control, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       control, hangul_l);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       control, hangul_v);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       control, hangul_t);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       control, hangul_lv_w2);
    TEST_WIDTH(4, sizes(1, 1, 1),    control, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       control, hangul_lvt_w2);
    TEST_WIDTH(4, sizes(1, 1, 1),    control, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       control, ext_pict);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, ext_pict);
    TEST_WIDTH(2, sizes(1, 1),       control, extend);
    TEST_WIDTH(2, sizes(1, 2),       control, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       control, zwj);
    TEST_WIDTH(2, sizes(1, 2),       control, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       control, other);
    TEST_WIDTH(3, sizes(1, 1, 1),    control, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       extend, other);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       extend, cr);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       extend, lf);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       extend, control);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, control);
    TEST_WIDTH(1, sizes(2),          extend, extend);
    TEST_WIDTH(1, sizes(3),          extend, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       extend, ri);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       extend, prepend);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, prepend);
    TEST_WIDTH(1, sizes(2),          extend, spacing_mark);
    TEST_WIDTH(1, sizes(3),          extend, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       extend, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       extend, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       extend, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       extend, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       extend, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       extend, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       extend, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          extend, extend);
    TEST_WIDTH(1, sizes(3),          extend, extend, extend);
    TEST_WIDTH(1, sizes(2),          extend, zwj);
    TEST_WIDTH(1, sizes(3),          extend, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       extend, other);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       ri, other);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       ri, cr);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       ri, lf);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       ri, control);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, control);
    TEST_WIDTH(1, sizes(2),          ri, extend);
    TEST_WIDTH(1, sizes(3),          ri, extend, extend);
    TEST_WIDTH(1, sizes(2),          ri, ri);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       ri, prepend);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, prepend);
    TEST_WIDTH(1, sizes(2),          ri, spacing_mark);
    TEST_WIDTH(1, sizes(3),          ri, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       ri, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       ri, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       ri, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       ri, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       ri, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       ri, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       ri, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       ri, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          ri, extend);
    TEST_WIDTH(1, sizes(3),          ri, extend, extend);
    TEST_WIDTH(1, sizes(2),          ri, zwj);
    TEST_WIDTH(1, sizes(3),          ri, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       ri, other);
    TEST_WIDTH(2, sizes(2, 1),       ri, extend, other);
    TEST_WIDTH(1, sizes(2),          prepend, other);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       prepend, cr);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       prepend, lf);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       prepend, control);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, control);
    TEST_WIDTH(1, sizes(2),          prepend, extend);
    TEST_WIDTH(1, sizes(3),          prepend, extend, extend);
    TEST_WIDTH(1, sizes(2),          prepend, ri);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, ri);
    TEST_WIDTH(1, sizes(2),          prepend, prepend);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, prepend);
    TEST_WIDTH(1, sizes(2),          prepend, spacing_mark);
    TEST_WIDTH(1, sizes(3),          prepend, extend, spacing_mark);
    TEST_WIDTH(1, sizes(2),          prepend, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, hangul_l);
    TEST_WIDTH(1, sizes(2),          prepend, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, hangul_v);
    TEST_WIDTH(1, sizes(2),          prepend, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, hangul_t);
    TEST_WIDTH(1, sizes(2),          prepend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       prepend, extend, hangul_lv_w2);
    TEST_WIDTH(1, sizes(2),          prepend, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       prepend, extend, hangul_lvt_w2);
    TEST_WIDTH(1, sizes(2),          prepend, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          prepend, extend);
    TEST_WIDTH(1, sizes(3),          prepend, extend, extend);
    TEST_WIDTH(1, sizes(2),          prepend, zwj);
    TEST_WIDTH(1, sizes(3),          prepend, extend, zwj);
    TEST_WIDTH(1, sizes(2),          prepend, other);
    TEST_WIDTH(2, sizes(2, 1),       prepend, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, other);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, cr);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, lf);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, control);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, control);
    TEST_WIDTH(1, sizes(2),          spacing_mark, extend);
    TEST_WIDTH(1, sizes(3),          spacing_mark, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, ri);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, prepend);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, prepend);
    TEST_WIDTH(1, sizes(2),          spacing_mark, spacing_mark);
    TEST_WIDTH(1, sizes(3),          spacing_mark, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       spacing_mark, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       spacing_mark, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       spacing_mark, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       spacing_mark, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          spacing_mark, extend);
    TEST_WIDTH(1, sizes(3),          spacing_mark, extend, extend);
    TEST_WIDTH(1, sizes(2),          spacing_mark, zwj);
    TEST_WIDTH(1, sizes(3),          spacing_mark, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       spacing_mark, other);
    TEST_WIDTH(2, sizes(2, 1),       spacing_mark, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, other);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, cr);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, lf);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, control);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, control);
    TEST_WIDTH(1, sizes(2),          hangul_l, extend);
    TEST_WIDTH(1, sizes(3),          hangul_l, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, ri);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, prepend);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, prepend);
    TEST_WIDTH(1, sizes(2),          hangul_l, spacing_mark);
    TEST_WIDTH(1, sizes(3),          hangul_l, extend, spacing_mark);
    TEST_WIDTH(1, sizes(2),          hangul_l, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, hangul_l);
    TEST_WIDTH(1, sizes(2),          hangul_l, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, hangul_t);
    TEST_WIDTH(1, sizes(2),          hangul_l, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       hangul_l, extend, hangul_lv_w2);
    TEST_WIDTH(1, sizes(2),          hangul_l, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       hangul_l, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          hangul_l, extend);
    TEST_WIDTH(1, sizes(3),          hangul_l, extend, extend);
    TEST_WIDTH(1, sizes(2),          hangul_l, zwj);
    TEST_WIDTH(1, sizes(3),          hangul_l, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       hangul_l, other);
    TEST_WIDTH(2, sizes(2, 1),       hangul_l, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, other);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, cr);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, lf);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, control);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, control);
    TEST_WIDTH(1, sizes(2),          hangul_v, extend);
    TEST_WIDTH(1, sizes(3),          hangul_v, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, ri);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, prepend);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, prepend);
    TEST_WIDTH(1, sizes(2),          hangul_v, spacing_mark);
    TEST_WIDTH(1, sizes(3),          hangul_v, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, hangul_l);
    TEST_WIDTH(1, sizes(2),          hangul_v, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, hangul_v);
    TEST_WIDTH(1, sizes(2),          hangul_v, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       hangul_v, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       hangul_v, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       hangul_v, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       hangul_v, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          hangul_v, extend);
    TEST_WIDTH(1, sizes(3),          hangul_v, extend, extend);
    TEST_WIDTH(1, sizes(2),          hangul_v, zwj);
    TEST_WIDTH(1, sizes(3),          hangul_v, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       hangul_v, other);
    TEST_WIDTH(2, sizes(2, 1),       hangul_v, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, other);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, cr);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, lf);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, control);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, control);
    TEST_WIDTH(1, sizes(2),          hangul_t, extend);
    TEST_WIDTH(1, sizes(3),          hangul_t, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, ri);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, prepend);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, prepend);
    TEST_WIDTH(1, sizes(2),          hangul_t, spacing_mark);
    TEST_WIDTH(1, sizes(3),          hangul_t, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, hangul_v);
    TEST_WIDTH(1, sizes(2),          hangul_t, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       hangul_t, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       hangul_t, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       hangul_t, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       hangul_t, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          hangul_t, extend);
    TEST_WIDTH(1, sizes(3),          hangul_t, extend, extend);
    TEST_WIDTH(1, sizes(2),          hangul_t, zwj);
    TEST_WIDTH(1, sizes(3),          hangul_t, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       hangul_t, other);
    TEST_WIDTH(2, sizes(2, 1),       hangul_t, extend, other);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, other);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, other);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, cr);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, cr);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, lf);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, lf);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, control);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, control);
    TEST_WIDTH(2, sizes(2),          hangul_lv_w2, extend);
    TEST_WIDTH(2, sizes(3),          hangul_lv_w2, extend, extend);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, ri);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, ri);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, prepend);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, prepend);
    TEST_WIDTH(2, sizes(2),          hangul_lv_w2, spacing_mark);
    TEST_WIDTH(2, sizes(3),          hangul_lv_w2, extend, spacing_mark);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, hangul_l);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, hangul_l);
    TEST_WIDTH(2, sizes(2),          hangul_lv_w2, hangul_v);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, hangul_v);
    TEST_WIDTH(2, sizes(2),          hangul_lv_w2, hangul_t);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, hangul_t);
    TEST_WIDTH(4, sizes(1, 1),       hangul_lv_w2, hangul_lv_w2);
    TEST_WIDTH(4, sizes(2, 1),       hangul_lv_w2, extend, hangul_lv_w2);
    TEST_WIDTH(4, sizes(1, 1),       hangul_lv_w2, hangul_lvt_w2);
    TEST_WIDTH(4, sizes(2, 1),       hangul_lv_w2, extend, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, ext_pict);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, ext_pict);
    TEST_WIDTH(2, sizes(2),          hangul_lv_w2, extend);
    TEST_WIDTH(2, sizes(3),          hangul_lv_w2, extend, extend);
    TEST_WIDTH(2, sizes(2),          hangul_lv_w2, zwj);
    TEST_WIDTH(2, sizes(3),          hangul_lv_w2, extend, zwj);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lv_w2, other);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, extend, other);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, other);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, other);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, cr);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, cr);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, lf);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, lf);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, control);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, control);
    TEST_WIDTH(2, sizes(2),          hangul_lvt_w2, extend);
    TEST_WIDTH(2, sizes(3),          hangul_lvt_w2, extend, extend);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, ri);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, ri);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, prepend);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, prepend);
    TEST_WIDTH(2, sizes(2),          hangul_lvt_w2, spacing_mark);
    TEST_WIDTH(2, sizes(3),          hangul_lvt_w2, extend, spacing_mark);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, hangul_l);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, hangul_l);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, hangul_v);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, hangul_v);
    TEST_WIDTH(2, sizes(2),          hangul_lvt_w2, hangul_t);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, hangul_t);
    TEST_WIDTH(4, sizes(1, 1),       hangul_lvt_w2, hangul_lv_w2);
    TEST_WIDTH(4, sizes(2, 1),       hangul_lvt_w2, extend, hangul_lv_w2);
    TEST_WIDTH(4, sizes(1, 1),       hangul_lvt_w2, hangul_lvt_w2);
    TEST_WIDTH(4, sizes(2, 1),       hangul_lvt_w2, extend, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, ext_pict);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, ext_pict);
    TEST_WIDTH(2, sizes(2),          hangul_lvt_w2, extend);
    TEST_WIDTH(2, sizes(3),          hangul_lvt_w2, extend, extend);
    TEST_WIDTH(2, sizes(2),          hangul_lvt_w2, zwj);
    TEST_WIDTH(2, sizes(3),          hangul_lvt_w2, extend, zwj);
    TEST_WIDTH(3, sizes(1, 1),       hangul_lvt_w2, other);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, other);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, cr);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, lf);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, control);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, control);
    TEST_WIDTH(1, sizes(2),          ext_pict, extend);
    TEST_WIDTH(1, sizes(3),          ext_pict, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, ri);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, prepend);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, prepend);
    TEST_WIDTH(1, sizes(2),          ext_pict, spacing_mark);
    TEST_WIDTH(1, sizes(3),          ext_pict, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       ext_pict, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       ext_pict, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       ext_pict, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       ext_pict, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          ext_pict, extend);
    TEST_WIDTH(1, sizes(3),          ext_pict, extend, extend);
    TEST_WIDTH(1, sizes(2),          ext_pict, zwj);
    TEST_WIDTH(1, sizes(3),          ext_pict, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       ext_pict, other);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       extend, other);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       extend, cr);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       extend, lf);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       extend, control);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, control);
    TEST_WIDTH(1, sizes(2),          extend, extend);
    TEST_WIDTH(1, sizes(3),          extend, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       extend, ri);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       extend, prepend);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, prepend);
    TEST_WIDTH(1, sizes(2),          extend, spacing_mark);
    TEST_WIDTH(1, sizes(3),          extend, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       extend, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       extend, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       extend, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       extend, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       extend, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       extend, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       extend, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          extend, extend);
    TEST_WIDTH(1, sizes(3),          extend, extend, extend);
    TEST_WIDTH(1, sizes(2),          extend, zwj);
    TEST_WIDTH(1, sizes(3),          extend, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       extend, other);
    TEST_WIDTH(2, sizes(2, 1),       extend, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       zwj, other);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       zwj, cr);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       zwj, lf);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       zwj, control);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, control);
    TEST_WIDTH(1, sizes(2),          zwj, extend);
    TEST_WIDTH(1, sizes(3),          zwj, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       zwj, ri);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       zwj, prepend);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, prepend);
    TEST_WIDTH(1, sizes(2),          zwj, spacing_mark);
    TEST_WIDTH(1, sizes(3),          zwj, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       zwj, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       zwj, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       zwj, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       zwj, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       zwj, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       zwj, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       zwj, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       zwj, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          zwj, extend);
    TEST_WIDTH(1, sizes(3),          zwj, extend, extend);
    TEST_WIDTH(1, sizes(2),          zwj, zwj);
    TEST_WIDTH(1, sizes(3),          zwj, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       zwj, other);
    TEST_WIDTH(2, sizes(2, 1),       zwj, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       other, other);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, other);
    TEST_WIDTH(2, sizes(1, 1),       other, cr);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, cr);
    TEST_WIDTH(2, sizes(1, 1),       other, lf);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, lf);
    TEST_WIDTH(2, sizes(1, 1),       other, control);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, control);
    TEST_WIDTH(1, sizes(2),          other, extend);
    TEST_WIDTH(1, sizes(3),          other, extend, extend);
    TEST_WIDTH(2, sizes(1, 1),       other, ri);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, ri);
    TEST_WIDTH(2, sizes(1, 1),       other, prepend);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, prepend);
    TEST_WIDTH(1, sizes(2),          other, spacing_mark);
    TEST_WIDTH(1, sizes(3),          other, extend, spacing_mark);
    TEST_WIDTH(2, sizes(1, 1),       other, hangul_l);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, hangul_l);
    TEST_WIDTH(2, sizes(1, 1),       other, hangul_v);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, hangul_v);
    TEST_WIDTH(2, sizes(1, 1),       other, hangul_t);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, hangul_t);
    TEST_WIDTH(3, sizes(1, 1),       other, hangul_lv_w2);
    TEST_WIDTH(3, sizes(2, 1),       other, extend, hangul_lv_w2);
    TEST_WIDTH(3, sizes(1, 1),       other, hangul_lvt_w2);
    TEST_WIDTH(3, sizes(2, 1),       other, extend, hangul_lvt_w2);
    TEST_WIDTH(2, sizes(1, 1),       other, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, ext_pict);
    TEST_WIDTH(1, sizes(2),          other, extend);
    TEST_WIDTH(1, sizes(3),          other, extend, extend);
    TEST_WIDTH(1, sizes(2),          other, zwj);
    TEST_WIDTH(1, sizes(3),          other, extend, zwj);
    TEST_WIDTH(2, sizes(1, 1),       other, other);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, other);
    TEST_WIDTH(4, sizes(2, 1, 1, 1), cr, lf, other, lf, extend);
    TEST_WIDTH(1, sizes(2),          other, extend);
    TEST_WIDTH(2, sizes(2, 1),       other, zwj, other);
    TEST_WIDTH(2, sizes(2, 1),       other, zwj, other);
    TEST_WIDTH(1, sizes(2),          hangul_l, hangul_l);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lv_w2, hangul_t, hangul_l);
    TEST_WIDTH(3, sizes(2, 1),       hangul_lvt_w2, hangul_t, hangul_l);
    TEST_WIDTH(3, sizes(2, 1, 1),    ri, ri, ri, other);
    TEST_WIDTH(4, sizes(1, 2, 1, 1), other, ri, ri, ri, other);
    TEST_WIDTH(4, sizes(1, 3, 1, 1), other, ri, ri, zwj, ri, other);
    TEST_WIDTH(4, sizes(1, 2, 2, 1), other, ri, zwj, ri, ri, other);
    TEST_WIDTH(4, sizes(1, 2, 2, 1), other, ri, ri, ri, ri, other);
    TEST_WIDTH(1, sizes(2),          other, zwj);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, other);
    TEST_WIDTH(2, sizes(2, 1),       other, spacing_mark, other);
    TEST_WIDTH(2, sizes(1, 2),       other, prepend, other);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, extend, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       other, extend, ext_pict);
    TEST_WIDTH(2, sizes(2, 3),       other, extend, ext_pict, zwj, ext_pict);
    TEST_WIDTH(1, sizes(6),          ext_pict, extend, extend, zwj, ext_pict, extend);
    TEST_WIDTH(1, sizes(3),          ext_pict, zwj, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       other, zwj, ext_pict);
    TEST_WIDTH(1, sizes(3),          ext_pict, zwj, ext_pict);
    TEST_WIDTH(2, sizes(2, 1),       ext_pict, zwj, other);

    // again, but using instead fullwidth characters:

    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, cr);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, lf);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, lf);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, control);
    TEST_WIDTH(2           , sizes(2, 1),       other, extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          other, extend_w2);
    TEST_WIDTH(1           , sizes(3),          other, extend_w2, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, ri);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, prepend);
    TEST_WIDTH(2           , sizes(2, 1),       other,    extend_w2, prepend);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, spacing_mark);
    TEST_WIDTH(1           , sizes(3),          other   , extend_w2, spacing_mark);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       other   , extend_w2, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, hangul_v);
    TEST_WIDTH(2           , sizes(2, 1),       other   , extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, hangul_t);
    TEST_WIDTH(2           , sizes(2, 1),       other   , extend_w2, hangul_t);
    TEST_WIDTH(3+1         , sizes(1, 1),       other_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       other_w2, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       other_w2, hangul_lvt_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       other_w2, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, extend_w2);
    TEST_WIDTH(1           , sizes(3),          other   , extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, zwj);
    TEST_WIDTH(1+1         , sizes(3),          other_w2, extend_w2, zwj);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       cr, other_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    cr, extend_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    cr, extend_w2, cr);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    cr, extend_w2, lf);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    cr, extend_w2, control);
    TEST_WIDTH(2+1         , sizes(1, 1),       cr, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       cr, extend_w2, extend_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    cr, extend_w2, ri);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    cr, extend_w2, prepend);
    TEST_WIDTH(2+1         , sizes(1, 2),       cr, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(1, 1),       cr, hangul_l_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    cr, extend_w2, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    cr, extend_w2, hangul_v);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    cr, extend_w2, hangul_t);
    TEST_WIDTH(3           , sizes(1, 1),       cr, hangul_lv_w2);
    TEST_WIDTH(4+1         , sizes(1, 1, 1),    cr, extend_w2, hangul_lv_w2);
    TEST_WIDTH(4+1         , sizes(1, 1, 1),    cr, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       cr, ext_pict_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    cr, extend_w2, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       cr, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       cr, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       cr, other_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    cr, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       lf, other_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    lf, extend_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    lf, extend_w2, cr);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    lf, extend_w2, lf);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    lf, extend_w2, control);
    TEST_WIDTH(2+1         , sizes(1, 1),       lf, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       lf, extend_w2, extend_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    lf, extend_w2, ri);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    lf, extend_w2, prepend);
    TEST_WIDTH(2+1         , sizes(1, 2),       lf, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(1, 1),       lf, hangul_l_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    lf, extend_w2, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    lf, extend_w2, hangul_v);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    lf, extend_w2, hangul_t);
    TEST_WIDTH(4+1         , sizes(1, 1, 1),    lf, extend_w2, hangul_lv_w2);
    TEST_WIDTH(4+1         , sizes(1, 1, 1),    lf, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       lf, ext_pict_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    lf, extend_w2, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       lf, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       lf, extend_w2, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       lf, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       lf, other_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    lf, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       control, other_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    control, extend_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    control, extend_w2, cr);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    control, extend_w2, lf);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    control, extend_w2, control);
    TEST_WIDTH(2+1         , sizes(1, 1),       control, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       control, extend_w2, extend_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    control, extend_w2, ri);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    control, extend_w2, prepend);
    TEST_WIDTH(2+1         , sizes(1, 2),       control, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(1, 1),       control, hangul_l_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    control, extend_w2, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    control, extend_w2, hangul_v);
    TEST_WIDTH(3+1         , sizes(1, 1, 1),    control, extend_w2, hangul_t);
    TEST_WIDTH(4+1         , sizes(1, 1, 1),    control, extend_w2, hangul_lv_w2);
    TEST_WIDTH(4+1         , sizes(1, 1, 1),    control, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       control, ext_pict_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    control, extend_w2, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       control, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       control, extend_w2, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       control, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       control, other_w2);
    TEST_WIDTH(3+1+1       , sizes(1, 1, 1),    control, extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       extend_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, lf);
    TEST_WIDTH(2           , sizes(2, 1),       extend   , extend_w2, lf);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, control);
    TEST_WIDTH(2           , sizes(2, 1),       extend   , extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          extend   , extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend   , extend   );
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, prepend);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, prepend);
    TEST_WIDTH(1+1         , sizes(2),          extend_w2, spacing_mark);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend_w2, spacing_mark);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, hangul_l_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       extend_w2, extend   , hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, hangul_t);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_t);
    TEST_WIDTH(3+1         , sizes(1, 1),       extend_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       extend_w2, hangul_lvt_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend   , extend_w2, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(2),          extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(2),          extend_w2, zwj);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend_w2, zwj);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       extend_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       ri, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       ri, extend_w2, other_w2);
    TEST_WIDTH(2           , sizes(2, 1),       ri, extend_w2, cr);
    TEST_WIDTH(2           , sizes(2, 1),       ri, extend_w2, lf);
    TEST_WIDTH(2           , sizes(2, 1),       ri, extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          ri, extend_w2);
    TEST_WIDTH(1           , sizes(3),          ri, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(2, 1),       ri, extend_w2, ri);
    TEST_WIDTH(2           , sizes(2, 1),       ri, extend_w2, prepend);
    TEST_WIDTH(1           , sizes(3),          ri, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(1, 1),       ri, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       ri, extend_w2, hangul_l_w2);
    TEST_WIDTH(2           , sizes(2, 1),       ri, extend_w2, hangul_v);
    TEST_WIDTH(2           , sizes(2, 1),       ri, extend_w2, hangul_t);
    TEST_WIDTH(3           , sizes(2, 1),       ri, extend_w2, hangul_lv_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       ri, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       ri, extend_w2, ext_pict_w2);
    TEST_WIDTH(1           , sizes(2),          ri, extend_w2);
    TEST_WIDTH(1           , sizes(3),          ri, extend_w2, extend_w2);
    TEST_WIDTH(1           , sizes(3),          ri, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       ri, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       ri, extend_w2, other_w2);
    TEST_WIDTH(1           , sizes(2),          prepend, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       prepend, extend_w2, other_w2);
    TEST_WIDTH(2           , sizes(2, 1),       prepend, extend_w2, cr);
    TEST_WIDTH(2           , sizes(2, 1),       prepend, extend_w2, lf);
    TEST_WIDTH(2           , sizes(2, 1),       prepend, extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          prepend, extend_w2);
    TEST_WIDTH(1           , sizes(3),          prepend, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(2, 1),       prepend, extend_w2, ri);
    TEST_WIDTH(2           , sizes(2, 1),       prepend, extend_w2, prepend);
    TEST_WIDTH(1           , sizes(3),          prepend, extend_w2, spacing_mark);
    TEST_WIDTH(1           , sizes(2),          prepend, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       prepend, extend_w2, hangul_l_w2);
    TEST_WIDTH(2           , sizes(2, 1),       prepend, extend_w2, hangul_v);
    TEST_WIDTH(2           , sizes(2, 1),       prepend, extend_w2, hangul_t);
    TEST_WIDTH(3           , sizes(2, 1),       prepend, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3           , sizes(2, 1),       prepend, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(1           , sizes(2),          prepend, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       prepend, extend_w2, ext_pict_w2);
    TEST_WIDTH(1           , sizes(2),          prepend, extend_w2);
    TEST_WIDTH(1           , sizes(3),          prepend, extend_w2, extend_w2);
    TEST_WIDTH(1           , sizes(3),          prepend, extend_w2, zwj);
    TEST_WIDTH(1           , sizes(2),          prepend, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       prepend, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       spacing_mark, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       spacing_mark, extend_w2, other_w2);
    TEST_WIDTH(2           , sizes(2, 1),       spacing_mark, extend_w2, cr);
    TEST_WIDTH(2           , sizes(2, 1),       spacing_mark, extend_w2, lf);
    TEST_WIDTH(2           , sizes(2, 1),       spacing_mark, extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          spacing_mark, extend_w2);
    TEST_WIDTH(1           , sizes(3),          spacing_mark, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(2, 1),       spacing_mark, extend_w2, ri);
    TEST_WIDTH(2           , sizes(2, 1),       spacing_mark, extend_w2, prepend);
    TEST_WIDTH(1           , sizes(3),          spacing_mark, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(1, 1),       spacing_mark, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       spacing_mark, extend_w2, hangul_l_w2);
    TEST_WIDTH(2           , sizes(2, 1),       spacing_mark, extend_w2, hangul_v);
    TEST_WIDTH(2           , sizes(2, 1),       spacing_mark, extend_w2, hangul_t);
    TEST_WIDTH(3           , sizes(2, 1),       spacing_mark, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3           , sizes(2, 1),       spacing_mark, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       spacing_mark, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       spacing_mark, extend_w2, ext_pict_w2);
    TEST_WIDTH(1           , sizes(2),          spacing_mark, extend_w2);
    TEST_WIDTH(1           , sizes(3),          spacing_mark, extend_w2, extend_w2);
    TEST_WIDTH(1           , sizes(3),          spacing_mark, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       spacing_mark, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       spacing_mark, extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       hangul_l_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       hangul_l_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_l_w2, cr);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_l_w2, extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_l_w2, lf);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_l_w2, extend_w2, lf);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_l_w2, control);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_l_w2, extend_w2, control);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          hangul_l_w2, extend_w2, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_l_w2, ri);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_l_w2, extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_l_w2, prepend);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_l_w2, extend_w2, prepend);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, spacing_mark);
    TEST_WIDTH(1+1         , sizes(3),          hangul_l_w2, extend_w2, spacing_mark);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, hangul_l_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       hangul_l_w2, extend_w2, hangul_l_w2);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_l_w2, extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_l_w2, hangul_t);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_l_w2, extend_w2, hangul_t);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_l_w2, extend_w2, hangul_lv_w2);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, hangul_lvt_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_l_w2, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       hangul_l_w2, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       hangul_l_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          hangul_l_w2, extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, zwj);
    TEST_WIDTH(1+1         , sizes(3),          hangul_l_w2, extend_w2, zwj);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       hangul_l_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       hangul_l_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_v, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_v, extend_w2, other_w2);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_v, extend_w2, cr);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_v, extend_w2, lf);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_v, extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          hangul_v, extend_w2);
    TEST_WIDTH(1           , sizes(3),          hangul_v, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_v, extend_w2, ri);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_v, extend_w2, prepend);
    TEST_WIDTH(1           , sizes(3),          hangul_v, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_v, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_v, extend_w2, hangul_l_w2);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_v, extend_w2, hangul_v);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_v, extend_w2, hangul_t);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_v, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3           , sizes(1, 1),       hangul_v, hangul_lvt_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_v, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_v, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_v, extend_w2, ext_pict_w2);
    TEST_WIDTH(1           , sizes(2),          hangul_v, extend_w2);
    TEST_WIDTH(1           , sizes(3),          hangul_v, extend_w2, extend_w2);
    TEST_WIDTH(1           , sizes(3),          hangul_v, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_v, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_v, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_t, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_t, extend_w2, other_w2);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_t, extend_w2, cr);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_t, extend_w2, lf);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_t, extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          hangul_t, extend_w2);
    TEST_WIDTH(1           , sizes(3),          hangul_t, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_t, extend_w2, ri);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_t, extend_w2, prepend);
    TEST_WIDTH(1           , sizes(3),          hangul_t, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_t, extend_w2, hangul_l_w2);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_t, extend_w2, hangul_v);
    TEST_WIDTH(2           , sizes(2, 1),       hangul_t, extend_w2, hangul_t);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_t, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_t, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_t, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_t, extend_w2, ext_pict_w2);
    TEST_WIDTH(1           , sizes(2),          hangul_t, extend_w2);
    TEST_WIDTH(1           , sizes(3),          hangul_t, extend_w2, extend_w2);
    TEST_WIDTH(1           , sizes(3),          hangul_t, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       hangul_t, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       hangul_t, extend_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lv_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lv_w2, extend_w2, other_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lv_w2, extend_w2, cr);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lv_w2, extend_w2, lf);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lv_w2, extend_w2, control);
    TEST_WIDTH(2           , sizes(2),          hangul_lv_w2, extend_w2);
    TEST_WIDTH(2           , sizes(3),          hangul_lv_w2, extend_w2, extend_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lv_w2, extend_w2, ri);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lv_w2, extend_w2, prepend);
    TEST_WIDTH(2           , sizes(3),          hangul_lv_w2, extend_w2, spacing_mark);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lv_w2, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lv_w2, extend_w2, hangul_l_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lv_w2, extend_w2, hangul_v);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lv_w2, extend_w2, hangul_t);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lv_w2, ext_pict_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lv_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(2           , sizes(2),          hangul_lv_w2, extend_w2);
    TEST_WIDTH(2           , sizes(3),          hangul_lv_w2, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(3),          hangul_lv_w2, extend_w2, zwj);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lv_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lv_w2, extend_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lvt_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lvt_w2, extend_w2, other_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lvt_w2, extend_w2, cr);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lvt_w2, extend_w2, lf);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lvt_w2, extend_w2, control);
    TEST_WIDTH(2           , sizes(2),          hangul_lvt_w2, extend_w2);
    TEST_WIDTH(2           , sizes(3),          hangul_lvt_w2, extend_w2, extend_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lvt_w2, extend_w2, ri);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lvt_w2, extend_w2, prepend);
    TEST_WIDTH(2           , sizes(3),          hangul_lvt_w2, extend_w2, spacing_mark);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lvt_w2, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lvt_w2, extend_w2, hangul_l_w2);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lvt_w2, extend_w2, hangul_v);
    TEST_WIDTH(3           , sizes(2, 1),       hangul_lvt_w2, extend_w2, hangul_t);
    TEST_WIDTH(4           , sizes(2, 1),       hangul_lvt_w2, extend_w2, hangul_lv_w2);
    TEST_WIDTH(4           , sizes(2, 1),       hangul_lvt_w2, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lvt_w2, ext_pict_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lvt_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(2           , sizes(2),          hangul_lvt_w2, extend_w2);
    TEST_WIDTH(2           , sizes(3),          hangul_lvt_w2, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(3),          hangul_lvt_w2, extend_w2, zwj);
    TEST_WIDTH(3+1         , sizes(1, 1),       hangul_lvt_w2, other_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lvt_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       ext_pict_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       ext_pict_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       ext_pict_w2, cr);
    TEST_WIDTH(2+1         , sizes(2, 1),       ext_pict_w2, extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(1, 1),       ext_pict_w2, lf);
    TEST_WIDTH(2+1         , sizes(2, 1),       ext_pict_w2, extend_w2, lf);
    TEST_WIDTH(2+1         , sizes(1, 1),       ext_pict_w2, control);
    TEST_WIDTH(2+1         , sizes(2, 1),       ext_pict_w2, extend_w2, control);
    TEST_WIDTH(1+1         , sizes(2),          ext_pict_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          ext_pict_w2, extend_w2, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       ext_pict_w2, ri);
    TEST_WIDTH(2+1         , sizes(2, 1),       ext_pict_w2, extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(1, 1),       ext_pict_w2, prepend);
    TEST_WIDTH(2+1         , sizes(2, 1),       ext_pict_w2, extend_w2, prepend);
    TEST_WIDTH(1+1         , sizes(2),          ext_pict_w2, spacing_mark);
    TEST_WIDTH(1+1         , sizes(3),          ext_pict_w2, extend_w2, spacing_mark);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       ext_pict_w2, hangul_l_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       ext_pict_w2, extend_w2, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       ext_pict_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(2, 1),       ext_pict_w2, extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(1, 1),       ext_pict_w2, hangul_t);
    TEST_WIDTH(2+1         , sizes(2, 1),       ext_pict_w2, extend_w2, hangul_t);
    TEST_WIDTH(3+1         , sizes(1, 1),       ext_pict_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       ext_pict_w2, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       ext_pict_w2, hangul_lvt_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       ext_pict_w2, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       ext_pict_w2, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       ext_pict_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(2),          ext_pict_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          ext_pict_w2, extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(2),          ext_pict_w2, zwj);
    TEST_WIDTH(1+1         , sizes(3),          ext_pict_w2, extend_w2, zwj);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       ext_pict_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       ext_pict_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       extend_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, lf);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, lf);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, control);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, control);
    TEST_WIDTH(1+1         , sizes(2),          extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend_w2, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, prepend);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, prepend);
    TEST_WIDTH(1+1         , sizes(2),          extend_w2, spacing_mark);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend_w2, spacing_mark);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, hangul_l_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       extend_w2, extend_w2, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(1, 1),       extend_w2, hangul_t);
    TEST_WIDTH(2+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_t);
    TEST_WIDTH(3+1         , sizes(1, 1),       extend_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       extend_w2, hangul_lvt_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       extend_w2, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       extend_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(2),          extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(2),          extend_w2, zwj);
    TEST_WIDTH(1+1         , sizes(3),          extend_w2, extend_w2, zwj);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       extend_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       zwj, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       zwj, extend_w2, other_w2);
    TEST_WIDTH(2           , sizes(2, 1),       zwj, extend_w2, cr);
    TEST_WIDTH(2           , sizes(2, 1),       zwj, extend_w2, lf);
    TEST_WIDTH(2           , sizes(2, 1),       zwj, extend_w2, control);
    TEST_WIDTH(1           , sizes(2),          zwj, extend_w2);
    TEST_WIDTH(1           , sizes(3),          zwj, extend_w2, extend_w2);
    TEST_WIDTH(2           , sizes(2, 1),       zwj, extend_w2, ri);
    TEST_WIDTH(2           , sizes(2, 1),       zwj, extend_w2, prepend);
    TEST_WIDTH(1           , sizes(3),          zwj, extend_w2, spacing_mark);
    TEST_WIDTH(2+1         , sizes(1, 1),       zwj, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       zwj, extend_w2, hangul_l_w2);
    TEST_WIDTH(2           , sizes(2, 1),       zwj, extend_w2, hangul_v);
    TEST_WIDTH(2           , sizes(2, 1),       zwj, extend_w2, hangul_t);
    TEST_WIDTH(3           , sizes(2, 1),       zwj, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3           , sizes(2, 1),       zwj, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       zwj, ext_pict_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       zwj, extend_w2, ext_pict_w2);
    TEST_WIDTH(1           , sizes(2),          zwj, extend_w2);
    TEST_WIDTH(1           , sizes(3),          zwj, extend_w2, extend_w2);
    TEST_WIDTH(1           , sizes(3),          zwj, extend_w2, zwj);
    TEST_WIDTH(2+1         , sizes(1, 1),       zwj, other_w2);
    TEST_WIDTH(2+1         , sizes(2, 1),       zwj, extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, cr);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, cr);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, lf);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, lf);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, control);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, control);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          other_w2, extend_w2, extend_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, ri);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, ri);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, prepend);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, prepend);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, spacing_mark);
    TEST_WIDTH(1+1         , sizes(3),          other_w2, extend_w2, spacing_mark);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, hangul_l_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, hangul_l_w2);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, hangul_v);
    TEST_WIDTH(2+1         , sizes(1, 1),       other_w2, hangul_t);
    TEST_WIDTH(2+1         , sizes(2, 1),       other_w2, extend_w2, hangul_t);
    TEST_WIDTH(3+1         , sizes(1, 1),       other_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       other_w2, extend_w2, hangul_lv_w2);
    TEST_WIDTH(3+1         , sizes(1, 1),       other_w2, hangul_lvt_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       other_w2, extend_w2, hangul_lvt_w2);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          other_w2, extend_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, zwj);
    TEST_WIDTH(1+1         , sizes(3),          other_w2, extend_w2, zwj);
    TEST_WIDTH(2+1+1       , sizes(1, 1),       other_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, other_w2);
    TEST_WIDTH(4+1+1       , sizes(2, 1, 1, 1), cr, lf, other_w2, lf, extend_w2);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, extend_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, zwj, other_w2);
    TEST_WIDTH(1+1         , sizes(2),          hangul_l_w2, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lv_w2, hangul_t, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(2, 1),       hangul_lvt_w2, hangul_t, hangul_l_w2);
    TEST_WIDTH(3+1         , sizes(2, 1, 1),    ri, ri, ri, other_w2);
    TEST_WIDTH(4+1+1       , sizes(1, 2, 1, 1), other_w2, ri, ri, ri, other_w2);
    TEST_WIDTH(4+1+1       , sizes(1, 3, 1, 1), other_w2, ri, ri, zwj, ri, other_w2);
    TEST_WIDTH(4+1+1       , sizes(1, 2, 2, 1), other_w2, ri, zwj, ri, ri, other_w2);
    TEST_WIDTH(4+1+1       , sizes(1, 2, 2, 1), other_w2, ri, ri, ri, ri, other_w2);
    TEST_WIDTH(1+1         , sizes(2),          other_w2, zwj);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, spacing_mark, other_w2);
    TEST_WIDTH(2+1         , sizes(1, 2),       other_w2, prepend, other_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       ext_pict_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, extend_w2, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 3),       other_w2, extend_w2, ext_pict_w2, zwj, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(6),          ext_pict_w2, extend_w2, extend_w2, zwj, ext_pict_w2, extend_w2);
    TEST_WIDTH(1+1         , sizes(3),          ext_pict_w2, zwj, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       other_w2, zwj, ext_pict_w2);
    TEST_WIDTH(1+1         , sizes(3),          ext_pict_w2, zwj, ext_pict_w2);
    TEST_WIDTH(2+1+1       , sizes(2, 1),       ext_pict_w2, zwj, other_w2);


    using namespace strf::width_literal;
    TEST_POS(2, 2_w, other, other, other);
    TEST_POS(1, 2_w, other_w2, other, other);
    TEST_POS(1, 3.999_w, other_w2, other_w2, other);
    TEST_POS(0, 1.999_w, other_w2, other_w2, other);
    TEST_POS(5, 3_w, other_w2, other, extend, extend, extend, other, other);
    TEST_POS(5, 3_w, other_w2, other, extend, extend, extend, other, other_w2);
    TEST_POS(5, 3_w, other_w2, other, extend, extend, extend, other, control);
    TEST_POS(5, 3_w, other_w2, other, extend, extend, extend, other, ext_pict);
    TEST_POS(5, 3_w, other_w2, other, extend, extend, extend, other, ri, ri);

    // todo
}

STRF_HD strf::width_t char32_width(char32_t ch) noexcept
{
    return strf::std_width_calc_t::char32_width(ch);
}

STRF_HD strf::width_t reference_char32_width(char32_t ch) noexcept
{
    using namespace strf::width_literal;
    return ( (0x1100 <= ch && ch <= 0x115F)
          || (0x2329 <= ch && ch <= 0x232A)
          || (0x2E80 <= ch && ch <= 0x303E)
          || (0x3040 <= ch && ch <= 0xA4CF)
          || (0xAC00 <= ch && ch <= 0xD7A3)
          || (0xF900 <= ch && ch <= 0xFAFF)
          || (0xFE10 <= ch && ch <= 0xFE19)
          || (0xFE30 <= ch && ch <= 0xFE6F)
          || (0xFF00 <= ch && ch <= 0xFF60)
          || (0xFFE0 <= ch && ch <= 0xFFE6)
          || (0x1F300 <= ch && ch <= 0x1F64F)
          || (0x1F900 <= ch && ch <= 0x1F9FF)
          || (0x20000 <= ch && ch <= 0x2FFFD)
          || (0x30000 <= ch && ch <= 0x3FFFD) ) ? 2_w : 1_w;
}

#define TEST_CHAR_WIDTH(CH) \
    TEST_TRUE(reference_char32_width(CH) == char32_width(CH));

STRF_HD void test_single_char32_width()
{
    TEST_CHAR_WIDTH(0);
    TEST_CHAR_WIDTH(0x10FF);
    TEST_CHAR_WIDTH(0x1100);
    TEST_CHAR_WIDTH(0x115F);
    TEST_CHAR_WIDTH(0x1160);
    TEST_CHAR_WIDTH(0x2328);
    TEST_CHAR_WIDTH(0x2329);
    TEST_CHAR_WIDTH(0x232A);
    TEST_CHAR_WIDTH(0x232B);
    TEST_CHAR_WIDTH(0x2E7F);
    TEST_CHAR_WIDTH(0x2E80);
    TEST_CHAR_WIDTH(0x303E);
    TEST_CHAR_WIDTH(0x303F);
    TEST_CHAR_WIDTH(0x3040);
    TEST_CHAR_WIDTH(0xA4CF);
    TEST_CHAR_WIDTH(0xA4D0);
    TEST_CHAR_WIDTH(0xABFF);
    TEST_CHAR_WIDTH(0xAC00);
    TEST_CHAR_WIDTH(0xD7A3);
    TEST_CHAR_WIDTH(0xD7A4);
    TEST_CHAR_WIDTH(0xF8FF);
    TEST_CHAR_WIDTH(0xF900);
    TEST_CHAR_WIDTH(0xFAFF);
    TEST_CHAR_WIDTH(0xFB00);
    TEST_CHAR_WIDTH(0xFE0F);
    TEST_CHAR_WIDTH(0xFE10);
    TEST_CHAR_WIDTH(0xFE19);
    TEST_CHAR_WIDTH(0xFE1A);
    TEST_CHAR_WIDTH(0xFE2F);
    TEST_CHAR_WIDTH(0xFE30);
    TEST_CHAR_WIDTH(0xFE6F);
    TEST_CHAR_WIDTH(0xFE70);
    TEST_CHAR_WIDTH(0xF3FF);
    TEST_CHAR_WIDTH(0xFF00);
    TEST_CHAR_WIDTH(0xFF60);
    TEST_CHAR_WIDTH(0xFF61);
    TEST_CHAR_WIDTH(0xFFDF);
    TEST_CHAR_WIDTH(0xFFE0);
    TEST_CHAR_WIDTH(0xFFE6);
    TEST_CHAR_WIDTH(0xFFE7);
    TEST_CHAR_WIDTH(0x1F2FF);
    TEST_CHAR_WIDTH(0x1F300);
    TEST_CHAR_WIDTH(0x1F64F);
    TEST_CHAR_WIDTH(0x1F650);
    TEST_CHAR_WIDTH(0x1F8FF);
    TEST_CHAR_WIDTH(0x1F900);
    TEST_CHAR_WIDTH(0x1F9FF);
    TEST_CHAR_WIDTH(0x1FA00);
    TEST_CHAR_WIDTH(0x1FFFF);
    TEST_CHAR_WIDTH(0x20000);
    TEST_CHAR_WIDTH(0x2FFFD);
    TEST_CHAR_WIDTH(0x2FFFE);
    TEST_CHAR_WIDTH(0x2FFFF);
    TEST_CHAR_WIDTH(0x30000);
    TEST_CHAR_WIDTH(0x3FFFD);
    TEST_CHAR_WIDTH(0x3FFFE);
    TEST_CHAR_WIDTH(0x10FFFF);
    TEST_CHAR_WIDTH(0x110000);
}

#define U16_other      u"A"
#define U16_ZWJ        u"\u200D"
#define U16_XPIC       u"\u00A9"
#define U16_EXTEND_x4  u"\u036F\u036F\u036F\u036F"
#define U16_EXTEND_x8  U16_EXTEND_x4 U16_EXTEND_x4
#define U16_EXTEND_x16 U16_EXTEND_x8 U16_EXTEND_x8

STRF_HD void test_std_width_decrementer()
{
    {   // cover recycle
        strf::detail::std_width_decrementer decr{(strf::width_t::max)()};
        strf::to(decr) (strf::conv(U16_XPIC));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));

        auto width = (strf::width_t::max)() - decr.get_remaining_width();
        TEST_TRUE(width == 1);
    }

    {   // cover recycle with width == 0
        strf::detail::std_width_decrementer decr{3};
        strf::to(decr) (strf::conv(U16_other U16_other U16_other));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_other U16_other U16_other));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_other U16_other U16_other));
        TEST_TRUE(0 == decr.get_remaining_width());
    }
    {   // cover get_remaining_width() when pointer() == buff_
        strf::detail::std_width_decrementer decr{(strf::width_t::max)()};
        strf::to(decr) (strf::conv(U16_other U16_other U16_other));
        decr.recycle();
        auto width = (strf::width_t::max)() - decr.get_remaining_width();
        TEST_TRUE(width == 3);
    }
}

STRF_HD void test_std_width_decrementer_with_pos()
{
    {   // when the remaining width is not zero
        strf::detail::std_width_decrementer_with_pos decr{(strf::width_t::max)()};
        strf::to(decr) (strf::conv(U16_XPIC));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));

        auto res = decr.get_remaining_width_and_codepoints_count();
        auto width = (strf::width_t::max)() - res.remaining_width;
        TEST_TRUE(width == 1);
        TEST_TRUE(res.whole_string_covered);
    }
    {   // when the remaining width is not zero, and recycle() is called
        // immediatelly before_remaining_width_and_codepoints_count
        strf::detail::std_width_decrementer_with_pos decr{(strf::width_t::max)()};
        strf::to(decr) (strf::conv(U16_XPIC));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));
        decr.recycle();
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x8));
        decr.recycle();

        auto res = decr.get_remaining_width_and_codepoints_count();
        auto width = (strf::width_t::max)() - res.remaining_width;
        TEST_TRUE(width == 1);
        TEST_TRUE(res.whole_string_covered);
    }
    {   // when the remaining width is zero, but all input was processed
        strf::detail::std_width_decrementer_with_pos decr{4};
        strf::to(decr) (U"ABC", strf::conv(U16_XPIC));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));

        auto res = decr.get_remaining_width_and_codepoints_count();
        TEST_TRUE(res.remaining_width == 0);
        TEST_TRUE(res.whole_string_covered);
        TEST_EQ(res.codepoints_count, 40);
    }
    {   // when the remaining width is zero, but all input was processed
        // and recycle() is called immediatelly before
        // get_remaining_width_and_codepoints_count
        strf::detail::std_width_decrementer_with_pos decr{4};
        strf::to(decr) (U"ABC", strf::conv(U16_XPIC));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));
        decr.recycle();

        auto res = decr.get_remaining_width_and_codepoints_count();
        TEST_TRUE(res.remaining_width == 0);
        TEST_TRUE(res.whole_string_covered);
        TEST_EQ(res.codepoints_count, 40);
    }
    {   // when the remaining width becames zero before
        // the whole content is processed
        strf::detail::std_width_decrementer_with_pos decr{4};
        strf::to(decr) (U"ABC", strf::conv(U16_XPIC));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));
        strf::to(decr) (U"ABC");
        auto res = decr.get_remaining_width_and_codepoints_count();
        TEST_TRUE(res.remaining_width == 0);
        TEST_TRUE(!res.whole_string_covered);
        TEST_EQ(res.codepoints_count, 40);
    }
    {   // when the remaining width becames zero before
        // the whole content is processed
        // and recycle() is called immediatelly before
        // get_remaining_width_and_codepoints_count
        strf::detail::std_width_decrementer_with_pos decr{4};
        strf::to(decr) (U"ABC", strf::conv(U16_XPIC));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));
        strf::to(decr) (strf::conv(U16_ZWJ U16_XPIC U16_EXTEND_x16));
        strf::to(decr) (U"ABC");
        decr.recycle();
        strf::to(decr) (U"ABC");
        decr.recycle();

        auto res = decr.get_remaining_width_and_codepoints_count();
        TEST_TRUE(res.remaining_width == 0);
        TEST_TRUE(!res.whole_string_covered);
        TEST_EQ(res.codepoints_count, 40);
    }
}

STRF_HD void other_tests()
{
    {
        TEST(u"..a" U16_EXTEND_x8 u"---").with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.'));

        TEST(u".....a" U16_EXTEND_x8).with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.').p(1));

        TEST(u"...a" U16_EXTEND_x8 u"--").with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.').p(3));

        TEST(u"...\u10FF") .with(strf::std_width_calc_t{})
            (strf::right(U'\u10FF', 4, '.'));

        TEST(u"..\u1100") .with(strf::std_width_calc_t{})
            (strf::right(U'\u1100', 4, '.'));
    }
    {
        // empty input to std_width_calc_func
        strf::width_t initial_width = 5;
        char32_t ch;
        auto r = strf::detail::std_width_calc_func(&ch, &ch, initial_width, 0, true);
        TEST_TRUE(r.ptr == &ch);
        TEST_EQ(r.state, 0);
        TEST_TRUE(r.width == initial_width);

        auto r2 = strf::detail::std_width_calc_func(&ch, &ch, initial_width, 0, false);
        TEST_EQ(r2.state, 0);
        TEST_TRUE(r2.width == initial_width);
    }
}

} // unnamed namespace

STRF_TEST_FUNC void test_std_width_calculator()
{
    test_single_char32_width();
    test_std_width_decrementer();
    test_std_width_decrementer_with_pos();
    other_tests();
    test_many_sequences();
}

REGISTER_STRF_TEST(test_std_width_calculator);
