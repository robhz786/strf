//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

using test_utils::simple_array;
using test_utils::make_simple_array;
using test_utils::span;

namespace {

template <std::size_t N>
using size_array  = simple_array<std::size_t, N>;

using char32_range = strf::detail::simple_string_view<char32_t>;

STRF_HD auto hex_ch32(char32_t ch)
    -> decltype(strf::join("", strf::hex(0U).p(6)))
{
    return strf::join("U+", strf::hex(static_cast<unsigned>(ch)).p(4));
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
    auto *obtained_sizes_it = size_buffer;
    const char32_t* grapheme_begin = chars.begin();
    for(const auto *ptr = chars.begin(); ptr < chars.end(); ++ptr) {
        auto r = strf::detail::std_width_calc_func(ptr, ptr + 1, remaining_width, state, false);
        if (ptr == chars.begin()) {
            remaining_width = r.remaining_width;
        } else if (r.remaining_width != remaining_width) {
            *obtained_sizes_it++ = strf::detail::safe_cast_size_t(ptr - grapheme_begin);
            grapheme_begin = ptr;
            remaining_width = r.remaining_width;
        }
        state = r.state;
    }
    *obtained_sizes_it++ = strf::detail::safe_cast_size_t(chars.end() - grapheme_begin);

    auto obtained_width = (initial_width - remaining_width).round();
    const span<const std::size_t> obtained_grapheme_sizes{size_buffer, obtained_sizes_it};

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
    auto obtained_width = (minuend - r.remaining_width).round();

    if (obtained_width != expected_width) {
        test_utils::test_failure
            ( filename, line, funcname, "Obtained width = ", obtained_width
            , " (expected ", expected_width, ")\n sequence = "
            , strf::separated_range(chars, " ", hex_ch32) );
    }
}

STRF_HD void test_pos
    ( const char* filename
    , int line
    , const char* funcname
    , std::ptrdiff_t expected_pos
    , strf::width_t max_width
    , std::initializer_list<char32_t> chars )
{
    auto r = strf::detail::std_width_calc_func(chars.begin(), chars.end(), max_width, 0, true);
    const std::ptrdiff_t obtained_pos = r.ptr - chars.begin();
    if (obtained_pos != expected_pos) {
        test_utils::test_failure
            ( filename, line, funcname, "Obtained pos = ", obtained_pos
            , " (expected ", expected_pos, ")\n sequence = "
            , strf::separated_range(chars, " ", hex_ch32) );
    }
}

template <std::size_t NumSizes, std::size_t NumChars>
STRF_HD void test_width
    ( const char* filename
    , int linenumber
    , const char* funcname
    , int expected_width
    , const size_array<NumSizes>& expected_grapheme_sizes_
    , simple_array<char32_t, NumChars> chars )
{
    const span<const std::size_t> expected_grapheme_sizes{expected_grapheme_sizes_.elements, NumSizes};
    test_width_one_pass( filename, linenumber, funcname, expected_width
                       , expected_grapheme_sizes
                       , chars );
    std::size_t size_buff[NumChars];
    test_width_char_by_char( filename, linenumber, funcname, expected_width
                           , size_buff
                           , expected_grapheme_sizes
                           , chars );
}

template <typename... T>
constexpr STRF_HD std::size_t count_args(const T&...) {
    return sizeof...(T);
}

#define TEST_WIDTH(EXPECTED_WIDTH, EXPECTED_GRAPHEME_SIZES, ...)          \
    test_width( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
              , (EXPECTED_WIDTH)                                          \
              , EXPECTED_GRAPHEME_SIZES                                   \
              , make_simple_array<char32_t>(__VA_ARGS__) );

#define TEST_POS(EXPECTED_POS, MAX_WIDTH, ...) \
    test_pos( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (EXPECTED_POS)  \
            , (MAX_WIDTH), {__VA_ARGS__});

template <typename... Args>
STRF_HD size_array<sizeof...(Args)> sizes(Args... args) {
    return size_array<sizeof...(Args)>{{static_cast<std::size_t>(args)...}};
}

STRF_HD void test_many_sequences() // NOLINT(google-readability-function-size,hicpp-function-size)
{
    constexpr char32_t control = 0x0001;
    constexpr char32_t cr = 0x000D;
    constexpr char32_t lf = 0x000A;
    constexpr char32_t prepend = 0x0605;
    constexpr char32_t hangul_l = 0xA97C;
    constexpr char32_t hangul_l_w2 = 0x115F;
    constexpr char32_t hangul_v = 0x11A7;
    constexpr char32_t hangul_t = 0x11FF;
    constexpr char32_t hangul_lv_w2 = 0xAC00;
    // need three samples of LVT to achieve full coverage
    constexpr char32_t hangul_lvt_w2   = 0xD789;
    constexpr char32_t hangul_lvt_w2_2 = 0xAC01;
    constexpr char32_t hangul_lvt_w2_3 = 0xAC08;
    constexpr char32_t ri = 0x1F1E6;
    constexpr char32_t ext_pict = 0x00A9;
    constexpr char32_t ext_pict_w2 = 0x3030;
    constexpr char32_t zwj = 0x200D;
    constexpr char32_t spacing_mark = 0x0903;
    constexpr char32_t extend = 0x036F;
    constexpr char32_t extend_w2 = 0x302F;
    constexpr char32_t other = 0x0020;
    constexpr char32_t other_w2 = 0x232A;

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
    TEST_WIDTH(3, sizes(2, 1),       other, extend, hangul_lvt_w2_2);
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
    TEST_WIDTH(3, sizes(1, 1),       cr, hangul_lvt_w2_3);
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

#define U32_other      U"A"
#define U32_ZWJ        U"\u200D"
#define U32_XPIC       U"\u00A9"
#define U32_EXTEND_x4  U"\u036F\u036F\u036F\u036F"
#define U32_EXTEND_x8  U32_EXTEND_x4 U32_EXTEND_x4
#define U32_EXTEND_x16 U32_EXTEND_x8 U32_EXTEND_x8

// STRF_HD void puts(strf::transcode_dest<char32_t>& dest, const char32_t* str) {
//     dest.write(str, strf::detail::str_length(str));
// }

// STRF_HD void test_std_width_decrementer()
// {
//     {   // cover flush()
//         strf::detail::std_width_decrementer decr{(strf::width_t::max)()};
//         puts(decr, U32_XPIC);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);
//         decr.flush();
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);
//         decr.flush();
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);

//         auto width = (strf::width_t::max)() - decr.get_remaining_width();
//         TEST_TRUE(width == 1);
//     }

//     {   // cover flush() with width == 0
//         strf::detail::std_width_decrementer decr{3};
//         puts(decr, U32_other U32_other U32_other);
//         decr.flush();
//         puts(decr, U32_other U32_other U32_other);
//         decr.flush();
//         puts(decr, U32_other U32_other U32_other);
//         TEST_TRUE(0 == decr.get_remaining_width());
//     }
//     {   // cover get_remaining_width() when buffer_ptr() == buff_
//         strf::detail::std_width_decrementer decr{(strf::width_t::max)()};
//         puts(decr, U32_other U32_other U32_other);
//         decr.flush();
//         auto width = (strf::width_t::max)() - decr.get_remaining_width();
//         TEST_TRUE(width == 3);
//     }
// }

// STRF_HD void test_std_width_decrementer_with_pos()
// {
//     {   // when the remaining width is not zero
//         strf::detail::std_width_decrementer_with_pos decr{(strf::width_t::max)()};
//         puts(decr, U32_XPIC);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);
//         decr.flush();
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);
//         decr.flush();
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);

//         auto res = decr.get_remaining_width_and_codepoints_count();
//         auto width = (strf::width_t::max)() - res.remaining_width;
//         TEST_TRUE(width == 1);
//         TEST_TRUE(res.whole_string_covered);
//     }
//     {   // when the remaining width is not zero, and recycle() is called
//         // immediatelly before_remaining_width_and_codepoints_count
//         strf::detail::std_width_decrementer_with_pos decr{(strf::width_t::max)()};
//         puts(decr, U32_XPIC);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);
//         decr.flush();
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);
//         decr.flush();
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x8);
//         decr.flush();

//         auto res = decr.get_remaining_width_and_codepoints_count();
//         auto width = (strf::width_t::max)() - res.remaining_width;
//         TEST_TRUE(width == 1);
//         TEST_TRUE(res.whole_string_covered);
//     }
//     {   // when the remaining width is zero, but all input was processed
//         strf::detail::std_width_decrementer_with_pos decr{4};
//         puts(decr, U"ABC");
//         puts(decr, U32_XPIC);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);

//         auto res = decr.get_remaining_width_and_codepoints_count();
//         TEST_TRUE(res.remaining_width == 0);
//         TEST_TRUE(res.whole_string_covered);
//         TEST_EQ(res.codepoints_count, 40);
//     }
//     {   // when the remaining width is zero, but all input was processed
//         // and recycle() is called immediatelly before
//         // get_remaining_width_and_codepoints_count
//         strf::detail::std_width_decrementer_with_pos decr{4};
//         puts(decr, U"ABC");
//         puts(decr, U32_XPIC);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);
//         decr.flush();

//         auto res = decr.get_remaining_width_and_codepoints_count();
//         TEST_TRUE(res.remaining_width == 0);
//         TEST_TRUE(res.whole_string_covered);
//         TEST_EQ(res.codepoints_count, 40);
//     }
//     {   // when the remaining width becames zero before
//         // the whole content is processed
//         strf::detail::std_width_decrementer_with_pos decr{4};
//         puts(decr, U"ABC");
//         puts(decr, U32_XPIC);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);
//         puts(decr, U"ABC");
//         auto res = decr.get_remaining_width_and_codepoints_count();
//         TEST_TRUE(res.remaining_width == 0);
//         TEST_TRUE(!res.whole_string_covered);
//         TEST_EQ(res.codepoints_count, 40);
//     }
//     {   // when the remaining width becames zero before
//         // the whole content is processed
//         // and recycle() is called immediatelly before
//         // get_remaining_width_and_codepoints_count
//         strf::detail::std_width_decrementer_with_pos decr{4};
//         puts(decr, U"ABC");
//         puts(decr, U32_XPIC);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);
//         puts(decr, U32_ZWJ U32_XPIC U32_EXTEND_x16);
//         puts(decr, U"ABC");
//         decr.flush();
//         puts(decr, U"ABC");
//         decr.flush();

//         auto res = decr.get_remaining_width_and_codepoints_count();
//         TEST_TRUE(res.remaining_width == 0);
//         TEST_TRUE(!res.whole_string_covered);
//         TEST_EQ(res.codepoints_count, 40);
//     }
// }

STRF_HD void other_tests()
{
    {
        TEST(u"..a" U16_EXTEND_x8 u"---").with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.'));

        TEST(u"......").with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.').p(0));

        TEST(u".....a" U16_EXTEND_x8).with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.').p(1));

        TEST(u"...a" U16_EXTEND_x8 u"--").with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.').p(3));

        TEST(u"..a" U16_EXTEND_x8 u"---").with(strf::std_width_calc_t{})
            (strf::right(u"a" U16_EXTEND_x8 u"---", 6, '.').p(4));

        TEST(u"...\u10FF") .with(strf::std_width_calc_t{})
            (strf::right(U'\u10FF', 4, '.'));

        TEST(u"..\u1100") .with(strf::std_width_calc_t{})
            (strf::right(U'\u1100', 4, '.'));
    }
    {
        // empty input to std_width_calc_func
        const strf::width_t initial_width = 5;
        const char32_t ch = 'X';
        auto r = strf::detail::std_width_calc_func(&ch, &ch, initial_width, 0, true);
        TEST_TRUE(r.ptr == &ch);
        TEST_EQ(r.state, 0);
        TEST_TRUE(r.remaining_width == initial_width);

        auto r2 = strf::detail::std_width_calc_func(&ch, &ch, initial_width, 0, false);
        TEST_EQ(r2.state, 0);
        TEST_TRUE(r2.remaining_width == initial_width);
    }
}

} // unnamed namespace

STRF_TEST_FUNC void test_std_width_calculator()
{
    test_single_char32_width();
    // test_std_width_decrementer();
    // test_std_width_decrementer_with_pos();
    other_tests();
    test_many_sequences();
}

REGISTER_STRF_TEST(test_std_width_calculator)
