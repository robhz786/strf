//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

class custom_char_wcalc
{
public:
    using category = strf::width_calculator_c;

    template <typename Charset>
    strf::width_t width
        ( const Charset& cs
        , strf::underlying_char_type<Charset::char_size> ch ) const noexcept
    {
        auto ch32 = cs.decode_char(ch);
        return ch32 == U'\u2014' ? 2 : 1;
    }

    template <typename Charset>
    constexpr STRF_HD strf::width_t width
        ( const Charset&
        , strf::width_t
        , const strf::underlying_char_type<Charset::char_size>*
        , std::size_t str_len
        , strf::invalid_seq_policy
        , strf::surrogate_policy ) const noexcept
    {
        return str_len;
    }
};


int main()
{
    strf::dynamic_charset<char> dyn_utf8{strf::utf<char>()};
    strf::dynamic_charset<char> dyn_ascii{strf::ascii<char>()};
    strf::dynamic_charset<char16_t> dyn_utf16{strf::utf<char16_t>()};
    strf::dynamic_charset<char32_t> dyn_utf32{strf::utf<char32_t>()};

    {
        auto a = dyn_utf8;
        auto b = strf::dynamic_charset<char>{strf::utf<char>()};
        TEST_TRUE(a == b);
        a = dyn_ascii;
        TEST_TRUE(a != b);
    }

    TEST("abc ? def").with(dyn_ascii) (strf::cv("abc \xC4\x80 def", dyn_utf8));
    TEST("abc ? def").with(dyn_ascii) (strf::cv("abc \xC4\x80 def", strf::utf<char>()));
    TEST("abc ? def").with(strf::ascii<char>()) (strf::cv("abc \xC4\x80 def", dyn_utf8));
    TEST("abc \xC4\x80 def").with(dyn_utf8) (strf::cv(u"abc \u0100 def"  , dyn_utf16));
    TEST("abc \xC4\x80 def").with(dyn_utf8) (strf::cv(U"abc \u0100 def"  , dyn_utf32));
    TEST(u"abcdef").with(dyn_utf16)         (strf::cv( "abcdef", dyn_ascii));
    TEST( "abcdef").with(dyn_ascii)         (strf::cv(u"abcdef", dyn_utf16));
    TEST(u"abc \u0100 def").with(dyn_utf16) (strf::cv( "abc \xC4\x80 def", dyn_utf8));
    TEST(u"abc \u0100 def").with(dyn_utf16) (strf::cv(U"abc \u0100 def"  , dyn_utf32));
    TEST(U"abc \u0100 def").with(dyn_utf32) (strf::cv( "abc \xC4\x80 def", dyn_utf8));
    TEST(U"abc \u0100 def").with(dyn_utf32) (strf::cv(u"abc \u0100 def"  , dyn_utf16));

    TEST("abc \xC4\x80 def").with(dyn_utf8) (strf::sani("abc \xC4\x80 def"));
    TEST(u"abc \u0100 def").with(dyn_utf16) (strf::sani(u"abc \u0100 def"));
    TEST(U"abc \u0100 def").with(dyn_utf32) (strf::sani(U"abc \u0100 def"));

    {
        auto punct = strf::monotonic_grouping<10>(1)
            .thousands_sep(0xFFFFFFFF)
            .decimal_point(0xFFFFFFFF);

        auto input = strf::fixed(10.5).fill(0xFFFFFF) ^ 6;

        TEST("?10?5?").with(punct, dyn_ascii) (input);
        TEST("\xEF\xBF\xBD" "10\xEF\xBF\xBD" "5\xEF\xBF\xBD").with(punct, dyn_utf8) (input);
        TEST(u"\uFFFD" u"10\uFFFD" u"5\uFFFD").with(punct, dyn_utf16) (input);
        //TEST(U"\uFFFD" U"10\uFFFD" U"5\uFFFD").with(punct, dyn_utf32) (input);
    }
    {
        TEST( "   x").with(custom_char_wcalc{}, dyn_ascii)(strf::fmt('x') > 4);
        TEST( "   x").with(custom_char_wcalc{}, dyn_utf8) (strf::fmt('x') > 4);
        TEST(u"   x").with(custom_char_wcalc{}, dyn_utf16)(strf::fmt(u'x') > 4);
        TEST(U"   x").with(custom_char_wcalc{}, dyn_utf32)(strf::fmt(U'x') > 4);
    }

    return test_finish();
}
