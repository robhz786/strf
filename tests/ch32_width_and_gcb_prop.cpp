#include <unicode/uchar.h>
#include <strf/to_cfile.hpp>

// http://www.unicode.org/reports/tr29/tr29-37.html
// https://unicode-org.github.io/icu-docs/apidoc/dev/icu4c/uchar_8h.html

enum class category {
    other,
    extend,
    control,
    cr,
    lf,
    spacing_mark,
    prepend,
    hangul_l,
    hangul_v,
    hangul_t,
    hangul_lv,
    hangul_lvt,
    regional_indicator,
    extended_picto,
    zwj,
};

const char* to_str(category cat) {
    switch(cat) {
        case category::other: return "other";
        case category::extend: return "extend";
        case category::control: return "control";
        case category::cr: return "cr";
        case category::lf: return "lf";
        case category::spacing_mark: return "spacing_mark";
        case category::prepend: return "prepend";
        case category::hangul_l: return "hangul_l";
        case category::hangul_v: return "hangul_v";
        case category::hangul_t: return "hangul_t";
        case category::hangul_lv: return "hangul_lv";
        case category::hangul_lvt: return "hangul_lvt";
        case category::regional_indicator: return "regional_indicator";
        case category::extended_picto: return "extended_picto";
        case category::zwj: return "zwj";

    }
    return "INVALID_CATEGORY";
}

inline bool is_zwj(UChar32 ch) {
    return ch == 0x200D;
}

inline bool is_extend(UChar32 ch) {
    return u_getIntPropertyValue(ch, UCHAR_GRAPHEME_EXTEND)
        || u_getIntPropertyValue(ch, UCHAR_EMOJI_MODIFIER);
}

inline bool is_extended_picto(UChar32 ch) {
    return u_getIntPropertyValue(ch, UCHAR_EXTENDED_PICTOGRAPHIC);
}

inline bool is_regional_indicator(UChar32 ch) {
    return u_getIntPropertyValue(ch, UCHAR_REGIONAL_INDICATOR);
}

bool is_spacing_mark(UChar32 ch) {
    auto gra_clu_brk = static_cast<UGraphemeClusterBreak>
        ( u_getIntPropertyValue(ch, UCHAR_GRAPHEME_CLUSTER_BREAK) );

    if (gra_clu_brk != U_GCB_EXTEND) {

        if (ch == 0x0E33 || ch == 0x0EB3) {
            return true;
        }
        auto gencat = static_cast<UCharCategory>
            (u_getIntPropertyValue(ch, UCHAR_GENERAL_CATEGORY));

        if ( gencat == U_COMBINING_SPACING_MARK
          && ch != 0x102B
          && ch != 0x102C
          && ch != 0x1038
          && ! (0x1062 <= ch && ch <= 0x1064)
          && ! (0x1067 <= ch && ch <= 0x106D)
          && ch != 0x1083
          && ! (0x1087 <= ch && ch <= 0x108C)
          && ch != 0x108F
          && ! (0x109A <= ch && ch <= 0x109C)
          && ch != 0x1A61
          && ch != 0x1A63
          && ch != 0x1A64
          && ch != 0xAA7B
          && ch != 0xAA7D
          && ch != 0x11720
          && ch != 0x11721 )
        {
            return true;
        }
    }
    return false;
}

bool is_prepend(UChar32 ch) {
    if (u_getIntPropertyValue(ch, UCHAR_PREPENDED_CONCATENATION_MARK))
        return true;

    auto insc = static_cast<UIndicSyllabicCategory>
        (u_getIntPropertyValue(ch, UCHAR_INDIC_SYLLABIC_CATEGORY));

    return insc == U_INSC_CONSONANT_PRECEDING_REPHA
        || insc == U_INSC_CONSONANT_PREFIXED;
}

bool is_hangul(UChar32 ch) {
    auto hst = static_cast<UHangulSyllableType>
        (u_getIntPropertyValue(ch, UCHAR_HANGUL_SYLLABLE_TYPE));

    return ( hst == U_HST_LEADING_JAMO
          || hst == U_HST_VOWEL_JAMO
          || hst == U_HST_TRAILING_JAMO
          || hst == U_HST_LV_SYLLABLE
          || hst == U_HST_LVT_SYLLABLE );
}

bool is_control(UChar32 ch) {
    if (ch == 0x200C || ch == 0x200D) {
        return false;
    }
    if (u_getIntPropertyValue(ch, UCHAR_PREPENDED_CONCATENATION_MARK)) {
        return false;
    }
    auto gencat = static_cast<UCharCategory>(u_getIntPropertyValue(ch, UCHAR_GENERAL_CATEGORY));

    return gencat == U_LINE_SEPARATOR || gencat == U_PARAGRAPH_SEPARATOR
        || gencat == U_CONTROL_CHAR   || gencat == U_FORMAT_CHAR
        || ( gencat == U_UNASSIGNED
          && u_getIntPropertyValue(ch, UCHAR_DEFAULT_IGNORABLE_CODE_POINT));
}

bool is_fullwidth(UChar32 ch) {
    // according to http://eel.is/c++draft/format.string.std#11
    return (0x1100 <= ch && ch <= 0x115F)
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
        || (0x30000 <= ch && ch <= 0x3FFFD);
}

category category_according_to_icu(UChar32 ch) {
    int is_control = ::is_control(ch);
    int is_extend = ::is_extend(ch);

    if (ch == 0x000D) {
        return category::cr;
    }
    if (ch == 0x000A) {
        return category::lf;
    }
    if (is_zwj(ch)) {
        return category::zwj;
    }
    if (is_spacing_mark(ch)) {
        return category::spacing_mark;
    }
    if (is_regional_indicator(ch)) {
        return category::regional_indicator;
    }
    if (is_extended_picto(ch)) {
        return category::extended_picto;
    }
    if (is_prepend(ch)) {
        return category::prepend;
    }
    auto hst = static_cast<UHangulSyllableType>
        (u_getIntPropertyValue(ch, UCHAR_HANGUL_SYLLABLE_TYPE));
    if (hst == U_HST_LEADING_JAMO) {
        return category::hangul_l;
    }
    if (hst == U_HST_VOWEL_JAMO) {
        return category::hangul_v;
    }
    if (hst == U_HST_TRAILING_JAMO) {
        return category::hangul_t;
    }
    if (hst == U_HST_LV_SYLLABLE) {
        return category::hangul_lv;
    }
    if (hst == U_HST_LVT_SYLLABLE) {
        return category::hangul_lvt;
    }
    if (is_extend) {
        return category::extend;
    }
    if (is_control) {
        return category::control;
    }
    return category::other;
}

struct codepoint_properties {
    category cat;
    strf::width_t width;
};

codepoint_properties props_according_to_strf(UChar32 ch) {
    using namespace strf::width_literal;
    strf::width_t ch_width = 1_w;
    if (ch <= 0x007E) {
        if (0x20 <= ch) {
            handle_other:
            return {category::other, ch_width};
        }
        if (0x000D == ch) { // CR
            return {category::cr, 1};
        }
        if (0x000A == ch) { // LF
            return {category::lf, 1};
        }
        handle_control:
        return {category::control, ch_width};
    }

#include <strf/detail/ch32_width_and_gcb_prop>
    goto force_failure;

    handle_zwj:
    return {category::zwj, ch_width};

    handle_extend:
    handle_extend_and_control:
    return {category::extend, ch_width};

    handle_spacing_mark:
    return {category::spacing_mark, ch_width};

    handle_prepend:
    return {category::prepend, ch_width};

    handle_regional_indicator:
    return {category::regional_indicator, ch_width};

    handle_extended_picto:
    return {category::extended_picto, ch_width};

    handle_hangul_l:
    return {category::hangul_l, ch_width};

    handle_hangul_v:
    return {category::hangul_v, ch_width};

    handle_hangul_t:
    return {category::hangul_t, ch_width};

    handle_hangul_lv_or_lvt:
    if ( ch <= 0xD788 // && 0xAC00 <= ch
         && 0 == (ch & 3)
         && 0 == ((ch - 0xAC00) >> 2) % 7)
    {   // LV
        return {category::hangul_lv, ch_width};
    } else {
        return {category::hangul_lvt, ch_width};
    }

    force_failure:
    return {static_cast<category>(100), 100};
}

int main() {
    using namespace strf::width_literal;
    strf::narrow_cfile_writer<char, 1000> out(stdout);
    auto print = strf::to(out).with(strf::uppercase);

    constexpr long max_failures_count = 500;
    long failures_count = 0;
    for (std::uint32_t ch = 0; ch < 0x110000; ++ch) {
        auto obtained = props_according_to_strf(ch);

        int obtained_width = obtained.width.round();
        auto obtained_category  = obtained.cat;

        auto expected_width = is_fullwidth(ch) ? 2 : 1;
        auto expected_category = category_according_to_icu(ch);

        if (obtained_width != expected_width || obtained_category != expected_category) {
            print("For codepoint U+", strf::hex(ch).p(4), ":");
            ++failures_count;
        }
        if (obtained_category != expected_category) {
            print( "    Got category \'", to_str(obtained_category)
                 , "\', expected \'", to_str(expected_category), "\'.\n");
        }
        if (obtained_width != expected_width) {
            print("     Got width = ", obtained_width, " (expected ", expected_width, ").\n");
        }
        if (failures_count >= max_failures_count) {
            print("\nToo many failures. Stopping before end.\n");
            break;
        }
    }
    if (failures_count == 0){
        print("All tests passed!\n");
    }
    return failures_count;
}
