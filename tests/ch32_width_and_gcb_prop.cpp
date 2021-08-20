#include <unicode/uchar.h>
#include <strf/to_cfile.hpp>

// http://www.unicode.org/reports/tr29/tr29-37.html
// https://unicode-org.github.io/icu-docs/apidoc/dev/icu4c/uchar_8h.html

const char* to_str(UGraphemeClusterBreak cat) {
    switch(cat) {
        case U_GCB_OTHER: return "OTHER";
        case U_GCB_CONTROL: return "CONTROL";
        case U_GCB_CR: return "CR";
        case U_GCB_LF: return "LF";
        case U_GCB_EXTEND: return "EXTEND";
        case U_GCB_L: return "L";
        case U_GCB_LV: return "LV";
        case U_GCB_LVT: return "LVT";
        case U_GCB_T: return "T";
        case U_GCB_V: return "V";
        case U_GCB_SPACING_MARK: return "SPACING_MARK";
        case U_GCB_PREPEND: return "PREPEND";
        case U_GCB_REGIONAL_INDICATOR: return "REGIONAL_INDICATOR";
        case U_GCB_E_BASE: return "E_BASE";
        case U_GCB_E_BASE_GAZ: return "E_BASE_GAZ";
        case U_GCB_E_MODIFIER: return "E_MODIFIER";
        case U_GCB_GLUE_AFTER_ZWJ: return "GLUE_AFTER_ZWJ";
        case U_GCB_ZWJ: return "ZWJ";
        default: return "IVALID";
    }
}

enum class category {
    other = U_GCB_OTHER,
    extend = U_GCB_EXTEND,
    control = U_GCB_CONTROL,
    cr = U_GCB_CR,
    lf = U_GCB_LF,
    spacing_mark = U_GCB_SPACING_MARK,
    prepend = U_GCB_PREPEND,
    hangul_l = U_GCB_L,
    hangul_v = U_GCB_V,
    hangul_t = U_GCB_T,
    hangul_lv = U_GCB_LV,
    hangul_lvt = U_GCB_LVT,
    regional_indicator = U_GCB_REGIONAL_INDICATOR,
    zwj = U_GCB_ZWJ,
    extended_picto,
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

bool is_fullwidth(UChar32 ch) {
    // according to http://eel.is/c++draft/format.string.std#11
    return  (0x1100 <= ch && ch <= 0x115F)
        ||  (0x2329 <= ch && ch <= 0x232A)
        ||  (0x2E80 <= ch && ch <= 0x303E)
        ||  (0x3040 <= ch && ch <= 0xA4CF)
        ||  (0xAC00 <= ch && ch <= 0xD7A3)
        ||  (0xF900 <= ch && ch <= 0xFAFF)
        ||  (0xFE10 <= ch && ch <= 0xFE19)
        ||  (0xFE30 <= ch && ch <= 0xFE6F)
        ||  (0xFF00 <= ch && ch <= 0xFF60)
        ||  (0xFFE0 <= ch && ch <= 0xFFE6)
        || (0x1F300 <= ch && ch <= 0x1F64F)
        || (0x1F900 <= ch && ch <= 0x1F9FF)
        || (0x20000 <= ch && ch <= 0x2FFFD)
        || (0x30000 <= ch && ch <= 0x3FFFD);
}

category category_according_to_icu(UChar32 ch) {

    if (u_getIntPropertyValue(ch, UCHAR_EXTENDED_PICTOGRAPHIC)) {
        return category::extended_picto;
    }
    if (ch == 0x11720 || ch == 0x11721) {
        return category::other;
    }
    auto gcb = u_getIntPropertyValue(ch, UCHAR_GRAPHEME_CLUSTER_BREAK);
    return static_cast<category>(gcb);
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
