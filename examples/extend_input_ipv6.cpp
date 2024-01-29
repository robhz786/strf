//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>
#include <strf/to_cfile.hpp>
#include <cstdint>
#include <vector>

#include "../tests/test_utils.hpp" // my own test framework

#if defined(__GNUC__) && (__GNUC__ == 7)
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace xxx {

// -----------------------------------------------------------------------------
// This is the type we are making printable
struct ipv6address
{
    std::uint16_t hextets[8];
};
// -----------------------------------------------------------------------------

// This class evaluates the visibility of each hextet and colon. Implementation
// is left at the end of this file, since it's the focus of this example.
class ipv6address_abbreviation
{
public:
    explicit ipv6address_abbreviation(ipv6address addr);
    constexpr ipv6address_abbreviation() noexcept;

    bool operator==(ipv6address_abbreviation other) const noexcept;

    constexpr  bool hextet_visible(int index) const noexcept;
    constexpr  bool colon_visible(int index) const noexcept;
    constexpr  int visible_hextets_count() const noexcept;
    constexpr  int visible_colons_count() const noexcept;

private:

    unsigned hextets_visibility_bits_;
    unsigned colons_visibility_bits_;
    int visible_hextets_count_;
    int visible_colons_count_;
};

} // namespace xxx

namespace strf {

// -----------------------------------------------------------------------------
// 1. Create some format functions
// -----------------------------------------------------------------------------

//   strf::fmt(addr) : print in the abbrevited form
//                     ( hidding the largest sequence of null hextets, if any )
//  +strf::fmt(addr) : print all hextets ( don't hide any )
// ++strf::fmt(addr) : print all hextets and print them with 4 digits each
// ( yes, I like to abuse on operator overloading sometimes ).

enum class ipv6style{ little, medium, big };

struct ipv6_format_specifier
{
    template <class T>
    class fn
    {
    public:
        constexpr fn() = default;

        template <typename U>
        constexpr explicit fn(const fn<U>& u)
            : style_(u.get_ipv6style())
        {
        }
        constexpr T&& operator++() &&
        {
            style_ = ipv6style::big;
            return static_cast<T&&>(*this);
        }
        constexpr T&& operator+() &&
        {
            style_ = ipv6style::medium;
            return static_cast<T&&>(*this);
        }
        ipv6style get_ipv6style() const
        {
            return style_;
        }

    private:
        ipv6style style_ = ipv6style::little;
    };
};

// -----------------------------------------------------------------------------
// 2. Create the printer class
// -----------------------------------------------------------------------------
template <typename CharT>
class ipv6_printer
{
public:

    template <typename PreMeasurements, typename FPack, typename... T>
    explicit ipv6_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , const strf::printable_with_fmt<T...>& arg )
        : addr_(arg.value())
        , alignment_fmt_(arg.get_alignment_format())
        , lettercase_(strf::use_facet<strf::lettercase_c, xxx::ipv6address>(facets))
        , style_(arg.get_ipv6style())
    {
        auto encoding = use_facet<strf::charset_c<CharT>, xxx::ipv6address>(facets);

        encode_fill_fn_ = encoding.encode_fill_func();
        init_and_premeasure_(pre, encoding);
    }

    void operator()(strf::destination<CharT>& dst) const;

private:

    strf::encode_fill_f<CharT> encode_fill_fn_;
    xxx::ipv6address addr_;
    xxx::ipv6address_abbreviation abbrev_;
    strf::alignment_format alignment_fmt_;
    strf::lettercase lettercase_;
    int fillcount_ = 0;
    ipv6style style_;

    int count_ipv6_characters() const;
    void print_ipv6(strf::destination<CharT>& dst) const;

    template <typename PreMeasurements, typename Charset>
    void init_and_premeasure_(PreMeasurements* pre, Charset charset);
};

template <typename CharT>
template <typename PreMeasurements, typename Charset>
void ipv6_printer<CharT>::init_and_premeasure_(PreMeasurements* pre, Charset charset)
{
    if (style_ == ipv6style::little) {
        abbrev_ = xxx::ipv6address_abbreviation{addr_};
    }
    auto count = count_ipv6_characters();
    if (strf::compare(alignment_fmt_.width, count) <= 0) {
        pre->add_width(static_cast<strf::width_t>(count));
    } else {
        fillcount_ = strf::sat_sub(alignment_fmt_.width, count).round();
        if (PreMeasurements::size_demanded && fillcount_ > 0) {
            pre->add_size(fillcount_ * charset.encoded_char_size(alignment_fmt_.fill));
        }
        pre->add_width(alignment_fmt_.width);
    }
    pre->add_size(count);
}

inline int hex_digits_count(std::uint16_t x)
{
    return x > 0xFFF ? 4 : ( x > 0xFFU ? 3 : ( x > 0xF ? 2 : 1 ) );
}

template <typename CharT>
int ipv6_printer<CharT>::count_ipv6_characters() const
{
    if (style_ == ipv6style::big) {
        return 39;
    }
    auto count = abbrev_.visible_colons_count();
    for (int i = 0; i < 8; ++i) {
        if (abbrev_.hextet_visible(i)) {
            count += hex_digits_count(addr_.hextets[i]);
        }
    }
    return count;
}

template <typename CharT>
void ipv6_printer<CharT>::operator()(strf::destination<CharT>& dst) const
{
    if (fillcount_ <= 0) {
        print_ipv6(dst);
    } else {
        switch(alignment_fmt_.alignment) {
            case strf::text_alignment::left:
                print_ipv6(dst);
                encode_fill_fn_(dst, fillcount_, alignment_fmt_.fill);
                break;
            case strf::text_alignment::right:
                encode_fill_fn_(dst, fillcount_, alignment_fmt_.fill);
                print_ipv6(dst);
                break;
            default:{
                assert(alignment_fmt_.alignment == strf::text_alignment::center);
                const auto halfcount = fillcount_ / 2;
                encode_fill_fn_(dst, halfcount, alignment_fmt_.fill);
                print_ipv6(dst);
                encode_fill_fn_(dst, fillcount_ - halfcount, alignment_fmt_.fill);
            }
        }
    }
}

template <typename CharT>
void ipv6_printer<CharT>::print_ipv6(strf::destination<CharT>& dst) const
{
    const int precision = (style_ == ipv6style::big ? 4 : 0);
    for (int i = 0; i < 8; ++i) {
        if (abbrev_.hextet_visible(i)) {
            strf::to(dst) (lettercase_, strf::hex(addr_.hextets[i]).p(precision));
        }
        if (abbrev_.colon_visible(i)) {
            strf::put(dst, static_cast<CharT>(':'));
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Specialize printable_traits template
// -----------------------------------------------------------------------------

template <>
struct printable_traits<xxx::ipv6address> {
    using representative_type = xxx::ipv6address;
    using forwarded_type = xxx::ipv6address;
    using format_specifiers = strf::tag<ipv6_format_specifier, strf::alignment_format_specifier>;

    template <typename CharT, typename PreMeasurements, typename FPack, typename... Fmts>
    static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , const strf::printable_with_fmt<printable_traits, Fmts...>& arg )
    {
        return ipv6_printer<CharT>{pre, fp, arg};
    }
};

} // namespace strf

void tests()
{
    xxx::ipv6address addr0{{0, 0, 0, 0, 0, 0, 0, 0}};

    // test formatting
    TEST("::") (addr0);
    TEST("0:0:0:0:0:0:0:0") (+strf::fmt(addr0));
    TEST("0000:0000:0000:0000:0000:0000:0000:0000") (++strf::fmt(addr0))     ;

    TEST("                                       ::") (strf::fmt(addr0) > 41)  ;
    TEST("                          0:0:0:0:0:0:0:0") (+strf::fmt(addr0) > 41) ;
    TEST("  0000:0000:0000:0000:0000:0000:0000:0000") (++strf::fmt(addr0) > 41);

    TEST("::                                       ") (strf::fmt(addr0) < 41)  ;
    TEST("0:0:0:0:0:0:0:0                          ") (+strf::fmt(addr0) < 41) ;
    TEST("0000:0000:0000:0000:0000:0000:0000:0000  ") (++strf::fmt(addr0) < 41);

    TEST("                   ::                    ") (strf::fmt(addr0) ^ 41)  ;
    TEST("             0:0:0:0:0:0:0:0             ") (+strf::fmt(addr0) ^ 41) ;
    TEST(" 0000:0000:0000:0000:0000:0000:0000:0000 ") (++strf::fmt(addr0) ^ 41);

    // test abbreviation
    TEST("::aa:bb:cc:dd:0:0") (xxx::ipv6address{{   0,    0, 0xaa, 0xbb, 0xcc, 0xdd,    0,    0}});
    TEST("0:0:0:aa::bb")      (xxx::ipv6address{{   0,    0,    0, 0xaa,    0,    0,    0, 0xbb}});
    TEST("0:0:aa::bb:0:0")    (xxx::ipv6address{{   0,    0, 0xaa,    0,    0, 0xbb,    0,    0}});
    TEST("0:0:aa:0:bb::cc")   (xxx::ipv6address{{   0,    0, 0xaa,    0, 0xbb,    0,    0, 0xcc}});
    TEST("0:0:0:aa:bb::")     (xxx::ipv6address{{   0,    0,    0, 0xaa, 0xbb,    0,    0,    0}});
    TEST("::aa:bb:0:0:cc")    (xxx::ipv6address{{   0,    0,    0, 0xaa, 0xbb,    0,    0, 0xcc}});
    TEST("0:0:aa:bb::")       (xxx::ipv6address{{   0,    0, 0xaa, 0xbb,    0,    0,    0,    0}});
    TEST("::aa:bb:cc:0:0")    (xxx::ipv6address{{   0,    0,    0, 0xaa, 0xbb, 0xcc,    0,    0}});
    TEST("0:0:aa:bb:cc::dd")  (xxx::ipv6address{{   0,    0, 0xaa, 0xbb, 0xcc,    0,    0, 0xdd}});
    TEST("0:aa::bb:0:0:cc")   (xxx::ipv6address{{   0, 0xaa,    0,    0, 0xbb,    0,    0, 0xcc}});
    TEST("aa::bb:0:0:cc:0")   (xxx::ipv6address{{0xaa,    0,    0, 0xbb,    0,    0, 0xcc,    0}});
    TEST("aa:0:0:bb::cc")     (xxx::ipv6address{{0xaa,    0,    0, 0xbb,    0,    0,    0, 0xcc}});
    TEST("0:aa::bb:0:0:cc")   (xxx::ipv6address{{   0, 0xaa,    0,    0, 0xbb,    0,    0, 0xcc}});
    TEST("0:aa:0:0:bb::")     (xxx::ipv6address{{   0, 0xaa,    0,    0, 0xbb,    0,    0,    0}});
    TEST("0:aa:0:bb:0:cc::")  (xxx::ipv6address{{   0, 0xaa,    0, 0xbb,    0, 0xcc,    0,    0}});
    TEST("aa::")              (xxx::ipv6address{{0xaa,    0,    0,    0,    0,    0,    0,    0}});
    TEST("0:aa::")            (xxx::ipv6address{{   0, 0xaa,    0,    0,    0,    0,    0,    0}});
    TEST("0:0:0:aa::")        (xxx::ipv6address{{   0,    0,    0, 0xaa,    0,    0,    0,    0}});
    TEST("0:0:0:aa:bb::")     (xxx::ipv6address{{   0,    0,    0, 0xaa, 0xbb,    0,    0,    0}});

    // test lettercase
    xxx::ipv6address addr_aabb12{{0xaa, 0xbb, 1, 2, 0, 0, 0, 0}};

    TEST("aa:bb:1:2::") (strf::lowercase, addr_aabb12);
    TEST("AA:BB:1:2::") (strf::uppercase, addr_aabb12);
    TEST("AA:BB:1:2::") (strf::mixedcase, addr_aabb12);

    TEST("....aa:bb:1:2::") (strf::lowercase, strf::right(addr_aabb12, 15, '.'));
    TEST("....AA:BB:1:2::") (strf::uppercase, strf::right(addr_aabb12, 15, '.'));
    TEST("....AA:BB:1:2::") (strf::mixedcase, strf::right(addr_aabb12, 15, '.'));
}

int main()
{
    strf::narrow_cfile_writer<char, 512> msg_dst(stdout);
    const test_utils::test_messages_destination_guard g(msg_dst);

    tests();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(msg_dst, "All test passed!\n");
    } else {
        strf::to(msg_dst) (err_count, " tests failed!\n");
    }
    return err_count;
}

namespace xxx {

// -----------------------------------------------------------------------------
// implementation of ipv6address_abbreviation member functions:
// -----------------------------------------------------------------------------
ipv6address_abbreviation::ipv6address_abbreviation(ipv6address addr)
{
    // Don't mind the mess

    int middle_zeros_start = 0;
    int middle_zeros_count = 0;
    int trailing_zeros_start = 0;

    if (addr.hextets[0] == 0 && addr.hextets[1] == 0) {
        int zi = 2;
        while(addr.hextets[zi] == 0) {
            ++zi;
            if (zi == 8) {
                // address is "::"
                visible_hextets_count_ = 0;
                visible_colons_count_ = 2;
                hextets_visibility_bits_ = 0x0;
                colons_visibility_bits_ = 0x3;
                return;
            }
        }
        const auto leading_zeros_count = zi;
        if (leading_zeros_count < 4) {
            while(1) {
                if (++zi == 6) {
                    goto hide_leading_zeros;
                }
                if (addr.hextets[zi] == 0 && addr.hextets[zi + 1] == 0) {
                    break;
                }
            }
            middle_zeros_start = zi;
            zi += 2;
            while(addr.hextets[zi] == 0){
                if (++zi == 8) {
                    trailing_zeros_start = middle_zeros_start;
                    goto hide_trailing_zeros;
                }
            }
            middle_zeros_count = zi - middle_zeros_start;
            if (middle_zeros_count >= leading_zeros_count) {
                goto hide_middle_zeros;
            }
            hide_leading_zeros:
            visible_hextets_count_ = 8 - leading_zeros_count;
            visible_colons_count_ = visible_hextets_count_ + 1;
            hextets_visibility_bits_ = 0xFFU << leading_zeros_count;
            colons_visibility_bits_ = 0x7FU & (0x7FU << (leading_zeros_count - 2));
        }
    } else {
        int zi = 2;
        while(1) {
            if (addr.hextets[zi - 1] == 0 && addr.hextets[zi] == 0) {
                break;
            }
            if (++zi == 8) {
                visible_hextets_count_ = 8;
                visible_colons_count_ = 7;
                hextets_visibility_bits_ = 0xFF;
                colons_visibility_bits_ = 0x7F;
            }
        }
        middle_zeros_start = zi - 1;
        do {
            if (++zi == 8) {
                trailing_zeros_start = middle_zeros_start;
                goto hide_trailing_zeros;
            }
        } while(addr.hextets[zi] == 0);
        middle_zeros_count = zi - middle_zeros_start;
        if (middle_zeros_count == 2) {
            while(1) {
                if (++zi >= 6) {
                    goto hide_middle_zeros;
                }
                if (addr.hextets[zi] == 0
                 && addr.hextets[zi + 1] == 0
                 && addr.hextets[zi + 2] == 0 ) {
                    if (addr.hextets[7] == 0) {
                        trailing_zeros_start = zi;
                        goto hide_trailing_zeros;
                    } else {
                        middle_zeros_start = zi;
                        middle_zeros_count = 3;
                        goto hide_middle_zeros;
                    }
                }
            }
        }
        goto hide_middle_zeros;
    }
    return;

    hide_middle_zeros:
    visible_hextets_count_ = 8 - middle_zeros_count;
    visible_colons_count_ = visible_hextets_count_;
    hextets_visibility_bits_ = ~(~(0xFFU << middle_zeros_count) << middle_zeros_start);
    colons_visibility_bits_  = 0x7FU & ~(~(0xFFU << (middle_zeros_count - 1)) << middle_zeros_start);
    return;

    hide_trailing_zeros:
    auto trailing_zeros_count = 8 - trailing_zeros_start;
    visible_hextets_count_ = trailing_zeros_start;
    visible_colons_count_  = trailing_zeros_start + 1;
    hextets_visibility_bits_ = 0xFFU >>  trailing_zeros_count;
    colons_visibility_bits_  = 0xFFU >> (trailing_zeros_count - 1);
}
constexpr ipv6address_abbreviation::ipv6address_abbreviation() noexcept
    : hextets_visibility_bits_(0xFF)
    , colons_visibility_bits_(0x7F)
    , visible_hextets_count_(8)
    , visible_colons_count_(7)
{
}
bool ipv6address_abbreviation::operator==(ipv6address_abbreviation other) const noexcept
{
    return ( hextets_visibility_bits_ == other.hextets_visibility_bits_
          && colons_visibility_bits_ == other.colons_visibility_bits_
          && visible_hextets_count_ == other.visible_hextets_count_
          && visible_colons_count_ == other.visible_colons_count_ );
}
constexpr bool ipv6address_abbreviation::hextet_visible(int index) const noexcept
{
    return 0 <= index && index < 8 && (hextets_visibility_bits_ & (1U << index));
}
constexpr bool ipv6address_abbreviation::colon_visible(int index) const noexcept
{
    return 0 <= index && index < 7 && (colons_visibility_bits_ & (1U << index));
}
constexpr int ipv6address_abbreviation::visible_hextets_count() const noexcept
{
    return visible_hextets_count_;
}
constexpr int ipv6address_abbreviation::visible_colons_count() const noexcept
{
    return visible_colons_count_;
}

} // namespace xxx
