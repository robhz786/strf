//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>
#include <strf/to_cfile.hpp>
#include <cstdint>
#include <vector>

#include "../tests/test_utils.hpp" // my own test framework

#if defined(__GNUC__)
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
    constexpr ipv6address_abbreviation(const ipv6address_abbreviation&) noexcept = default;
    constexpr ipv6address_abbreviation& operator=(const ipv6address_abbreviation&) noexcept = default;
    bool operator==(ipv6address_abbreviation other) const noexcept;

    constexpr  bool hextet_visible(unsigned index) const noexcept;
    constexpr  bool colon_visible(unsigned index) const noexcept;
    constexpr  std::uint8_t visible_hextets_count() const noexcept;
    constexpr  std::uint8_t visible_colons_count() const noexcept;

private:

    std::uint8_t hextets_visibility_bits_;
    std::uint8_t colons_visibility_bits_;
    std::uint8_t visible_hextets_count_;
    std::uint8_t visible_colons_count_;
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

struct ipv6_formatter
{
    template <class T>
    class fn
    {
    public:
        constexpr fn() = default;
        constexpr fn(const fn&) = default;

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
class ipv6_printer: public strf::arg_printer<CharT>
{
public:

    template <typename... T>
    ipv6_printer(strf::usual_arg_printer_input<CharT, T...> input)
        : addr_(input.arg.value())
        , alignment_fmt_(input.arg.get_alignment_format())
        , lettercase_(strf::use_facet<strf::lettercase_c, xxx::ipv6address>(input.facets))
        , style_(input.arg.get_ipv6style())
    {
        auto encoding = use_facet<strf::charset_c<CharT>, xxx::ipv6address>(input.facets);

        encode_fill_fn_ = encoding.encode_fill_func();
        init_(input.preview, encoding);
    }

    void print_to(strf::print_dest<CharT>& dest) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_fn_;
    xxx::ipv6address addr_;
    xxx::ipv6address_abbreviation abbrev_;
    const strf::alignment_format alignment_fmt_;
    const strf::lettercase lettercase_;
    std::uint16_t fillcount_ = 0;
    ipv6style style_;

    std::uint16_t count_ipv6_characters() const;
    void print_ipv6(strf::print_dest<CharT>& dest) const;

    template <typename Preview, typename Charset>
    void init_(Preview& preview, Charset charset);
};

template <typename CharT>
template <typename Preview, typename Charset>
void ipv6_printer<CharT>::init_(Preview& preview, Charset charset)
{
    if (style_ == ipv6style::little) {
        abbrev_ = xxx::ipv6address_abbreviation{addr_};
    }
    auto count = count_ipv6_characters();
    if (count > alignment_fmt_.width) {
        preview.subtract_width(count);
    } else {
        fillcount_ = (alignment_fmt_.width - count).round();
        if (Preview::size_required && fillcount_ > 0) {
            preview.add_size(fillcount_ * charset.encoded_char_size(alignment_fmt_.fill));
        }
        preview.subtract_width(alignment_fmt_.width);
    }
    preview.add_size(count);
}

inline std::uint16_t hex_digits_count(std::uint16_t x)
{
    return x > 0xFFF ? 4 : ( x > 0xFFu ? 3 : ( x > 0xF ? 2 : 1 ) );
}

template <typename CharT>
std::uint16_t ipv6_printer<CharT>::count_ipv6_characters() const
{
    if (style_ == ipv6style::big) {
        return 39;
    }
    std::uint16_t count = abbrev_.visible_colons_count();
    for (int i = 0; i < 8; ++i) {
        if (abbrev_.hextet_visible(i)) {
            count += hex_digits_count(addr_.hextets[i]);
        }
    }
    return count;
}

template <typename CharT>
void ipv6_printer<CharT>::print_to(strf::print_dest<CharT>& dest) const
{
    if (fillcount_ == 0) {
        print_ipv6(dest);
    } else switch(alignment_fmt_.alignment) {
        case strf::text_alignment::left:
            print_ipv6(dest);
            encode_fill_fn_(dest, fillcount_, alignment_fmt_.fill);
            break;
        case strf::text_alignment::right:
            encode_fill_fn_(dest, fillcount_, alignment_fmt_.fill);
            print_ipv6(dest);
            break;
        default:{
            assert(alignment_fmt_.alignment == strf::text_alignment::center);
            std::uint16_t halfcount = fillcount_ / 2;
            encode_fill_fn_(dest, halfcount, alignment_fmt_.fill);
            print_ipv6(dest);
            encode_fill_fn_(dest, fillcount_ - halfcount, alignment_fmt_.fill);
        }
    }
}

template <typename CharT>
void ipv6_printer<CharT>::print_ipv6(strf::print_dest<CharT>& dest) const
{
    const unsigned precision = (style_ == ipv6style::big ? 4 : 0);
    for (int i = 0; i < 8; ++i) {
        if (abbrev_.hextet_visible(i)) {
            strf::to(dest).with(lettercase_) (strf::hex(addr_.hextets[i]).p(precision));
        }
        if (abbrev_.colon_visible(i)) {
            strf::put(dest, (CharT)':');
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Specialize printing_traits template
// -----------------------------------------------------------------------------

template <>
struct printing_traits<xxx::ipv6address> {
    using override_tag = xxx::ipv6address;
    using forwarded_type = xxx::ipv6address;
    using formatters = strf::tag<ipv6_formatter, strf::alignment_formatter>;

    template <typename CharT, typename Preview, typename FPack, typename... T>
    static auto make_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , strf::value_with_formatters<T...> arg )
        -> strf::usual_arg_printer_input
            < CharT
            , Preview
            , FPack
            , strf::value_with_formatters<T...>
            , ipv6_printer<CharT> >
    {
        return {preview, fp, arg};
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

    TEST("aa:bb:1:2::").with(strf::lowercase)   (addr_aabb12);
    TEST("AA:BB:1:2::").with(strf::uppercase)   (addr_aabb12);
    TEST("AA:BB:1:2::").with(strf::mixedcase)   (addr_aabb12);

    TEST("....aa:bb:1:2::").with(strf::lowercase)   (strf::right(addr_aabb12, 15, '.'));
    TEST("....AA:BB:1:2::").with(strf::uppercase)   (strf::right(addr_aabb12, 15, '.'));
    TEST("....AA:BB:1:2::").with(strf::mixedcase)   (strf::right(addr_aabb12, 15, '.'));
}


namespace test_utils {

static strf::print_dest<char>*& test_messages_destination_ptr()
{
    static strf::print_dest<char>* ptr = nullptr;
    return ptr;
}

void set_test_messages_destination(strf::print_dest<char>& dest)
{
    test_messages_destination_ptr() = &dest;
}

strf::print_dest<char>& test_messages_destination()
{
    auto * ptr = test_messages_destination_ptr();
    return *ptr;
}

} // namespace test_utils


int main()
{
    strf::narrow_cfile_writer<char, 512> msg_dest(stdout);
    test_utils::set_test_messages_destination(msg_dest);

    tests();

    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(msg_dest, "All test passed!\n");
    } else {
        strf::to(msg_dest) (err_count, " tests failed!\n");
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

    unsigned middle_zeros_start = 0;
    unsigned middle_zeros_count = 0;
    unsigned trailing_zeros_start = 0;

    if (addr.hextets[0] == 0 && addr.hextets[1] == 0) {
        unsigned zi = 2;
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
                if (++zi == 6)
                    goto hide_leading_zeros;
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
            visible_hextets_count_ = static_cast<std::uint8_t>(8 - leading_zeros_count);
            visible_colons_count_ = visible_hextets_count_ + 1;
            hextets_visibility_bits_ = 0xFFu << leading_zeros_count;
            colons_visibility_bits_ = 0x7Fu & (0x7Fu << (leading_zeros_count - 2));
        }
    } else {
        unsigned zi = 2;
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
    visible_hextets_count_ = static_cast<std::uint8_t>(8 - middle_zeros_count);
    visible_colons_count_ = visible_hextets_count_;
    hextets_visibility_bits_ = ~(~(0xFFu << middle_zeros_count) << middle_zeros_start);
    colons_visibility_bits_  = 0x7Fu & ~(~(0xFFu << (middle_zeros_count - 1)) << middle_zeros_start);
    return;

    hide_trailing_zeros:
    auto trailing_zeros_count = static_cast<std::uint8_t>(8 - trailing_zeros_start);
    visible_hextets_count_ = static_cast<std::uint8_t>(trailing_zeros_start);
    visible_colons_count_  = static_cast<std::uint8_t>(trailing_zeros_start + 1);
    hextets_visibility_bits_ = 0xFFu >>  trailing_zeros_count;
    colons_visibility_bits_  = 0xFFu >> (trailing_zeros_count - 1);
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
constexpr  bool ipv6address_abbreviation::hextet_visible(unsigned index) const noexcept
{
    return index < 8 && (hextets_visibility_bits_ & (1 << index));
}
constexpr  bool ipv6address_abbreviation::colon_visible(unsigned index) const noexcept
{
    return index < 7 && (colons_visibility_bits_ & (1 << index));
}
constexpr  std::uint8_t ipv6address_abbreviation::visible_hextets_count() const noexcept
{
    return visible_hextets_count_;
}
constexpr  std::uint8_t ipv6address_abbreviation::visible_colons_count() const noexcept
{
    return visible_colons_count_;
}

} // namespace xxx
