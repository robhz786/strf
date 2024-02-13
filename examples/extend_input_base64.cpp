//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>
#include <array>
#include <vector>

namespace xxx {

struct base64_facet_c;

struct base64_facet
{
    using category = base64_facet_c;

    unsigned line_length = 64;
    char eol[2] = { '\r', '\n' };
    char char62 = '+';
    char char63 = '/';

    bool single_line() const
    {
        return line_length == 0 || eol[0] == '\0';
    }
};

struct base64_facet_c
{
    static constexpr bool constrainable = true;

    constexpr static base64_facet get_default()
    {
        return {};
    }
};

struct base64_input
{
    const void* bytes = nullptr;
    std::ptrdiff_t num_bytes = 0;
};

struct base64_format_specifier
{
    template <typename T>
    class fn
    {
    public:

        fn() = default;

        template <typename U>
        explicit fn(const fn<U>& other) : indentation_(other.indentation())
        {
        }

        T&& indentation(unsigned _)&&
        {
            indentation_ = _;
            return static_cast<T&&>(*this);
        }

        unsigned indentation() const
        {
            return indentation_;
        }

    private:

        unsigned indentation_ = 0;
    };
};

template <typename CharT>
class base64_printer
{
public:

    base64_printer
        ( base64_facet facet
        , base64_input input
        , unsigned indentation_size )
        : facet_(facet)
        , input_(input)
        , indentation_size_(indentation_size)
    {
    }

    std::ptrdiff_t calculate_size() const;

    void operator()(strf::destination<CharT>& dst) const;

private:


    void write_single_line_(strf::destination<CharT>& dst) const;

    void encode_all_data_in_this_line_(strf::destination<CharT>& dst) const;

    void write_multiline_(strf::destination<CharT>& dst) const;

    void write_indentation_(strf::destination<CharT>& dst) const;

    void write_end_of_line_(strf::destination<CharT>& dst) const;

    void encode_3bytes_
        ( CharT* dst
        , const std::uint8_t* data
        , std::ptrdiff_t data_size ) const;

    CharT encode_(unsigned hextet) const;

    base64_facet facet_ {};
    base64_input input_ {} ;
    unsigned indentation_size_ = 0;
};

template <typename CharT>
std::ptrdiff_t base64_printer<CharT>::calculate_size() const
{
    std::ptrdiff_t size = 0;

    auto num_digits = 4 * (input_.num_bytes + 2) / 3;
    size += num_digits;
    if (facet_.line_length > 0 && facet_.eol[0] != '\0') {
        auto num_lines
            = (num_digits + facet_.line_length - 1)
            / facet_.line_length;
        auto eol_size = 1 + (facet_.eol[1] != '\0');
        size += num_lines * (indentation_size_ + eol_size);
    }
    return size;
}

template <typename CharT>
void base64_printer<CharT>::operator()(strf::destination<CharT>& dst) const
{
    if (facet_.single_line()) {
        write_single_line_(dst);
    } else {
        write_multiline_(dst);
    }
}

template <typename CharT>
void base64_printer<CharT>::write_single_line_(strf::destination<CharT>& dst) const
{
    write_indentation_(dst);
    encode_all_data_in_this_line_(dst);
}

template <typename CharT>
void base64_printer<CharT>::write_indentation_(strf::destination<CharT>& dst) const
{
    strf::to(dst) (strf::multi((CharT)' ', indentation_size_));
}

template <typename CharT>
void base64_printer<CharT>::encode_all_data_in_this_line_(strf::destination<CharT>& dst) const
{
    const auto* data_it = static_cast<const std::uint8_t*>(input_.bytes);
    for (auto count = input_.num_bytes; count > 0; count -= 3) {
        dst.ensure(4);
        encode_3bytes_(dst.buffer_ptr(), data_it, count);
        dst.advance(4);
        data_it += 3;
    }
}

template <typename CharT>
void base64_printer<CharT>::encode_3bytes_
    ( CharT* dst
    , const std::uint8_t* data
    , std::ptrdiff_t data_size ) const
{
    dst[0] = encode_(data[0] >> 2U);
    dst[1] = encode_(((data[0] & 0x03U) << 4U) |
                      (data_size < 2 ? 0U : ((data[1] & 0xF0U) >> 4U)));
    dst[2] = (data_size < 2)
        ? '='
        : encode_(((data[1] & 0x0FU) << 2U) |
                 (data_size < 3 ? 0U : ((data[2] & 0xC0U) >> 6U)));
    dst[3] = data_size < 3 ? '=' : encode_(data[2] & 0x3FU);
}

template <typename CharT>
auto base64_printer<CharT>::encode_(unsigned hextet) const -> CharT
{
    assert(hextet <= 63);
    const int ch =
        ( hextet < 26 ?  ('A' + (int)hextet)
        : hextet < 52 ?  ('a' + (int)hextet - 26)
        : hextet < 62 ?  ('0' + (int)hextet - 52)
        : hextet == 62 ? facet_.char62
        : /*hextet == 63*/ facet_.char63 );

    return static_cast<CharT>(ch);
}

template <typename CharT>
void base64_printer<CharT>::write_multiline_(strf::destination<CharT>& dst) const
{
    write_indentation_(dst);

    const auto *data_it = static_cast<const std::uint8_t*>(input_.bytes);
    auto remaining_bytes = input_.num_bytes;
    unsigned cursor_pos = 0;

    while (remaining_bytes > 0) {
        if (cursor_pos + 4 < facet_.line_length) {
            dst.ensure(4);
            encode_3bytes_(dst.buffer_ptr(), data_it, remaining_bytes);
            dst.advance(4);
            cursor_pos += 4;
        } else {
            CharT tmp[4];
            encode_3bytes_(tmp, data_it, remaining_bytes);
            for(int i=0; i < 4; ++i) {
                if (cursor_pos == facet_.line_length) {
                    cursor_pos = 0;
                    write_end_of_line_(dst);
                    write_indentation_(dst);
                }
                dst.ensure(1);
                * dst.buffer_ptr() = tmp[i];
                dst.advance(1);
                ++cursor_pos;
            }
        }
        data_it += 3;
        remaining_bytes -= 3;
    }
    if (cursor_pos != 0) {
        write_end_of_line_(dst);
    }
}

template <typename CharT>
void base64_printer<CharT>::write_end_of_line_(strf::destination<CharT>& dst) const
{
    dst.ensure(2);
    dst.buffer_ptr()[0] = facet_.eol[0];
    dst.buffer_ptr()[1] = facet_.eol[1];
    dst.advance(facet_.eol[1] == '\0' ? 1 : 2);
}

} // namespace xxx

namespace strf {

template <>
struct printable_def<xxx::base64_input>
{
    using representative_type = xxx::base64_input;
    using forwarded_type = xxx::base64_input;
    using format_specifiers = strf::tag<xxx::base64_format_specifier>;

    using input_with_fmt =
        strf::value_and_format<printable_def<xxx::base64_input>, xxx::base64_format_specifier>;
    
    template <typename CharT, typename FPack>
    static auto make_printer
        ( strf::tag<CharT>
        , strf::no_premeasurements*
        , const FPack& facets
        , const input_with_fmt& arg )
    {
        auto f = strf::use_facet<xxx::base64_facet_c, representative_type>(facets);
        return xxx::base64_printer<CharT>{f, arg.value(), arg.indentation()};
    }

    template <typename CharT, typename FPack>
    static auto make_printer
        ( strf::tag<CharT>
        , strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>* pre
        , const FPack& facets
        , const input_with_fmt& arg )
    {
        auto f = strf::use_facet<xxx::base64_facet_c, representative_type>(facets);
        xxx::base64_printer<CharT> printer{f, arg.value(), arg.indentation()};
        pre->add_size(printer.calculate_size());
        return printer;
    }
};

} // namespace strf

namespace xxx {

inline auto base64(const void* bytes, std::size_t num_bytes)
{
    const base64_input data{ static_cast<const unsigned char*>(bytes)
                           , static_cast<std::ptrdiff_t>(num_bytes) };
    
    return strf::fmt(data);
}

} // namespace xxx



void tests()
{
    const char* data = "The quick brown fox jumps over the lazy dog.";
    auto data_size = strlen(data);

    {
        auto result = strf::to_string(xxx::base64(data, data_size)) ;
        assert(result == "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZy4=\r\n");
    }

    {
        // customizing line length, end of line and indentation
        auto result = strf::to_string
            .with(xxx::base64_facet{50, {'\n', '\0'}})
            (xxx::base64(data, data_size).indentation(4));

        const auto *expected =
            "    VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYX\n"
            "    p5IGRvZy4=\n";

        assert(result == expected);
        (void)result;
        (void)expected;
    }
    {
        // When the length of last line is exactly as base64_facet::line_length,
        auto result = strf::to_string
            .with(xxx::base64_facet{30})
            (xxx::base64(data, data_size).indentation(4));

        const auto *expected =
            "    VGhlIHF1aWNrIGJyb3duIGZveCBqdW\r\n"
            "    1wcyBvdmVyIHRoZSBsYXp5IGRvZy4=\r\n";

        assert(result == expected);
        (void)result;
        (void)expected;
    }
    {
        // When base64_facet::line_length == 1
        auto result = strf::to_string
            .with(xxx::base64_facet{1, {'\n', '\0'}})
            (xxx::base64("  >  ?", 6).indentation(2));

        assert(result == "  I\n  C\n  A\n  +\n  I\n  C\n  A\n  /\n");
    }
    {
        // When base64_facet::line_length == 3
        auto result = strf::to_string
            .with(xxx::base64_facet{3, {'\n', '\0'}})
            (xxx::base64("  >  ?", 6).indentation(2));

        assert(result == "  ICA\n  +IC\n  A/\n");
    }
    {
        // When base64_facet::line_length == 4
        auto result = strf::to_string
            .with(xxx::base64_facet{4, {'\n', '\0'}})
            (xxx::base64("  >  ?", 6).indentation(2));

        assert(result == "  ICA+\n  ICA/\n");
    }
    {
        // The default character for index 62 is '+'
        // and for index 63 is '/'
        auto result = strf::to_string(xxx::base64("  >  ?", 6));
        assert(result == "ICA+ICA/\r\n");
    }

    {
        // customizing characters for index 62 and 63
        auto result = strf::to_string
            .with(xxx::base64_facet{50, {'\r', '\n'}, '-', '_'})
            (xxx::base64("  >  ?", 6));

        assert(result == "ICA-ICA_\r\n");
    }

    {
        // when base64_facet::line_length == 0'
        // then the result has no end of line
        auto result = strf::to_string
            .with(xxx::base64_facet{0, {'\r', '\n'}})
            (xxx::base64("  >  ?", 6));

        assert(result == "ICA+ICA/");
    }
    {
        // when base64_facet::eol[0] == '\0'
        // then the result has no end of line
        auto result = strf::to_string
            .with(xxx::base64_facet{50, {'\0', '\n'}, '-', '_'})
            (xxx::base64("  >  ?", 6));

        assert(result == "ICA-ICA_");
    }
    {
        // test indentation on single line mode
        auto result = strf::to_string
            .with(xxx::base64_facet{0})
            (xxx::base64("  >  ?", 6).indentation(4));

        assert(result == "    ICA+ICA/");
    }
    {
        //test in ranges

        const char* msg0 = "abc";
        const char* msg1 = "abcd";
        const char* msg2 = "abcde";
        const std::vector<xxx::base64_input> vec =
            { {msg0, (std::ptrdiff_t)strlen(msg0)}
            , {msg1, (std::ptrdiff_t)strlen(msg1)}
            , {msg2, (std::ptrdiff_t)strlen(msg2)} };

        auto result = strf::to_string
            .with(xxx::base64_facet{50, {'\n', '\0'}})
            (strf::fmt_separated_range(vec, "------------\n").indentation(4));

        const auto *expected =
            "    YWJj\n"
            "------------\n"
            "    YWJjZA==\n"
            "------------\n"
            "    YWJjZGU=\n";

        assert(result == expected);
        (void)result;
        (void)expected;
    }

}


void sample()
{
    const char* msg  = "The quick brown fox jumps over the lazy dog.";

    auto obtained = strf::to_string
        .with(xxx::base64_facet{50, {'\n', '\0'}})
        ( xxx::base64(msg, strlen(msg)).indentation(4) );

    const auto *expected =
        "    VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYX\n"
        "    p5IGRvZy4=\n";

    assert(obtained == expected);
    (void)obtained;
    (void)expected;
}

int main()
{
    tests();
    sample();
    return 0;
}
