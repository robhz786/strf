//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <array>
#include <vector>

auto write_out = strf::to(stdout);

namespace xxx {

struct base64_facet_c;

struct base64_facet
{
    using category = base64_facet_c;
    static constexpr bool store_by_value = true;

    unsigned line_length = 64;
    char eol[2] = {'\r', '\n'};
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
    std::size_t num_bytes = 0;
};

struct base64_format
{
    template <typename T>
    class fn
    {
    public:

        fn() = default;

        template <typename U>
        fn(const fn<U>& other) : indentation_(other.indentation())
        {
        }

        T&& indentation(unsigned _) &&
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


using base64_input_with_format = strf::value_with_format< base64_input
                                                        , base64_format >;

inline auto base64(const void* bytes, std::size_t num_bytes)
{
    base64_input data{reinterpret_cast<const unsigned char*>(bytes), num_bytes};
    return base64_input_with_format{data};
}

} // namespace xxx

// To enable `base64_input` to work in `fmt_range`:
namespace strf {

template <> struct fmt_traits<xxx::base64_input>
{
    using fmt_type = xxx::base64_input_with_format;
};

} // namespace strf

namespace xxx {

template <std::size_t CharSize>
class base64_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename ... T>
    base64_printer
        ( const strf::usual_printer_input<T...>& input)
        : base64_printer
            ( strf::get_facet<base64_facet_c, base64_facet_c>(input.fp)
            , input.preview
            , input.arg )
    {
    }

    template <strf::preview_size PreviewSize>
    base64_printer
        ( base64_facet facet
        , strf::print_preview<PreviewSize, strf::preview_width::no>& preview
        , const base64_input_with_format& fmt );

    void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    void calc_size_(strf::size_preview<false>&) const
    {
    }

    void calc_size_(strf::size_preview<true>&) const;

    void write_single_line_(strf::underlying_outbuf<CharSize>& ob) const;

    void encode_all_data_in_this_line_(strf::underlying_outbuf<CharSize>& ob) const;

    void write_multiline_(strf::underlying_outbuf<CharSize>& ob) const;

    void write_identation_(strf::underlying_outbuf<CharSize>& ob) const;

    void write_end_of_line_(strf::underlying_outbuf<CharSize>& ob) const;

    void encode_3bytes_
        ( char_type* dest
        , const std::uint8_t* data
        , std::size_t data_size ) const;

    char_type encode_(std::uint8_t hextet) const;

    const base64_facet facet_;
    const base64_input_with_format fmt_;
};

template <std::size_t CharSize>
template <strf::preview_size PreviewSize>
base64_printer<CharSize>::base64_printer
    ( base64_facet facet
    , strf::print_preview<PreviewSize, strf::preview_width::no>& preview
    , const base64_input_with_format& fmt )
    : facet_(facet)
    , fmt_(fmt)
{
    calc_size_(preview);
}

template <std::size_t CharSize>
void base64_printer<CharSize>::calc_size_(strf::size_preview<true>& preview) const
{
    std::size_t num_digits = 4 * (fmt_.value().num_bytes + 2) / 3;
    preview.add_size(num_digits);
    if (facet_.line_length > 0 && facet_.eol[0] != '\0') {
        std::size_t num_lines
            = (num_digits + facet_.line_length - 1)
            / facet_.line_length;
        std::size_t eol_size = 1 + (facet_.eol[1] != '\0');
        preview.add_size(num_lines * (fmt_.indentation() + eol_size));
    }
}

//[ base64_printer__write

template <std::size_t CharSize>
void base64_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    if (facet_.single_line()) {
        write_single_line_(ob);
    } else {
        write_multiline_(ob);
    }
}

template <std::size_t CharSize>
void base64_printer<CharSize>::write_single_line_(strf::underlying_outbuf<CharSize>& ob) const
{
    write_identation_(ob);
    encode_all_data_in_this_line_(ob);
}

template <std::size_t CharSize>
void base64_printer<CharSize>::write_identation_(strf::underlying_outbuf<CharSize>& ob) const
{
    using traits = std::char_traits<char_type>;
    std::size_t count = fmt_.indentation();
    while(true) {
        std::size_t buff_size = ob.size();
        if (buff_size >= count) {
            traits::assign(ob.pointer(), count, char_type(' '));
            ob.advance(count);
            return;
        }
        traits::assign(ob.pointer(), buff_size, char_type(' '));
        count -= buff_size;
        ob.advance_to(ob.end());
        ob.recycle();
    };
}

template <std::size_t CharSize>
void base64_printer<CharSize>::encode_all_data_in_this_line_(strf::underlying_outbuf<CharSize>& ob) const
{
    auto data_it = static_cast<const std::uint8_t*>(fmt_.value().bytes);
    for (std::ptrdiff_t count = fmt_.value().num_bytes; count > 0; count -= 3) {
        ob.ensure(4);
        encode_3bytes_(ob.pointer(), data_it, count);
        ob.advance(4);
        data_it += 3;
    }
}

template <std::size_t CharSize>
void base64_printer<CharSize>::encode_3bytes_
    ( char_type* dest
    , const std::uint8_t* data
    , std::size_t data_size ) const
{
    dest[0] = encode_(data[0] >> 2);
    dest[1] = encode_(((data[0] & 0x03) << 4) |
                      (data_size < 2 ? 0 : ((data[1] & 0xF0) >> 4)));
    dest[2] = (data_size < 2)
        ? '='
        : encode_(((data[1] & 0x0F) << 2) |
                 (data_size < 3 ? 0 : ((data[2] & 0xC0) >> 6)));
    dest[3] = data_size < 3 ? '=' : encode_(data[2] & 0x3F);
}

template <std::size_t CharSize>
auto base64_printer<CharSize>::encode_(std::uint8_t hextet) const -> char_type
{
    assert(hextet <= 63);
    std::uint8_t ch =
        hextet < 26 ?  static_cast<std::uint8_t>('A') + hextet :
        hextet < 52 ?  static_cast<std::uint8_t>('a') + hextet - 26 :
        hextet < 62 ?  static_cast<std::uint8_t>('0') + hextet - 52 :
        hextet == 62 ? static_cast<std::uint8_t>(facet_.char62) :
      /*hextet == 63*/ static_cast<std::uint8_t>(facet_.char63) ;

    return ch;
}
//]

template <std::size_t CharSize>
void base64_printer<CharSize>::write_multiline_(strf::underlying_outbuf<CharSize>& ob) const
{
    write_identation_(ob);

    auto data_it = static_cast<const std::uint8_t*>(fmt_.value().bytes);
    std::ptrdiff_t remaining_bytes = fmt_.value().num_bytes;
    unsigned cursor_pos = 0;

    while (remaining_bytes > 0) {
        if (cursor_pos + 4 < facet_.line_length) {
            ob.ensure(4);
            encode_3bytes_(ob.pointer(), data_it, remaining_bytes);
            ob.advance(4);
            cursor_pos += 4;
        } else {
            char_type tmp[4];
            encode_3bytes_(tmp, data_it, remaining_bytes);
            for(int i=0; i < 4; ++i) {
                if (cursor_pos == facet_.line_length) {
                    cursor_pos = 0;
                    write_end_of_line_(ob);
                    write_identation_(ob);
                }
                ob.ensure(1);
                * ob.pointer() = tmp[i];
                ob.advance(1);
                ++cursor_pos;
            }
        }
        data_it += 3;
        remaining_bytes -= 3;
    }
    if (cursor_pos != 0) {
        write_end_of_line_(ob);
    }
}

template <std::size_t CharSize>
void base64_printer<CharSize>::write_end_of_line_(strf::underlying_outbuf<CharSize>& ob) const
{
    ob.ensure(2);
    ob.pointer()[0] = facet_.eol[0];
    ob.pointer()[1] = facet_.eol[1];
    ob.advance(facet_.eol[1] == '\0' ? 1 : 2);
}

} //namespace xxx

namespace strf {

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, xxx::base64_input>
    : strf::usual_printable_traits<CharT, FPack, xxx::base64_printer<sizeof(CharT)>>
{
    static_assert(!Preview::require_width, "");
};

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, xxx::base64_input_with_format>
    : strf::usual_printable_traits<CharT, FPack, xxx::base64_printer<sizeof(CharT)>>
{
    static_assert(!Preview::width_required, "");
};

}


void tests()
{
    const char* data = "The quick brown fox jumps over the lazy dog.";
    auto data_size = strlen(data);

    {
        auto result = strf::to_string(xxx::base64(data, data_size)) ;
        assert(result == "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZy4=\r\n");
    }

    {
        // customizing line length, end of line and identation
        auto result = strf::to_string
            .with(xxx::base64_facet{50, {'\n', '\0'}})
            (xxx::base64(data, data_size).indentation(4));

        auto expected =
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

        auto expected =
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
        std::vector<xxx::base64_input> vec =
            { {msg0, strlen(msg0)}
            , {msg1, strlen(msg1)}
            , {msg2, strlen(msg2)} };

        auto result = strf::to_string
            .with(xxx::base64_facet{50, {'\n', '\0'}})
            (strf::fmt_separated_range(vec, "------------\n").indentation(4));

        auto expected =
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
//[base64_sample
    const char* msg  = "The quick brown fox jumps over the lazy dog.";

    auto obtained = strf::to_string
        .with(xxx::base64_facet{50, {'\n', '\0'}})
        ( xxx::base64(msg, strlen(msg)).indentation(4) );

    auto expected =
        "    VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYX\n"
        "    p5IGRvZy4=\n";

    assert(obtained == expected);
//]
    (void)obtained;
    (void)expected;
};

int main()
{
    tests();
    sample();
    return 0;
}
