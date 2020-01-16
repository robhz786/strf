//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <array>
#include <vector>

auto write_out = strf::to(stdout);

//[ base64_facet

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

    static base64_facet get_default()
    {
        return {};
    }
};

} // namespace xxx
//]

//[ fmt_base64_input

namespace xxx {
struct base64_input
{
    const void* bytes = nullptr;
    std::size_t num_bytes = 0;
};
} // namespace xxx

//]

//[ base64_format

namespace xxx {

struct base64_format
{
    template <typename T>
    class fn
    {
    public:

        fn() = default;

        template <typename U>
        fn(const fn<U>& other) : _indentation(other.indentation())
        {
        }

        T&& indentation(unsigned _) &&
        {
            _indentation = _;
            return static_cast<T&&>(*this);
        }

        unsigned indentation() const
        {
            return _indentation;
        }

    private:

        unsigned _indentation = 0;
    };
};

} //namespace xxx
//]

//[ base64_input_with_format

namespace xxx {

using base64_input_with_format = strf::value_with_format< base64_input
                                                        , base64_format >;

inline auto base64(const void* bytes, std::size_t num_bytes)
{
    base64_input data{reinterpret_cast<const unsigned char*>(bytes), num_bytes};
    return base64_input_with_format{data};
}

/*<< Although `strf::fmt` is not needed to work with `base64_input` since the
`base64` function already instantiates `base64_input_with_format`, we still
 need to overload `make_fmt` if we want `base64_input` to work in
 [link ranges fmt_range]
 >>*/inline auto make_fmt(strf::rank<1>, const base64_input& d)
{
    return base64_input_with_format{d};
}

}  // namespace xxx
//]

namespace xxx {

template <typename CharT>
class base64_printer: public strf::printer<CharT>
{
public:

    template <bool PreviewSize>
    base64_printer
        ( base64_facet facet
        , strf::print_preview<PreviewSize, false>& preview
        , const base64_input_with_format& fmt );

    void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    void _calc_size(strf::size_preview<false>&) const
    {
    }

    void _calc_size(strf::size_preview<true>&) const;

    void _write_single_line(strf::basic_outbuf<CharT>& ob) const;

    void _encode_all_data_in_this_line(strf::basic_outbuf<CharT>& ob) const;

    void _write_multiline(strf::basic_outbuf<CharT>& ob) const;

    void _write_identation(strf::basic_outbuf<CharT>& ob) const;

    void _write_end_of_line(strf::basic_outbuf<CharT>& ob) const;

    void _encode_3bytes
        ( CharT* dest
        , const std::uint8_t* data
        , std::size_t data_size ) const;

    CharT _encode(std::uint8_t hextet) const;

    const base64_facet _facet;
    const base64_input_with_format _fmt;
};

template <typename CharT>
template <bool PreviewSize>
base64_printer<CharT>::base64_printer
    ( base64_facet facet
    , strf::print_preview<PreviewSize, false>& preview
    , const base64_input_with_format& fmt )
    : _facet(facet)
    , _fmt(fmt)
{
    _calc_size(preview);
}

template <typename CharT>
void base64_printer<CharT>::_calc_size(strf::size_preview<true>& preview) const
{
    std::size_t num_digits = 4 * (_fmt.value().num_bytes + 2) / 3;
    preview.add_size(num_digits);
    if (_facet.line_length > 0 && _facet.eol[0] != '\0') {
        std::size_t num_lines
            = (num_digits + _facet.line_length - 1)
            / _facet.line_length;
        std::size_t eol_size = 1 + (_facet.eol[1] != '\0');
        preview.add_size(num_lines * (_fmt.indentation() + eol_size));
    }
}

//[ base64_printer__write

template <typename CharT>
void base64_printer<CharT>::print_to(strf::basic_outbuf<CharT>& ob) const
{
    if (_facet.single_line()) {
        _write_single_line(ob);
    } else {
        _write_multiline(ob);
    }
}

template <typename CharT>
void base64_printer<CharT>::_write_single_line(strf::basic_outbuf<CharT>& ob) const
{
    _write_identation(ob);
    _encode_all_data_in_this_line(ob);
}

template <typename CharT>
void base64_printer<CharT>::_write_identation(strf::basic_outbuf<CharT>& ob) const
{
    using traits = std::char_traits<CharT>;
    std::size_t count = _fmt.indentation();
    while(true) {
        std::size_t buff_size = ob.size();
        if (buff_size >= count) {
            traits::assign(ob.pos(), count, CharT(' '));
            ob.advance(count);
            return;
        }
        traits::assign(ob.pos(), buff_size, CharT(' '));
        count -= buff_size;
        ob.advance_to(ob.end());
        ob.recycle();
    };
}

template <typename CharT>
void base64_printer<CharT>::_encode_all_data_in_this_line(strf::basic_outbuf<CharT>& ob) const
{
    auto data_it = static_cast<const std::uint8_t*>(_fmt.value().bytes);
    for (std::ptrdiff_t count = _fmt.value().num_bytes; count > 0; count -= 3) {
        ob.ensure(4);
        _encode_3bytes(ob.pos(), data_it, count);
        ob.advance(4);
        data_it += 3;
    }
}

template <typename CharT>
void base64_printer<CharT>::_encode_3bytes
    ( CharT* dest
    , const std::uint8_t* data
    , std::size_t data_size ) const
{
    dest[0] = _encode(data[0] >> 2);
    dest[1] = _encode(((data[0] & 0x03) << 4) |
                      (data_size < 2 ? 0 : ((data[1] & 0xF0) >> 4)));
    dest[2] = (data_size < 2)
        ? '='
        : _encode(((data[1] & 0x0F) << 2) |
                 (data_size < 3 ? 0 : ((data[2] & 0xC0) >> 6)));
    dest[3] = data_size < 3 ? '=' : _encode(data[2] & 0x3F);
}

template <typename CharT>
CharT base64_printer<CharT>::_encode(std::uint8_t hextet) const
{
    assert(hextet <= 63);
    std::uint8_t ch =
        hextet < 26 ?  static_cast<std::uint8_t>('A') + hextet :
        hextet < 52 ?  static_cast<std::uint8_t>('a') + hextet - 26 :
        hextet < 62 ?  static_cast<std::uint8_t>('0') + hextet - 52 :
        hextet == 62 ? static_cast<std::uint8_t>(_facet.char62) :
      /*hextet == 63*/ static_cast<std::uint8_t>(_facet.char63) ;

    return ch;
}
//]

template <typename CharT>
void base64_printer<CharT>::_write_multiline(strf::basic_outbuf<CharT>& ob) const
{
    _write_identation(ob);

    auto data_it = static_cast<const std::uint8_t*>(_fmt.value().bytes);
    std::ptrdiff_t remaining_bytes = _fmt.value().num_bytes;
    unsigned cursor_pos = 0;

    while (remaining_bytes > 0) {
        if (cursor_pos + 4 < _facet.line_length) {
            ob.ensure(4);
            _encode_3bytes(ob.pos(), data_it, remaining_bytes);
            ob.advance(4);
            cursor_pos += 4;
        } else {
            CharT tmp[4];
            _encode_3bytes(tmp, data_it, remaining_bytes);
            for(int i=0; i < 4; ++i) {
                if (cursor_pos == _facet.line_length) {
                    cursor_pos = 0;
                    _write_end_of_line(ob);
                    _write_identation(ob);
                }
                ob.ensure(1);
                * ob.pos() = tmp[i];
                ob.advance(1);
                ++cursor_pos;
            }
        }
        data_it += 3;
        remaining_bytes -= 3;
    }
    if (cursor_pos != 0) {
        _write_end_of_line(ob);
    }
}

template <typename CharT>
void base64_printer<CharT>::_write_end_of_line(strf::basic_outbuf<CharT>& ob) const
{
    ob.ensure(2);
    ob.pos()[0] = _facet.eol[0];
    ob.pos()[1] = _facet.eol[1];
    ob.advance(_facet.eol[1] == '\0' ? 1 : 2);
}


} //namespace xxx

//[ make_printer_base64

namespace xxx {

template <typename CharT, typename FPack, typename Preview>
inline base64_printer<CharT> make_printer( strf::rank<1>
                                         , const FPack& fp
                                         , Preview& preview
                                         , const base64_input_with_format& fmt )
{
  /*<< see [link facets_pack get_facet.]
>>*/auto facet = strf::get_facet<base64_facet_c, base64_input>(fp);
    return {facet, preview, fmt};
}


template <typename CharT, typename FPack, typename Preview>
inline base64_printer<CharT> make_printer( strf::rank<1> r
                                         , const FPack& fp
                                         , Preview& preview
                                         , const base64_input& input )
{
    return make_printer(r, fp, preview, base64_input_with_format{input});
}

} // namespace xxx

//]


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
