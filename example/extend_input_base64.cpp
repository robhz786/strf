//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include <array>
#include <vector>

namespace strf = boost::stringify::v0;

auto write_out = strf::write(stdout);

//[ base64_facet

namespace xxx {

struct base64_facet_category;

struct base64_facet
{
    using category = base64_facet_category;

    unsigned line_length = 64;
    char eol[2] = {'\r', '\n'};
    char char62 = '+';
    char char63 = '/';

    bool single_line() const
    {
        return line_length == 0 || eol[0] == '\0';
    }
};

struct base64_facet_category
{
    static const base64_facet& get_default()
    {
        static const base64_facet obj{};
        return obj;
    }
};

//]

} // namespace xxx


//[ fmt_base64_input

namespace xxx {

struct base64_input
{
    const void* bytes = nullptr;
    std::size_t num_bytes = 0;
};

}

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
        fn(const fn<U>& other) : m_indentation(other.indentation())
        {
        }

        T&& indentation(unsigned _) &&
        {
            m_indentation = _;
            return static_cast<T&&>(*this);
        }

        unsigned indentation() const
        {
            return m_indentation;
        }

    private:

        unsigned m_indentation = 0;
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
 >>*/inline auto make_fmt(strf::tag, const base64_input& d)
{
    return base64_input_with_format{d};
} // namespace xxx

}
//]



namespace xxx {

class base64_common_impl
{
public:

    base64_common_impl(base64_facet f) : m_facet(f) {};

    std::size_t necessary_size(std::size_t num_bytes, unsigned indentation) const;

    std::array<char, 4> encode(const unsigned char* octets, std::size_t num) const;

    char encode(unsigned hextet) const;

    base64_facet facet() const
    {
        return m_facet;
    }

private:

    const base64_facet m_facet;
};

std::size_t base64_common_impl::necessary_size(std::size_t num_bytes, unsigned indentation) const
{
    std::size_t num_digits = 4 * (num_bytes + 2) / 3;
    if (m_facet.line_length > 0 && m_facet.eol[0] != '\0')
    {
        std::size_t num_lines
            = (num_digits + m_facet.line_length - 1)
            / m_facet.line_length;
        std::size_t eol_size = 1 + (m_facet.eol[1] != '\0');
        return num_digits + num_lines * (indentation + eol_size);
    }
    return num_digits;
}

std::array<char, 4> base64_common_impl::encode(const unsigned char* octets, std::size_t num) const
{
    char ch0 = encode(octets[0] >> 2);
    char ch1 = encode(((octets[0] & 0x03) << 4) |
                     (num < 2 ? 0 : ((octets[1] & 0xF0) >> 4)));
    char ch2 = (num < 2)
        ? '='
        : encode(((octets[1] & 0x0F) << 2) |
                 (num < 3 ? 0 : ((octets[2] & 0xC0) >> 6)));
    char ch3 = num < 3 ? '=' : encode(octets[2] & 0x3F);

    return {{ch0, ch1, ch2, ch3}};
}

char base64_common_impl::encode(unsigned hextet) const
{
    BOOST_ASSERT(hextet <= 63);
    unsigned ch =
        hextet < 26 ?  static_cast<unsigned>('A') + hextet :
        hextet < 52 ?  static_cast<unsigned>('a') + hextet - 26 :
        hextet < 62 ?  static_cast<unsigned>('0') + hextet - 52 :
        hextet == 62 ? static_cast<unsigned>(m_facet.char62) :
      /*hextet == 63*/ static_cast<unsigned>(m_facet.char63) ;

    return static_cast<char>(ch);
}



//[single_line_base64_pm_input
template <typename CharT>
class single_line_base64_pm_input
    : public strf::piecemeal_input<CharT>
    , private base64_common_impl
{
public:

    single_line_base64_pm_input
        ( base64_facet facet
        , const base64_input_with_format& fmt )
        : base64_common_impl(facet)
        , m_fmt(fmt)
    {
        BOOST_ASSERT(facet.single_line());
    }

    CharT* get_some(CharT* begin, CharT* end) override;

private:

    CharT* write_indentation(CharT* begin, CharT* end);
    CharT* write_block(CharT* it);
    const unsigned char* bytes() const
    {
        return reinterpret_cast<const unsigned char*>(m_fmt.value().bytes);
    }
    auto num_bytes() const
    {
        return m_fmt.value().num_bytes;
    }

    const base64_input_with_format m_fmt;
    std::size_t m_index = 0;
    unsigned m_column = 0;
};


template <typename CharT>
CharT* single_line_base64_pm_input<CharT>::get_some(CharT* begin, CharT* end)
{
    CharT* it =  begin;

    if (m_column < m_fmt.indentation())
    {
        it = write_indentation(it, end);
    }
    while(it < end - 4 && m_index < num_bytes())
    {
        it = write_block(it);
    }
    if (m_index >= num_bytes())
    {
        this->report_success();
    }
    return it;
}


template <typename CharT>
CharT* single_line_base64_pm_input<CharT>::write_block(CharT* it)
{
    std::array<char, 4> arr = encode( bytes() + m_index
                                    , num_bytes() - m_index );
    it[0] = arr[0];
    it[1] = arr[1];
    it[2] = arr[2];
    it[3] = arr[3];
    m_index += 3;
    return it + 4;
}


template <typename CharT>
CharT* single_line_base64_pm_input<CharT>::write_indentation(CharT* begin, CharT* end)
{
    BOOST_ASSERT(m_column < m_fmt.indentation());

    std::size_t availabe_space = end - begin;
    std::size_t remaining_indentation_size = m_fmt.indentation() - m_column;
    auto count = (std::min) (availabe_space, remaining_indentation_size);
    std::fill(begin, begin + count, static_cast<CharT>(' '));
    m_column += static_cast<unsigned>(count);
    return begin + count;
}
//]

template <typename CharT>
class multiline_base64_pm_input
    : public strf::piecemeal_input<CharT>
    , private base64_common_impl
{
public:

    multiline_base64_pm_input
        ( base64_facet facet
        , const base64_input_with_format& fmt );

    CharT* get_some(CharT* begin, CharT* end) override;

private:

    CharT* write_indentation(CharT* begin, CharT* end);
    CharT* write_whole_block_in_this_line(CharT* it);
    CharT* begin_partial_block(CharT* it);
    CharT* continue_partial_block(CharT* it);
    CharT* write_eol(CharT* it);
    const unsigned char* bytes() const
    {
        return reinterpret_cast<const unsigned char*>(m_fmt.value().bytes);
    }
    auto num_bytes() const
    {
        return m_fmt.value().num_bytes;
    }

    const base64_input_with_format m_fmt;
    std::size_t m_index = 0;
    unsigned m_column = 0;
    unsigned m_block_sub_index = 0;
    std::array<char, 4> m_split_block;
    const unsigned m_total_line_length;
};


template <typename CharT>
multiline_base64_pm_input<CharT>::multiline_base64_pm_input
    ( base64_facet facet
    , const base64_input_with_format& fmt )
    : base64_common_impl(facet)
    , m_fmt(fmt)
    , m_total_line_length(facet.line_length + fmt.indentation())
{
    BOOST_ASSERT( ! facet.single_line());
}

//[ multiline_base64_pm_input__get_some
template <typename CharT>
CharT* multiline_base64_pm_input<CharT>::get_some(CharT* begin, CharT* end)
{
    auto it = begin;
    while(m_index < num_bytes())
    {
        if(m_column < m_fmt.indentation())
        {
            it = write_indentation(it, end);
            if (it == end)
            {
                return it;
            }
        }
        else if(m_block_sub_index != 0)
        {
            if(it + 9 > end)
            {
                return it;
            }
            it = continue_partial_block(it);
        }
        else if(m_column + 4 >= m_total_line_length)
        {
            if(it + 12 > end)
            {
                return it;
            }
            it = begin_partial_block(it);
        }
        else
        {
            if(it + 4 > end)
            {
                return it;
            }
            it = write_whole_block_in_this_line(it);
        }
    }
    if(m_column != 0)
    {
        if(it + 2 > end)
        {
            return it;
        }
        it = write_eol(it);
        m_column = 0;
    }
    this->report_success();
    return it;
}
//]

template <typename CharT>
CharT* multiline_base64_pm_input<CharT>::write_indentation
    ( CharT* begin, CharT* end )
{
    BOOST_ASSERT(m_column < m_fmt.indentation());

    std::size_t availabe_space = end - begin;
    std::size_t remaining_indentation_size = m_fmt.indentation() - m_column;
    auto count = (std::min) (availabe_space, remaining_indentation_size);
    std::fill(begin, begin + count, static_cast<CharT>(' '));
    m_column += static_cast<unsigned>(count);
    return begin + count;
}

template <typename CharT>
CharT* multiline_base64_pm_input<CharT>::write_whole_block_in_this_line(CharT* it)
{
    BOOST_ASSERT(m_block_sub_index == 0);
    BOOST_ASSERT(m_column >= m_fmt.indentation());
    BOOST_ASSERT(m_column + 4 < m_total_line_length);
    BOOST_ASSERT(m_index < num_bytes());

    auto arr = encode(bytes() + m_index, num_bytes() - m_index);
    it[0] = arr[0];
    it[1] = arr[1];
    it[2] = arr[2];
    it[3] = arr[3];
    m_column += 4;
    m_index += 3;
    return it + 4;
}

template <typename CharT>
CharT* multiline_base64_pm_input<CharT>::begin_partial_block(CharT* it)
{
    BOOST_ASSERT(m_block_sub_index == 0);

    m_split_block = encode( bytes() + m_index
                          , num_bytes() - m_index);
    return continue_partial_block(it);
}

template <typename CharT>
CharT* multiline_base64_pm_input<CharT>::continue_partial_block(CharT* it)
{
    BOOST_ASSERT(m_column >= m_fmt.indentation());
    BOOST_ASSERT(m_column < m_total_line_length);

    while(m_block_sub_index != 4)
    {
        *it = m_split_block[m_block_sub_index];
        ++ it;
        ++ m_column;
        ++ m_block_sub_index;
        if(m_column == m_total_line_length)
        {
            it = write_eol(it);
            m_column = 0;
            if(m_fmt.indentation() > 0)
            {
                break;
            }
        }
    }
    if(m_block_sub_index == 4)
    {
        m_block_sub_index = 0;
        m_index += 3;
    }
    return it;

}

template <typename CharT>
CharT* multiline_base64_pm_input<CharT>::write_eol(CharT* it)
{
    * it = facet().eol[0];
    if(facet().eol[1] != '\0')
    {
        * ++it = facet().eol[1];
    }
    return it + 1;
}


template <typename CharT>
class base64_printer: public strf::printer<CharT>
{
public:

    base64_printer
        ( strf::output_writer<CharT>& out
        , base64_facet facet
        , const base64_input_with_format& fmt );

    int remaining_width(int w) const override;

    std::size_t necessary_size() const override;

    void write() const override;

private:

    strf::output_writer<CharT>& m_out;
    const base64_facet m_facet;
    const base64_input_with_format m_fmt;
};

template <typename CharT>
base64_printer<CharT>::base64_printer
    ( strf::output_writer<CharT>& out
    , base64_facet facet
    , const base64_input_with_format& fmt )
    : m_out(out)
    , m_facet(facet)
    , m_fmt(fmt)
{
}

template <typename CharT>
int base64_printer<CharT>::remaining_width(int w) const
{
    (void)w;
    return 0;
}

template <typename CharT>
std::size_t base64_printer<CharT>::necessary_size() const
{
    base64_common_impl impl{m_facet};
    return impl.necessary_size(m_fmt.value().num_bytes, m_fmt.indentation());
}

//[ base64_printer__write
template <typename CharT>
void base64_printer<CharT>::write() const
{
    if(m_facet.single_line())
    {
        single_line_base64_pm_input<CharT> pm_input{m_facet, m_fmt};
        m_out.put(pm_input);
    }
    else
    {
        multiline_base64_pm_input<CharT> pm_input{m_facet, m_fmt};
        m_out.put(pm_input);
    }
}
//]

} //namespace xxx

//[ make_printer_base64

namespace xxx {

template <typename CharT, typename FPack>
inline base64_printer<CharT> make_printer( strf::output_writer<CharT>& out
                                         , const FPack& fp
                                         , const base64_input_with_format& fmt )
{
  /*<< see [link facets_pack get_facet.]
>>*/auto facet = strf::get_facet<base64_facet_category, base64_input>(fp);
    return {out, facet, fmt};
}


template <typename CharT, typename FPack>
inline base64_printer<CharT> make_printer( strf::output_writer<CharT>& out
                                         , const FPack& fp
                                         , const base64_input& input )
{
    return make_printer(out, fp, base64_input_with_format{input});
}

} // namespace xxx

//]


void tests()
{
    const char* data = "The quick brown fox jumps over the lazy dog.";
    auto data_size = strlen(data);

    {
        auto result = strf::to_string(xxx::base64(data, data_size)) .value();
        BOOST_TEST(result == "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZy4=\r\n");
    }

    {
        // customizing line length, end of line and identation
        auto result = strf::to_string
            .facets(xxx::base64_facet{50, {'\n', '\0'}})
            (xxx::base64(data, data_size).indentation(4))
            .value();

        auto expected =
            "    VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYX\n"
            "    p5IGRvZy4=\n";

        BOOST_TEST(result == expected);
    }
    {
        // When the length of last line is exactly as base64_facet::line_length,
        auto result = strf::to_string
            .facets(xxx::base64_facet{30})
            (xxx::base64(data, data_size).indentation(4))
            .value();

        auto expected =
            "    VGhlIHF1aWNrIGJyb3duIGZveCBqdW\r\n"
            "    1wcyBvdmVyIHRoZSBsYXp5IGRvZy4=\r\n";

        BOOST_TEST(result == expected);
    }
    {
        // When base64_facet::line_length == 1
        auto result = strf::to_string
            .facets(xxx::base64_facet{1, {'\n', '\0'}})
            (xxx::base64("  >  ?", 6).indentation(2))
            .value();

        BOOST_TEST(result == "  I\n  C\n  A\n  +\n  I\n  C\n  A\n  /\n");
    }
    {
        // When base64_facet::line_length == 3
        auto result = strf::to_string
            .facets(xxx::base64_facet{3, {'\n', '\0'}})
            (xxx::base64("  >  ?", 6).indentation(2))
            .value();
        BOOST_TEST(result == "  ICA\n  +IC\n  A/\n");
    }
    {
        // When base64_facet::line_length == 4
        auto result = strf::to_string
            .facets(xxx::base64_facet{4, {'\n', '\0'}})
            (xxx::base64("  >  ?", 6).indentation(2))
            .value();
        BOOST_TEST(result == "  ICA+\n  ICA/\n");
    }
    {
        // The default character for index 62 is '+'
        // and for index 63 is '/'
        auto result = strf::to_string(xxx::base64("  >  ?", 6)).value();
        BOOST_TEST(result == "ICA+ICA/\r\n");
    }

    {
        // customizing characters for index 62 and 63
        auto result = strf::to_string
            .facets(xxx::base64_facet{50, {'\r', '\n'}, '-', '_'})
            (xxx::base64("  >  ?", 6))
            .value();
        BOOST_TEST(result == "ICA-ICA_\r\n");
    }

    {
        // when base64_facet::line_length == 0'
        // then the result has no end of line
        auto result = strf::to_string
            .facets(xxx::base64_facet{0, {'\r', '\n'}})
            (xxx::base64("  >  ?", 6))
            .value();

        BOOST_TEST(result == "ICA+ICA/");
    }
    {
        // when base64_facet::eol[0] == '\0'
        // then the result has no end of line
        auto result = strf::to_string
            .facets(xxx::base64_facet{50, {'\0', '\n'}, '-', '_'})
            (xxx::base64("  >  ?", 6))
            .value();
        BOOST_TEST(result == "ICA-ICA_");
    }
    {
        // test indentation on single line mode
        auto result = strf::to_string
            .facets(xxx::base64_facet{0})
            (xxx::base64("  >  ?", 6).indentation(4))
            .value();
        BOOST_TEST(result == "    ICA+ICA/");
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
            .facets(xxx::base64_facet{50, {'\n', '\0'}})
            (strf::fmt_range(vec, "------------\n").indentation(4))
            .value();

        auto expected =
            "    YWJj\n"
            "------------\n"
            "    YWJjZA==\n"
            "------------\n"
            "    YWJjZGU=\n";

        BOOST_TEST(result == expected);
    }

}


void sample()
{
//[base64_sample
    const char* msg  = "The quick brown fox jumps over the lazy dog.";

    auto obtained = strf::to_string
        .facets(xxx::base64_facet{50, {'\n', '\0'}})
        ( xxx::base64(msg, strlen(msg)).indentation(4) )
        .value();

    auto expected =
        "    VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYX\n"
        "    p5IGRvZy4=\n";

    BOOST_ASSERT(obtained == expected);
//]
    (void)expected;
};

int main()
{
    tests();
    sample();
    return boost::report_errors();
}
