#include <boost/stringify.hpp>
#include <array>

namespace strf = boost::stringify::v0;

namespace xxx {

struct base64_facet_category;

struct base64_facet final
{
    typedef base64_facet_category category;

    int line_length = 64;
    char eol[2] = {'\r', '\n'};
    char char62 = '+';
    char char63 = '/';
};

struct base64_facet_category
{
    static const base64_facet& get_default()
    {
        static const base64_facet obj{};
        return obj;
    }
};


struct base64_tag
{
    const unsigned char* bytes = nullptr;
    std::size_t num_bytes = 0;;
};

#if 0

struct fmt_base64
{
    fmt_base64(base64_tag d)
        : value(d)
    {
    }

    fmt_base64&& indentation(int _) &&
    {
        m_indentation = _;
        return static_cast<fmt_base64&&>(*this);
    }

    int indentation() const
    {
        return m_indentation;
    }

    int m_indentation = 0;
    base64_tag value;
};

#else

template <typename T>
class base64_formatting
{
    using derived = strf::fmt_derived<base64_formatting<T>, T>;

public:

    template <typename U>
    using fmt_other = base64_formatting<U>;

    base64_formatting() = default;

    template <typename U>
    base64_formatting(const base64_formatting<U>& other)
        : m_indentation(other.indentation())
    {
    }

    derived&& indentation(int _) &&
    {
        m_indentation = _;
        return static_cast<derived&&>(*this);
    }

    int indentation() const
    {
        return m_indentation;
    }

private:

    int m_indentation = 0;
};

struct fmt_base64: public base64_formatting<fmt_base64>
{
    fmt_base64(const base64_tag& v)
        : value(v)
    {
    }

    template <typename U>
    fmt_base64(const base64_tag& v, const base64_formatting<U>& fmt)
        : base64_formatting<fmt_base64>(fmt)
        , value(v)
    {
    }

    base64_tag value;
};

#endif

inline fmt_base64 base64(const void* bytes, std::size_t num_bytes)
{
    return base64_tag{reinterpret_cast<const unsigned char*>(bytes), num_bytes};
}


class base64_printer_impl
{
public:

    base64_printer_impl(base64_facet f) : m_facet(f) {};

    std::size_t length(std::size_t num_bytes, int indentation) const
    {
        std::size_t num_digits = 4 * (num_bytes + 2) / 3;
        std::size_t num_lines
            = (num_digits + m_facet.line_length - 1) / m_facet.line_length;

        std::size_t eol_size = 1 + (m_facet.eol[1] != '\0');

        return num_digits + num_lines * (indentation + eol_size);
    }

    std::array<char, 4> encode(const unsigned char* octets, int num) const
    {
        return
            { encode(octets[0] >> 2)
            , encode(((octets[0] & 0x03) << 4) |
                     (num < 2 ? 0 : ((octets[1] & 0xF0) >> 4)))
            , num < 2 ? '=' : encode(((octets[1] & 0x0F) << 2) |
                                     (num < 3 ? 0 : ((octets[2] & 0xC0) >> 6)))
            , num < 3 ? '=' : encode(octets[2] & 0x3F)
            };
    }

    char encode(int hextet) const
    {
        BOOST_ASSERT(0 <= hextet && hextet <= 63);
        int ch =
            hextet < 26 ?  static_cast<int>('A') + hextet :
            hextet < 52 ?  static_cast<int>('a') + hextet - 26 :
            hextet < 62 ?  static_cast<int>('0') + hextet - 52 :
            hextet == 62 ? static_cast<int>(m_facet.char62) :
          /*hextet == 63*/ static_cast<int>(m_facet.char63) ;

        return static_cast<char>(ch);
    }

    base64_facet facet() const
    {
        return m_facet;
    }

private:

    const base64_facet m_facet;
};



template <typename CharT>
class base64_printer: public strf::printer<CharT>
{
public:

    template <typename FPack>
    base64_printer
        ( strf::output_writer<CharT>& out
        , const FPack& fp
        , const fmt_base64& fmt )
        : base64_printer(out, strf::get_facet<base64_facet_category, base64_tag>(fp), fmt)
    {
    }

    base64_printer
        ( strf::output_writer<CharT>& out
        , base64_facet facet
        , const fmt_base64& fmt )
        : m_out(out)
        , m_impl(facet)
        , m_data(fmt.value)
        , m_indentation(fmt.indentation())
    {
    }

    int remaining_width(int w) const override
    {
        (void)w;
        return 0;
    }

    std::size_t length() const override
    {
        return m_impl.length(m_data.num_bytes, m_indentation);
    }

    void write() const override
    {
        m_column = 0;
        for(std::size_t i = 0; i < m_data.num_bytes; i += 3)
        {
            auto arr = m_impl.encode(&m_data.bytes[i], m_data.num_bytes - i);
            put_digit(arr[0]);
            put_digit(arr[1]);
            put_digit(arr[2]);
            put_digit(arr[3]);
        }
    }

private:

    void put_digit(CharT ch) const
    {
        if(m_column == 0 && m_indentation != 0)
        {
            m_out.put(m_indentation, static_cast<CharT>(' '));
        }
        m_out.put(ch);
        if(++m_column == m_impl.facet().line_length)
        {
            write_eol();
            m_column = 0;
        }
    }

    void write_eol() const
    {
        CharT eol0 = m_impl.facet().eol[0];
        CharT eol1 = m_impl.facet().eol[1];
        if(eol0 != 0)
        {
            m_out.put(eol0);
            if(eol1 != 0)
            {
                m_out.put(eol1);
            }
        }
    }

    strf::output_writer<CharT>& m_out;
    base64_printer_impl m_impl;
    const base64_tag& m_data;
    int m_indentation;
    mutable int m_column = 0;
};


template <typename CharT, typename FPack>
inline base64_printer<CharT> stringify_make_printer
    ( strf::output_writer<CharT>& out
    , const FPack& fp
    , const base64_tag& d )
{
    return {out, fp, d};
}

template <typename CharT, typename FPack>
inline base64_printer<CharT> stringify_make_printer
    ( strf::output_writer<CharT>& out
    , const FPack& fp
    , const fmt_base64& f )
{
    return {out, fp, f};
}

inline fmt_base64 stringify_fmt(const base64_tag& d)
{
    return d;
}

} // namespace xxx


int main()
{
    //const char* msg  = "Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure.";

    const char* msg1 = "any carnal pleas";

    
    auto s = strf::to_u16string
        //.facets(xxx::base64_facet{76, {'\n', '\0'}})
        (xxx::base64(msg1, strlen(msg1)).indentation(4))
        .value();

    BOOST_ASSERT(s == u"    YW55IGNhcm5hbCBwbGVhcw==");
    
    //fflush(stdout);
    //BOOST_ASSERT(false);

    return 0;
};
