#include <boost/stringify.hpp>
#include <cstdint>
#include <vector>

namespace strf = boost::stringify::v0;

namespace xxx {

struct ipv6_addr
{
    std::uint16_t hextets[8];
};


static int abbreviation(ipv6_addr addr)
{
    const auto* const begin  = & addr.hextets[0];
    const auto* const end    = & addr.hextets[8];

    int greatest_zgroup_size = 0;
    auto greatest_zgroup_it = end;
    auto it = std::find(begin, end, 0);
    while (it != end)
    {
        auto zgroup_size = 1;
        auto zgroup_it = it;
        while(++it != end && *it == 0)
        {
            ++ zgroup_size;
        }
        if (zgroup_size > 1 && zgroup_size > greatest_zgroup_size)
        {
            greatest_zgroup_size = zgroup_size;
            greatest_zgroup_it = zgroup_it;
        }
        it = std::find(it, end, 0);
    }
    int greatest_zgroup_idx = static_cast<int>(greatest_zgroup_it - begin);
    return (0xFF & ~((0xFF << greatest_zgroup_idx) ^
                     (0xFF << (greatest_zgroup_idx + greatest_zgroup_size))));
}


template <class T>
class ipv6_formatting: public strf::align_formatting<T>
{

    using derived = strf::fmt_derived<ipv6_formatting<T>, T>;

public:

    template <typename U>
    using fmt_other = ipv6_formatting<U>;

    constexpr ipv6_formatting() = default;

    constexpr ipv6_formatting(const ipv6_formatting&) = default;

    template <typename U>
    constexpr ipv6_formatting(const ipv6_formatting<U>& u)
        : strf::align_formatting<T>(u)
        , m_abbreviate(u.abbreviated())
    {
    }

    constexpr derived&& abbreviate() &&
    {
        m_abbreviate = true;
        return static_cast<derived&&>(*this);
    }

    constexpr bool abbreviated() const
    {
        return m_abbreviate;
    }

    void operator%(int) const = delete;

private:

    bool m_abbreviate = false;
};


class fmt_ipv6: public ipv6_formatting<fmt_ipv6>
{
public:

    fmt_ipv6(const fmt_ipv6&) = default;

    fmt_ipv6(ipv6_addr a) : addr(a) {}

    template <typename U>
    fmt_ipv6(ipv6_addr a, const ipv6_formatting<U>& fmt)
        : ipv6_formatting<fmt_ipv6>(fmt)
        , addr(a)
    {
    }

    ipv6_addr addr;
};


template <typename CharOut>
class ipv6_printer: public strf::streamed_printer<CharOut>
{
public:

    ipv6_printer(strf::output_writer<CharOut>& out, ipv6_addr addr)
        : strf::streamed_printer<CharOut>{out}
        , m_fmt(addr)
        , m_hextets
            { {out, fp, strf::hex(addr.hextets[0])}
            , {out, fp, strf::hex(addr.hextets[1])}
            , {out, fp, strf::hex(addr.hextets[2])}
            , {out, fp, strf::hex(addr.hextets[3])}
            , {out, fp, strf::hex(addr.hextets[4])}
            , {out, fp, strf::hex(addr.hextets[5])}
            , {out, fp, strf::hex(addr.hextets[6])}
            , {out, fp, strf::hex(addr.hextets[7])} }
        , m_colon{out, fp, static_cast<CharOut>(':')}

    {
    }

    ipv6_printer(strf::output_writer<CharOut>& out, fmt_ipv6 fmt)
        : strf::streamed_printer<CharOut>{out}
        , m_fmt(fmt)
        , m_hextets
            { {out, fp, strf::hex(fmt.addr.hextets[0])}
            , {out, fp, strf::hex(fmt.addr.hextets[1])}
            , {out, fp, strf::hex(fmt.addr.hextets[2])}
            , {out, fp, strf::hex(fmt.addr.hextets[3])}
            , {out, fp, strf::hex(fmt.addr.hextets[4])}
            , {out, fp, strf::hex(fmt.addr.hextets[5])}
            , {out, fp, strf::hex(fmt.addr.hextets[6])}
            , {out, fp, strf::hex(fmt.addr.hextets[7])} }
        , m_colon{out, fp, static_cast<CharOut>(':')}

    {
    }

protected:

    void compose(strf::stream<CharOut>& out) const override
    {
        if(m_fmt.abbreviated())
        {
            compose_abbreviated(out);
        }
        else
        {
            compose_non_abbreviated(out);
        }
    }

    strf::align_formatting<void> formatting() const override
    {
        return m_fmt;
    }

private:

    void compose_non_abbreviated(strf::stream<CharOut>& out) const
    {
        bool cont = out.put(m_hextets[0]);
        for(int i = 1; cont && i < 8; ++i)
        {
            cont = cont
                && out.put(m_colon)
                && out.put(m_hextets[i]);
        }
    }


    void compose_abbreviated(strf::stream<CharOut>& out) const
    {
        int abbr_bits = abbreviation(m_fmt.addr);
        bool prev_show = true;
        for (int i = 0; i < 8; ++i)
        {
            bool show_hextet = abbr_bits & 1;
            if (show_hextet)
            {
                if(i > 0)
                {
                    out.put(m_colon);
                }
                out.put(m_hextets[i]);
            }
            else if(prev_show)
            {
                out.put(m_colon);
            }
            prev_show = show_hextet;
            abbr_bits = abbr_bits >> 1;
        }
        if (!prev_show)
        {
            out.put(m_colon);
        }
    }


    strf::facets_pack<> fp;
    fmt_ipv6 m_fmt;
    strf::printer_impl<CharOut, strf::facets_pack<>, unsigned short> m_hextets[8];
    strf::printer_impl<CharOut, strf::facets_pack<>, CharOut> m_colon;
};


template <typename CharOut, typename FPack>
inline ipv6_printer<CharOut> stringify_make_printer
    ( boost::stringify::v0::output_writer<CharOut>& ow
    , const FPack& fp
    , const ipv6_addr& addr
    )
{
    (void)fp;
    return ipv6_printer<CharOut>{ow, addr};
}

template <typename CharOut, typename FPack>
inline ipv6_printer<CharOut> stringify_make_printer
    ( boost::stringify::v0::output_writer<CharOut>& ow
    , const FPack& fp
    , const fmt_ipv6& addr
    )
{
    (void)fp;
    return ipv6_printer<CharOut>{ow, addr};
}

inline fmt_ipv6 stringify_fmt(const ipv6_addr& addr)
{
    return {addr};
}

} // namespace xxx

int main()
{
    xxx::ipv6_addr addr{0xaa, 0, 0, 0, 0xbb, 0, 0, 0xcc};
    auto s = strf::to_string(addr).value();
    BOOST_ASSERT(s == "aa:0:0:0:bb:0:0:cc");

    s = strf::to_string(strf::right(addr, 20)).value();
    BOOST_ASSERT(s == "  aa:0:0:0:bb:0:0:cc");

    s = strf::to_string(strf::join_right(22, U'.')(strf::left(addr, 20))).value();
    BOOST_ASSERT(s == "..aa:0:0:0:bb:0:0:cc  ");

    s = strf::to_string(strf::center(addr, 20)).value();
    BOOST_ASSERT(s == " aa:0:0:0:bb:0:0:cc ");

    s = strf::to_string(strf::fmt(addr).abbreviate()) .value();
    BOOST_ASSERT(s == "aa::bb:0:0:cc");

    std::vector<xxx::ipv6_addr> vec =
        { {0,0,0,0,0,0}
        , {0,0,0,1,2,3}
        , {1,2,3,0,0,0}
        , {0,0,1,0,0,0}
        , {0,0,0,1,0,0} };

    s = strf::to_string
        ( strf::fmt_range(vec, "\n").abbreviate().fill(U'~') > 20, "\n" )
        .value();

    const char* expected_result =
        "~~~~~~~~~~~~~~~~~~::\n"
        "~~~~~~~~~::1:2:3:0:0\n"
        "~~~~~~~~~~~~~1:2:3::\n"
        "~~~~~~~~~~~~~0:0:1::\n"
        "~~~~~~~~~~~0:0:0:1::\n";

    BOOST_ASSERT(s == expected_result);

    return 0;
}
