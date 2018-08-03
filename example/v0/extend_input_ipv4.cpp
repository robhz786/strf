#include <boost/stringify.hpp>
#include <cstdint>
#include <vector>

namespace strf = boost::stringify::v0;

namespace xxx {

struct ipv4_addr
{
    ipv4_addr(std::uint32_t _) : addr(_) {}

    ipv4_addr
        ( unsigned char byte0
        , unsigned char byte1
        , unsigned char byte2
        , unsigned char byte3 )
        : bytes{byte0, byte1, byte2, byte3}
    {
    }

    ipv4_addr(const ipv4_addr&) = default;

    union
    {
        unsigned char bytes[4];
        std::uint32_t addr;
    };
};


template <typename CharT, typename FPack>
inline auto stringify_make_printer
    ( strf::output_writer<CharT>& out
    , const FPack& ft
    , ipv4_addr addr )
{
    (void)ft;
    constexpr CharT dot('.');
    return strf::stringify_make_printer<CharT, strf::facets_pack<>>
        ( out
        , strf::facets_pack<>{}
        , strf::join()
            ( addr.bytes[0], dot
            , addr.bytes[1], dot
            , addr.bytes[2], dot
            , addr.bytes[3] ));
}


class fmt_ipv4_addr: public strf::align_formatting<fmt_ipv4_addr>
{
public:

    fmt_ipv4_addr(ipv4_addr _) : addr(_) {}
    fmt_ipv4_addr(const fmt_ipv4_addr&) = default;

    template <typename U>
    fmt_ipv4_addr(ipv4_addr value, const strf::align_formatting<U>& fmt)
        : strf::align_formatting<fmt_ipv4_addr>(fmt)
        , addr(value)
    {
    }

    void operator%(int) const = delete;

    ipv4_addr addr;
};


fmt_ipv4_addr stringify_fmt(ipv4_addr x)
{
    return {x};
}


template <typename CharT, typename FPack>
inline auto stringify_make_printer
    ( strf::output_writer<CharT>& out
    , const FPack& fp
    , fmt_ipv4_addr fmt_addr )
{
    (void)fp;
    constexpr CharT dot('.');

    return strf::stringify_make_printer<CharT, strf::facets_pack<>>
        ( out
        , strf::facets_pack<>{}
        , strf::join(fmt_addr.width(), fmt_addr.alignment(), fmt_addr.fill())
            ( fmt_addr.addr.bytes[0], dot
            , fmt_addr.addr.bytes[1], dot
            , fmt_addr.addr.bytes[2], dot
            , fmt_addr.addr.bytes[3] ));
}

} // namespace xxx


int main()
{
    xxx::ipv4_addr addr {146, 20, 110, 251};

    auto s1 = strf::to_string("The IP address of boost.org is ", addr).value();
    BOOST_ASSERT(s1 == "The IP address of boost.org is 146.20.110.251");

    auto s2 = strf::to_string("boost.org :", strf::right(addr, 18)) .value();
    BOOST_ASSERT(s2 == "boost.org :    146.20.110.251");

    std::vector<xxx::ipv4_addr> vec { {127, 0, 0, 1}, addr, {110, 110, 110, 110} };

    auto s3 = strf::to_string("[", strf::range(vec, " ; "), "]").value();
    BOOST_ASSERT(s3 == "[127.0.0.1 ; 146.20.110.251 ; 110.110.110.110]");

    auto s4 = strf::to_string("[", strf::fmt_range(vec) > 16, "]").value();
    BOOST_ASSERT(s4 == "[       127.0.0.1  146.20.110.251 110.110.110.110]");

    return 0;
}
