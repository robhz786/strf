//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <strf.hpp>

//[ ipv4address_type
namespace xxx {

struct ipv4address
{
    unsigned char bytes[4];
};

//  The `alignment_format` class template provides the format functions
//  related to alignment.
using ipv4address_with_format = strf::value_with_format<ipv4address, strf::alignment_format>;

// The `rank<1>` paramenter is not used. Its only purpose is to  ensure that
// there is no other `make_fmt` function  around there with the same signature.
inline ipv4address_with_format make_fmt(  strf::rank<1>, ipv4address x) { return ipv4address_with_format{x}; }

template <typename CharT, typename FPack, typename Preview>
struct ipv4_printer_traits
{
    constexpr static auto make_input
        ( const FPack&, Preview& preview, xxx::ipv4address addr )
    {
        return strf::make_printer_input<CharT>
            ( strf::pack()
            , preview
            , strf::join( addr.bytes[0], CharT{'.'}
                        , addr.bytes[1], CharT{'.'}
                        , addr.bytes[2], CharT{'.'}
                        , addr.bytes[3] ) );
    }

    constexpr static auto make_input
        ( const FPack&, Preview& preview, xxx::ipv4address_with_format x)
    {
        return strf::make_printer_input<CharT>
            ( strf::pack()
            , preview
            , strf::join
                ( x.value().bytes[0], CharT{'.'}
                , x.value().bytes[1], CharT{'.'}
                , x.value().bytes[2], CharT{'.'}
                , x.value().bytes[3] )
                .set(x.get_alignment_format_data() ) );
    }
};

template <typename CharT, typename FPack, typename Preview>
ipv4_printer_traits<CharT, FPack, Preview> get_printer_traits(Preview&, xxx::ipv4address);

template <typename CharT, typename FPack, typename Preview>
ipv4_printer_traits<CharT, FPack, Preview> get_printer_traits
(Preview&, xxx::ipv4address_with_format);

} // namespace xxx

int main()
{
    xxx::ipv4address addr {{198, 199, 109, 141}};

    // basic sample
    auto s = strf::to_string("The IP address of isocpp.org is ", addr);
    assert(s == "The IP address of isocpp.org is 198.199.109.141");

    // ipv4address in ranges
    s = strf::to_string("isocpp.org: ", strf::right(addr, 20, U'.'));
    assert(s == "isocpp.org: .....198.199.109.141");

    // formatted ipv4address in ranges
    std::vector<xxx::ipv4address> vec = { {{127, 0, 0, 1}}
                                        , {{146, 20, 110, 251}}
                                        , {{110, 110, 110, 110}} };
    auto s2 = strf::to_string("[", strf::fmt_separated_range(vec, " ;") > 16, "]");
    assert(s2 == "[       127.0.0.1 ;  146.20.110.251 ; 110.110.110.110]");

    return 0;
}


