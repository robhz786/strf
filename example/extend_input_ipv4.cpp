//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <strf/to_string.hpp>

//[ ipv4address_type
namespace xxx {

struct ipv4address
{
    unsigned char bytes[4];
};

//  The `alignment_format` class template provides the format functions
//  related to alignment.
using ipv4address_with_format = strf::value_with_format<ipv4address, strf::alignment_format>;

} // namespace xxx

namespace strf {

constexpr xxx::ipv4address_with_format tag_invoke(strf::fmt_tag, xxx::ipv4address x) noexcept
{
    return xxx::ipv4address_with_format{ x };
}

template <typename CharT, typename FPack, typename Preview>
constexpr auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , xxx::ipv4address_with_format arg
    , const FPack& fp
    , Preview& preview ) noexcept
{
    return strf::make_default_printer_input<CharT>
        ( strf::join
            ( arg.value().bytes[0], CharT{'.'}
            , arg.value().bytes[1], CharT{'.'}
            , arg.value().bytes[2], CharT{'.'}
            , arg.value().bytes[3] )
            .set(arg.get_alignment_format_data())
        , fp, preview );
}

template <typename CharT, typename FPack, typename Preview>
constexpr auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , xxx::ipv4address arg
    , const FPack& fp
    , Preview& preview ) noexcept
{
    return strf::make_default_printer_input<CharT>
        ( strf::join
            ( arg.bytes[0], CharT{'.'}
            , arg.bytes[1], CharT{'.'}
            , arg.bytes[2], CharT{'.'}
            , arg.bytes[3] )
        , fp, preview );
}

} // namespace strf

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


