//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <strf/to_string.hpp>

namespace xxx {

struct ipv4address
{
    unsigned char bytes[4];
};

} // namespace xxx

namespace strf {

template <>
struct printable_traits<xxx::ipv4address> {

    using representative_type = xxx::ipv4address;
    using forwarded_type = xxx::ipv4address;
    using formatters = strf::tag<strf::alignment_formatter>;

    template <typename CharT>
    static auto transform_arg(forwarded_type arg)
    {
        constexpr CharT dot = '.';
        const auto* bytes = arg.bytes;
        return strf::join(bytes[0], dot, bytes[1], dot, bytes[2], dot, bytes[3]);
    }

    template <typename CharT, typename PreMeasurements, typename FPack>
    static auto make_input
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , forwarded_type arg )
    {
        auto arg2 = transform_arg<CharT>(arg);
        return strf::make_default_printer_input<CharT>(pre, fp, arg2);
    }

    template <typename CharT, typename PreMeasurements, typename FPack, typename... T>
    static auto make_input
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::printable_with_fmt<T...> arg )
    {
        auto arg2 = transform_arg<CharT>(arg.value());
        auto arg3 = arg2.set_alignment_format(arg.get_alignment_format());
        return strf::make_default_printer_input<CharT>(pre, fp, arg3);
    }
};

} // namespace strf

int main()
{
    xxx::ipv4address addr {{198, 199, 109, 141}};

    // basic sample
    auto str = strf::to_string("The IP address of isocpp.org is ", addr);
    assert(str == "The IP address of isocpp.org is 198.199.109.141");

    // formatted ipv4address
    str = strf::to_string("isocpp.org: ", strf::right(addr, 20, U'.'));
    assert(str == "isocpp.org: .....198.199.109.141");

    // ipv4address in ranges
    const std::vector<xxx::ipv4address> vec = { {{127, 0, 0, 1}}
                                              , {{146, 20, 110, 251}}
                                              , {{110, 110, 110, 110}} };
    str = strf::to_string("[", strf::separated_range(vec, " ; "), "]");
    assert(str == "[127.0.0.1 ; 146.20.110.251 ; 110.110.110.110]");

    // formatted ipv4address in ranges
    str = strf::to_string("[", strf::fmt_separated_range(vec, " ; ") > 15, "]");
    assert(str == "[      127.0.0.1 ;  146.20.110.251 ; 110.110.110.110]");

    return 0;
}


