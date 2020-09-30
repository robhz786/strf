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
struct print_traits<xxx::ipv4address> {

    using facet_tag = xxx::ipv4address;
    using forwarded_type = xxx::ipv4address;
    using fmt_type =  strf::value_with_formatters
        < strf::print_traits<xxx::ipv4address>
        , strf::alignment_formatter>;

    template <typename CharT>
    static auto transform_arg(forwarded_type arg)
    {
        constexpr CharT dot = '.';
        const auto* bytes = arg.bytes;
        return strf::join(bytes[0], dot, bytes[1], dot, bytes[2], dot, bytes[3]);
    }

    template <typename CharT, typename Preview, typename FPack>
    static auto make_printer_input(Preview& preview, const FPack& fp, forwarded_type arg)
    {
        auto arg2 = transform_arg<CharT>(arg);
        return strf::make_default_printer_input<CharT>(preview, fp, arg2);
    }

    template <typename CharT, typename Preview, typename FPack>
    static auto make_printer_input(Preview& preview, const FPack& fp, fmt_type arg)
    {
        auto join = transform_arg<CharT>(arg.value());
        auto aligned_join = join.set_alignment_format(arg.get_alignment_format());
        return strf::make_default_printer_input<CharT>(preview, fp, aligned_join);
    }
};

// constexpr ipv4address_printing tag_invoke(strf::print_traits_tag, xxx::ipv4address) noexcept
// {
//     return {};
// }

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


