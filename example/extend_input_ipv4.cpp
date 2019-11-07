//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <vector>
#include <boost/detail/lightweight_test.hpp>

//[ ipv4address_type
namespace xxx {

struct ipv4address
{
    unsigned char bytes[4];
};

}
//]


//[ make_printer_ipv4address
#include <boost/stringify.hpp>

namespace xxx {

template <typename CharT, typename FPack, typename Preview>
auto make_printer(const FPack& fp, Preview& preview, ipv4address addr)
{
    (void)fp;
    return make_printer<CharT>
        ( /*<< Note we are not forwarding `fp` but instead passing an empty
facets pack. In others cases, however, you may want to propagate some or
all of the facets.
 >>*/strf::pack()
        , preview
        , strf::join( addr.bytes[0], CharT{'.'}
                    , addr.bytes[1], CharT{'.'}
                    , addr.bytes[2], CharT{'.'}
                    , addr.bytes[3] ) );
}

} // namespace xxx
//]


void basic_sample()
{
//[ ipv4_basic_sample
    xxx::ipv4address addr {{146, 20, 110, 251}};
    auto s = strf::to_string("The IP address of boost.org is ", addr);
    BOOST_TEST(s == "The IP address of boost.org is 146.20.110.251");
//]
}


//[ipv4address_with_format
namespace xxx {

using ipv4address_with_format = strf::value_with_format<ipv4address, /*<<
    The `alignment_format` class template provides the [link format_functions
    formatting functions] related to alignment. >>*/strf::alignment_format>;

inline ipv4address_with_format make_fmt( /*<< The `tag` paramenter is not used.
     Its only purpose is to ensure there is no other `make_fmt` function
     around there with the same signature. >>*/ strf::tag, ipv4address x) { return ipv4address_with_format{x}; }

} // namespace xxx
//]


//[make_printer_fmt_ipv4
namespace xxx {

template <typename CharT, typename FPack, typename Preview>
auto make_printer( const FPack& fp
                 , Preview& preview
                 , ipv4address_with_format fmt_addr )
{
    (void)fp;
    xxx::ipv4address addr = fmt_addr.value();
    return strf::make_printer<CharT>
        ( strf::pack()
        , preview
        , strf::join_align(fmt_addr.width(), fmt_addr.alignment(), fmt_addr.fill())
            ( addr.bytes[0], CharT{'.'}
            , addr.bytes[1], CharT{'.'}
            , addr.bytes[2], CharT{'.'}
            , addr.bytes[3] ) );
}

} // namespace xxx
//]

void sample_fmt_sample()
{

//[formatted_ipv4address
    xxx::ipv4address addr {{146, 20, 110, 251}};

    auto s = strf::to_string("boost.org: ", strf::right(addr, 20, U'.'));
    assert(s == "boost.org: ......146.20.110.251");
//]


//[formatted_ipv4address_in_ranges
    std::vector<xxx::ipv4address> vec = { {{127, 0, 0, 1}}
                                        , {{146, 20, 110, 251}}
                                        , {{110, 110, 110, 110}} };
    auto s2 = strf::to_string("[", strf::fmt_range(vec, " ;") > 16, "]");
    assert(s2 == "[       127.0.0.1 ;  146.20.110.251 ; 110.110.110.110]");
//]
    // auto s3 = strf::to_string("[", strf::range(vec, " ; "), "]");
    // assert(s3 == "[127.0.0.1 ; 146.20.110.251 ; 110.110.110.110]");
}


#include <vector>

int main()
{
    basic_sample();
    sample_fmt_sample();
    return boost::report_errors();
}
