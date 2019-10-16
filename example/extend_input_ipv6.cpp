//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include <cstdint>
#include <vector>

namespace strf = boost::stringify::v0;

namespace xxx {

//[ ipv6address
struct ipv6address
{
    std::uint16_t hextets[8];
};
//]

static int abbreviation(ipv6address addr)
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

//[ ipv6_format
enum class ipv6_format_style{small, average, big};

struct ipv6_format
{
    template <class T>
    class fn
    {
    public:

        constexpr fn() = default;

        template <typename U>
        constexpr fn(const fn<U>& u) : _style(u._style)
        {
        }

        // format functions

        constexpr T&& big() &&
        {
            _style = ipv6_format_style::big;
            return static_cast<T&&>(*this);
        }

        constexpr T&& small() &&
        {
            _style = ipv6_format_style::small;
            return static_cast<T&&>(*this);
        }

        // observers

        constexpr bool is_small() const
        {
            return _style == ipv6_format_style::small;
        }

        constexpr bool is_big() const
        {
            return _style == ipv6_format_style::big;
        }

    private:

        template <typename> friend class fn;

        ipv6_format_style _style = ipv6_format_style::average;

    }; // ipv6_format::template fn

}; // ipv6_format

//]

//[ ipv6addr_with_format
using ipv6addr_with_format = strf::value_with_format< ipv6address
                                                    , ipv6_format
                                                    , strf::alignment_format >;

inline auto make_fmt(strf::tag, const ipv6address& addr)
{
    return ipv6addr_with_format{addr};
}
//]

//[ ipv6_printer
template <typename CharT>
class ipv6_printer: public strf::dynamic_join_printer<CharT>
{
public:

    template <typename FacetsPack>
    ipv6_printer(const FacetsPack& fp, ipv6addr_with_format fmt)
        : ipv6_printer
          ( fmt
          , fmt.is_big() ? 4 : 0
          , strf::facets_pack< strf::encoding<CharT>
                             , strf::encoding_error
                             , strf::surrogate_policy >
              { strf::get_facet<strf::encoding_c<CharT>, ipv6_printer>(fp)
              , strf::get_facet<strf::encoding_error_c, ipv6_printer>(fp)
              , strf::get_facet<strf::surrogate_policy_c, ipv6_printer>(fp) } )
    {
    }

    ipv6_printer(ipv6_printer&&) = default;

protected:

    void compose(strf::printers_receiver<CharT>& out) const override;

    strf::alignment_format_data formatting() const override;

private:

    ipv6_printer( ipv6addr_with_format fmt
                , int precision
                , strf::facets_pack< strf::encoding<CharT>
                                   , strf::encoding_error
                                   , strf::surrogate_policy > fp );

    void compose_non_abbreviated(strf::printers_receiver<CharT>& out) const;
    void compose_abbreviated(strf::printers_receiver<CharT>& out) const;

    ipv6addr_with_format _fmt;

  /*<< `printer_impl<CharT, FPack, Arg>` is equivalent to
     `decltype(make_printer(ob, fp, std::declval<Arg>())`
      where the type of `ob` is `strf::basic_outbuf<CharT>&`,
      and the type of `fp` is `const Fpack&`.
      Hence the type of `_colon` derives from `printer<CharT>`,
      and so do the elements of `_hextets`.
 >>*/strf::printer_impl<CharT, strf::facets_pack<>, CharT> _colon;

    using fmt_hextet = decltype(strf::fmt(ipv6address{}.hextets[0]).hex().p(0));

    strf::printer_impl<CharT, strf::facets_pack<>, fmt_hextet> _hextets[8];
};
//]

//[ipv6_printer__contructor
template <typename CharT>
ipv6_printer<CharT>::ipv6_printer
    ( ipv6addr_with_format fmt
    , int precision
    , strf::facets_pack< strf::encoding<CharT>
                       , strf::encoding_error
                       , strf::surrogate_policy > fp )
    : strf::dynamic_join_printer<CharT>
        { strf::get_facet<strf::encoding_c<CharT>, void>(fp)
        , strf::get_facet<strf::encoding_error_c, void>(fp)
        , strf::get_facet<strf::surrogate_policy_c, void>(fp) }
    , _fmt(fmt)
    , _colon{strf::make_printer<CharT>(fp, static_cast<CharT>(':'))}
    , _hextets
        { { strf::make_printer<CharT>( /*<< It is not a problem if `fp`
        is a temporary object. It won't lead to dangling references because,
        by convention, a printer class don't store any reference to the
        facets_pack that it is been constructed with.
                     >>*/fp, strf::hex(fmt.value().hextets[0]).p(precision)) }
        , { strf::make_printer<CharT>(fp, strf::hex(fmt.value().hextets[1]).p(precision)) }
        , { strf::make_printer<CharT>(fp, strf::hex(fmt.value().hextets[2]).p(precision)) }
        , { strf::make_printer<CharT>(fp, strf::hex(fmt.value().hextets[3]).p(precision)) }
        , { strf::make_printer<CharT>(fp, strf::hex(fmt.value().hextets[4]).p(precision)) }
        , { strf::make_printer<CharT>(fp, strf::hex(fmt.value().hextets[5]).p(precision)) }
        , { strf::make_printer<CharT>(fp, strf::hex(fmt.value().hextets[6]).p(precision)) }
        , { strf::make_printer<CharT>(fp, strf::hex(fmt.value().hextets[7]).p(precision)) } }
{
}

//]


//[ ipv6_printer__formatting
template <typename CharT>
strf::alignment_format_data ipv6_printer<CharT>::formatting() const
{
    return _fmt.get_alignment_format_data();
}
//]


//[ ipv6_printer__compose
template <typename CharT>
void ipv6_printer<CharT>::compose(strf::printers_receiver<CharT>& out) const
{
    if(_fmt.is_small())
    {
        compose_abbreviated(out);
    }
    else
    {
        compose_non_abbreviated(out);
    }
}


template <typename CharT>
void ipv6_printer<CharT>::compose_non_abbreviated
    ( strf::printers_receiver<CharT>& out ) const
{
    out.put(_hextets[0]);
    for(int i = 1; i < 8; ++i)
    {
        out.put(_colon);
        out.put(_hextets[i]);
    }
}


template <typename CharT>
void ipv6_printer<CharT>::compose_abbreviated
    ( strf::printers_receiver<CharT>& out ) const
{
    int abbr_bits = /*<<
    You can see the implementation of the `abbreviation` function
    in the [@../../example/v0/extend_input_ipv6.cpp source file].
    Each of the eight rightmost bits of the returned value tells
    whether the corresponding hextext shall be displayed or
    omitted in the abbreviated IPv6 representation
    >>*/ abbreviation(_fmt.value());
    bool prev_show = true;
    for (int i = 0; i < 8; ++i)
    {
        bool show_hextet = abbr_bits & 1;
        abbr_bits >>= 1;

        if (show_hextet)
        {
            if(i > 0)
            {
                out.put(_colon);
            }
            out.put(_hextets[i]);
        }
        else if(prev_show)
        {
            out.put(_colon);
        }
        prev_show = show_hextet;
    }
    if (!prev_show)
    {
        return out.put(_colon);
    }
}
//]

//[ipv6__make_printer
template <typename CharT, typename FPack>
inline ipv6_printer<CharT> make_printer( const FPack& fp
                                       , const ipv6address& addr )
{
    return ipv6_printer<CharT>{fp, ipv6addr_with_format{addr}};
}

template <typename CharT, typename FPack>
inline ipv6_printer<CharT> make_printer( const FPack& fp
                                       , const ipv6addr_with_format& addr )
{
    return ipv6_printer<CharT>{fp, addr};
}
//]

} // namespace xxx

int main()
{
    //[ ipv6_samples
    xxx::ipv6address addr{{0xaa, 0, 0, 0, 0xbb, 0, 0, 0xcc}};

    auto s = strf::to_string(addr);
    BOOST_ASSERT(s == "aa:0:0:0:bb:0:0:cc");

    s = strf::to_string(strf::fmt(addr).big());
    BOOST_ASSERT(s == "00aa:0000:0000:0000:00bb:0000:0000:00cc");

    s = strf::to_string(strf::right(addr, 20, U'.').small());
    BOOST_ASSERT(s == ".......aa::bb:0:0:cc");
    //]

    s = strf::to_string(strf::right(addr, 20));
    BOOST_ASSERT(s == "  aa:0:0:0:bb:0:0:cc");

    s = strf::to_string(strf::join_right(22, U'.')(strf::left(addr, 20)));
    BOOST_ASSERT(s == "..aa:0:0:0:bb:0:0:cc  ");

    s = strf::to_string(strf::center(addr, 20));
    BOOST_ASSERT(s == " aa:0:0:0:bb:0:0:cc ");

    std::vector<xxx::ipv6address> vec =
        { {{0, 0, 0, 0, 0, 0}}
        , {{0, 0, 0, 1, 2, 3}}
        , {{1, 2, 3, 0, 0, 0}}
        , {{0, 0, 1, 0, 0, 0}}
        , {{0, 0, 0, 1, 0, 0}} };

    s = strf::to_string
        ( strf::fmt_range(vec, "\n").small().fill(U'~') > 20
        , "\n" );

    const char* expected_result =
        "~~~~~~~~~~~~~~~~~~::\n"
        "~~~~~~~~~::1:2:3:0:0\n"
        "~~~~~~~~~~~~~1:2:3::\n"
        "~~~~~~~~~~~~~0:0:1::\n"
        "~~~~~~~~~~~0:0:0:1::\n";

    BOOST_ASSERT(s == expected_result);

    (void)expected_result;
    return 0;
}
