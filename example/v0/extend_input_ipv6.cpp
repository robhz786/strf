//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include <cstdint>
#include <vector>

namespace strf = boost::stringify::v0;

namespace xxx {

//[ ipv6_address
struct ipv6_address
{
    std::uint16_t hextets[8];
};
//]

static int abbreviation(ipv6_address addr)
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

//[ ipv6_formatting
template <class T>
class ipv6_formatting: public strf::align_formatting<T>
{
  /*<< `fmt_derived<This, Derived>` is an alias to `This` when `Derived`
        is `void`, otherwise it is an alias to `Derived`
>>*/using derived_type = strf::fmt_derived<ipv6_formatting<T>, T>;

public:

  /*<< This template alias is a required by [link ranges `fmt_range`]
>>*/template <typename U>
    using fmt_other = ipv6_formatting<U>;

  /*<< Default constructor is also a required by [link ranges `fmt_range`]
>>*/constexpr ipv6_formatting() = default;

    constexpr ipv6_formatting(const ipv6_formatting&) = default;

  /*<< This kind of copy constructor template is also required by [link ranges `fmt_range`]
       ( it has to be a template like this )
>>*/template <typename U>
    constexpr ipv6_formatting(const ipv6_formatting<U>& u)
        : strf::align_formatting<T>(u)
        , m_abbreviate(u.abbreviated())
    {
    }

    constexpr derived_type&& abbreviate() &&
    {
        m_abbreviate = true;
        return static_cast<derived_type&&>(*this);
    }

    constexpr bool abbreviated() const
    {
        return m_abbreviate;
    }

    void operator%(int) const = delete;

private:

    bool m_abbreviate = false;
};
//]

//[ fmt_ipv6
class fmt_ipv6: public ipv6_formatting<fmt_ipv6>
{
public:

    fmt_ipv6(const fmt_ipv6&) = default;

    fmt_ipv6(ipv6_address a) : addr(a) {}

 /*<< This constructor template is required by [link ranges fmt_range].
    The first argument will be `*it`, where `it` is an iterator of the range.
>>*/template <typename U>
    fmt_ipv6(ipv6_address a, const ipv6_formatting<U>& fmt)
        : ipv6_formatting<fmt_ipv6>(fmt)
        , addr(a)
    {
    }

    ipv6_address addr;
};


inline fmt_ipv6 make_fmt(strf::tag, const ipv6_address& addr)
{
    return {addr};
}
//]

//[ ipv6_printer
template <typename CharT>
class ipv6_printer: public strf::dynamic_join_printer<CharT>
{
public:

    ipv6_printer(strf::output_writer<CharT>& out, fmt_ipv6 fmt);

protected:

    void compose(strf::printers_receiver<CharT>& out) const override;
    strf::align_formatting<void> formatting() const override;

private:

    void compose_non_abbreviated(strf::printers_receiver<CharT>& out) const;
    void compose_abbreviated(strf::printers_receiver<CharT>& out) const;

    strf::facets_pack<> fp;
    fmt_ipv6 m_fmt;
    strf::printer_impl<CharT, strf::facets_pack<>, std::uint16_t> m_hextets[8];
  /*<< The `printer_impl` is a template alias.
     `printer_impl<CharT, FPack, Arg>` is equivalent to
     `decltype(make_printer(out, fp, std::declval<Arg>())`
      where the type `out` of is `output_writer<CharT>&`,
      and the type of `fp` is `const Fpack&`
 >>*/strf::printer_impl<CharT, strf::facets_pack<>, CharT> m_colon;
};
//]

//[ipv6_printer__contructor
template <typename CharT>
ipv6_printer<CharT>::ipv6_printer( strf::output_writer<CharT>& out
                                 , fmt_ipv6 fmt )
    : strf::dynamic_join_printer<CharT>{out}
    , m_fmt(fmt)
    , m_hextets
        { {strf::make_printer(out, /*<< Note that we can not use
          `strf::pack()` as the argument instead of `fp` here. Because
           then the printer object would hold a dangling reference.
           >>*/fp, strf::hex(fmt.addr.hextets[0]))}
        , {strf::make_printer(out, fp, strf::hex(fmt.addr.hextets[1]))}
        , {strf::make_printer(out, fp, strf::hex(fmt.addr.hextets[2]))}
        , {strf::make_printer(out, fp, strf::hex(fmt.addr.hextets[3]))}
        , {strf::make_printer(out, fp, strf::hex(fmt.addr.hextets[4]))}
        , {strf::make_printer(out, fp, strf::hex(fmt.addr.hextets[5]))}
        , {strf::make_printer(out, fp, strf::hex(fmt.addr.hextets[6]))}
        , {strf::make_printer(out, fp, strf::hex(fmt.addr.hextets[7]))} }
    , m_colon{strf::make_printer(out, fp, static_cast<CharT>(':'))}
{
}
//]


//[ ipv6_printer__formatting
template <typename CharT>
strf::align_formatting<void> ipv6_printer<CharT>::formatting() const
{
    return m_fmt;
}
//]


//[ ipv6_printer__compose
template <typename CharT>
void ipv6_printer<CharT>::compose(strf::printers_receiver<CharT>& out) const
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


template <typename CharT>
void ipv6_printer<CharT>::compose_non_abbreviated
    ( strf::printers_receiver<CharT>& out ) const
{
    bool good = out.put(m_hextets[0]);
    for(int i = 1; good && i < 8; ++i)
    {
        good = good
            && out.put(m_colon)
            && out.put(m_hextets[i]);
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
    >>*/ abbreviation(m_fmt.addr);
    bool prev_show = true;
    bool good = true;
    for (int i = 0; good && i < 8; ++i) 
    {
        bool show_hextet = abbr_bits & 1;
        abbr_bits >>= 1;

        if (show_hextet)
        {
            if(i > 0)
            {
                good = good && out.put(m_colon);
            }
            good = good && out.put(m_hextets[i]);
        }
        else if(prev_show)
        {
            good = good && out.put(m_colon);
        }
        prev_show = show_hextet;
    }
    if (!prev_show && good)
    {
        out.put(m_colon);
    }
}
//]

//[ipv6__make_printer
template <typename CharT, typename FPack>
inline ipv6_printer<CharT> make_printer
    ( strf::output_writer<CharT>& ow
    , const FPack& fp
    , const ipv6_address& addr )
{
    (void)fp;
    return ipv6_printer<CharT>{ow, addr};
}

template <typename CharT, typename FPack>
inline ipv6_printer<CharT> make_printer
    ( strf::output_writer<CharT>& ow
    , const FPack& fp
    , const fmt_ipv6& addr )
{
    (void)fp;
    return ipv6_printer<CharT>{ow, addr};
}
//]

} // namespace xxx

int main()
{
    //[ ipv6_samples
    xxx::ipv6_address addr{{0xaa, 0, 0, 0, 0xbb, 0, 0, 0xcc}};
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

    std::vector<xxx::ipv6_address> vec =
        { {{0, 0, 0, 0, 0, 0}}
        , {{0, 0, 0, 1, 2, 3}}
        , {{1, 2, 3, 0, 0, 0}}
        , {{0, 0, 1, 0, 0, 0}}
        , {{0, 0, 0, 1, 0, 0}} };

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
    //]
    return 0;
}
