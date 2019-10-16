#ifndef BOOST_STRINGIFY_V0_DYNAMIC_JOIN_PRINTER_HPP
#define BOOST_STRINGIFY_V0_DYNAMIC_JOIN_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/stringify/v0/detail/format_functions.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharOut>
class printers_receiver
{
public:

    virtual ~printers_receiver()
    {
    }

    virtual void put(const stringify::v0::printer<CharOut>& ) = 0;
};

namespace detail {

template <typename CharOut>
class width_sum: public printers_receiver<CharOut>
{
public:

    width_sum(int limit)
        : _limit(limit)
    {
    }

    void put(const stringify::v0::printer<CharOut>& p) override;

    int result() const
    {
        return _sum;
    }

private:

    int _limit;
    int _sum = 0;
};

template <typename CharOut>
void width_sum<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    _sum += p.width(_limit - _sum);
    //return _sum < _limit;
}


template <typename CharOut>
class necessary_size_sum: public printers_receiver<CharOut>
{
public:

    necessary_size_sum() = default;

    void put(const stringify::v0::printer<CharOut>& p) override;

    std::size_t accumulated_length() const
    {
        return _len;
    }

private:

    std::size_t _len = 0;
};

template <typename CharOut>
void necessary_size_sum<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    _len += p.necessary_size();
}

template <typename CharOut>
class serial_writer: public printers_receiver<CharOut>
{
public:

    serial_writer(stringify::v0::basic_outbuf<CharOut>& ob) noexcept
        : _ob(ob)
    {
    }

    void put(const stringify::v0::printer<CharOut>& p) override;

private:

    stringify::v0::basic_outbuf<CharOut>& _ob;
};

template <typename CharOut>
void serial_writer<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    p.write(_ob);
}

} // namespace detail

template <typename CharOut>
class dynamic_join_printer: public stringify::v0::printer<CharOut>
{
public:

    dynamic_join_printer
        ( stringify::v0::encoding<CharOut> enc
        , stringify::v0::encoding_error enc_err
        , stringify::v0::surrogate_policy allow_surr )
        : _encoding(enc)
        , _enc_err(enc_err)
        , _allow_surr(allow_surr)
    {
    }

    std::size_t necessary_size() const override;

    void write(stringify::v0::basic_outbuf<CharOut>& ob) const override;

    int width(int limit) const override;

protected:

    virtual stringify::v0::alignment_format_data formatting() const;

    virtual void compose(stringify::v0::printers_receiver<CharOut>& out) const = 0;

private:

    void write_with_fill
        ( int fillcount
        , stringify::v0::basic_outbuf<CharOut>& ob ) const;

    void write_without_fill
        ( stringify::v0::basic_outbuf<CharOut>& ob ) const;

    void write_fill
        ( int count
        , char32_t ch
        , stringify::v0::basic_outbuf<CharOut>& ob ) const;

    stringify::v0::encoding<CharOut> _encoding;
    const stringify::v0::encoding_error _enc_err;
    const stringify::v0::surrogate_policy _allow_surr;
};

template <typename CharOut>
stringify::v0::alignment_format_data
dynamic_join_printer<CharOut>::formatting() const
{
    return {};
}

template <typename CharOut>
std::size_t dynamic_join_printer<CharOut>::necessary_size() const
{
    std::size_t fill_len = 0;
    const auto fmt = formatting();
    if(fmt.width > 0)
    {
        stringify::v0::detail::width_sum<CharOut> wsum{fmt.width};
        compose(wsum);
        auto fillcount = ( fmt.width > wsum.result()
                         ? fmt.width - wsum.result()
                         : 0 );
        fill_len = _encoding.char_size(fmt.fill, _enc_err) * fillcount;
    }

    stringify::v0::detail::necessary_size_sum<CharOut> s;
    compose(s);
    return s.accumulated_length() + fill_len;
}

template <typename CharOut>
int dynamic_join_printer<CharOut>::width(int limit) const
{
    const auto fmt_width = formatting().width;
    if (fmt_width > limit)
    {
        return limit;
    }
    stringify::v0::detail::width_sum<CharOut> acc{limit};
    compose(acc);
    return std::max(acc.result(), fmt_width);
}

template <typename CharOut>
void dynamic_join_printer<CharOut>::write
    ( stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    auto fmt_width = formatting().width;
    if(fmt_width > 0)
    {
        stringify::v0::detail::width_sum<CharOut> ws{fmt_width};
        compose(ws);
        if (fmt_width > ws.result())
        {
            auto fillcount = fmt_width - ws.result();
            write_with_fill(fillcount, ob);
            return;
        }
    }
    write_without_fill(ob);
}

template <typename CharOut>
void dynamic_join_printer<CharOut>::write_without_fill
    ( stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    stringify::v0::detail::serial_writer<CharOut> s(ob);
    compose(s);
}

template <typename CharOut>
void dynamic_join_printer<CharOut>::write_with_fill
    ( int fillcount
    , stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    auto fmt = formatting();
    char32_t fill_char = fmt.fill;
    switch (fmt.alignment)
    {
        case stringify::v0::text_alignment::left:
        {
            write_without_fill(ob);
            write_fill(fillcount, fill_char, ob);
            break;
        }
        case stringify::v0::text_alignment::center:
        {
            auto halfcount = fillcount >> 1;
            write_fill(halfcount, fill_char, ob);
            write_without_fill(ob);
            write_fill(fillcount - halfcount, fill_char, ob);
            break;
        }
        //case stringify::v0::text_alignment::split:
        //case stringify::v0::text_alignment::right:
        default:
        {
            write_fill(fillcount, fill_char, ob);
            write_without_fill(ob);
        }
    }
}

template <typename CharOut>
void dynamic_join_printer<CharOut>::write_fill
    ( int count
    , char32_t ch
    , stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    _encoding.encode_fill(ob, count, ch, _enc_err, _allow_surr);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DYNAMIC_JOIN_PRINTER_HPP

