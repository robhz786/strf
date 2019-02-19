#ifndef BOOST_STRINGIFY_V0_DYNAMIC_JOIN_PRINTER_HPP
#define BOOST_STRINGIFY_V0_DYNAMIC_JOIN_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/detail/format_functions.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharOut>
class printers_receiver
{
public:

    virtual ~printers_receiver()
    {
    }

    virtual bool put(const stringify::v0::printer<CharOut>& ) = 0;
};

namespace detail {

template <typename CharOut>
class width_subtracter: public printers_receiver<CharOut>
{
public:

    width_subtracter(int w)
        : _width(w)
    {
    }

    bool put(const stringify::v0::printer<CharOut>& p) override;

    int remaining_width() const
    {
        return _width;
    }

private:

    int _width;
};

template <typename CharOut>
bool width_subtracter<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    _width = p.remaining_width(_width);
    return _width > 0;
}

template <typename CharOut>
class necessary_size_sum: public printers_receiver<CharOut>
{
public:

    necessary_size_sum() = default;

    bool put(const stringify::v0::printer<CharOut>& p) override;

    std::size_t accumulated_length() const
    {
        return _len;
    }

private:

    std::size_t _len = 0;
};

template <typename CharOut>
bool necessary_size_sum<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    _len += p.necessary_size();
    return true;
}

template <typename CharOut>
class serial_writer: public printers_receiver<CharOut>
{
public:

    serial_writer(stringify::v0::output_buffer<CharOut>& ob) noexcept
        : _ob(ob)
    {
    }

    bool put(const stringify::v0::printer<CharOut>& p) override;

private:

    stringify::v0::output_buffer<CharOut>& _ob;
};

template <typename CharOut>
bool serial_writer<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    return p.write(_ob);
}

} // namespace detail

template <typename CharOut>
class dynamic_join_printer: public stringify::v0::printer<CharOut>
{
public:

    dynamic_join_printer
        ( stringify::v0::encoding<CharOut> enc
        , stringify::v0::encoding_policy epoli )
        : _encoding(enc)
        , _epoli(epoli)
    {
    }

    std::size_t necessary_size() const override;

    bool write(stringify::v0::output_buffer<CharOut>& ob) const override;

    int remaining_width(int w) const override;

protected:

    virtual stringify::v0::alignment_format::fn<void> formatting() const;

    virtual bool compose(stringify::v0::printers_receiver<CharOut>& out) const = 0;

private:

    bool write_with_fill
        ( int fillcount
        , stringify::v0::output_buffer<CharOut>& ob ) const;

    bool write_without_fill
        ( stringify::v0::output_buffer<CharOut>& ob ) const;

    bool write_fill
        ( int count
        , char32_t ch
        , stringify::v0::output_buffer<CharOut>& ob ) const;

    stringify::v0::encoding<CharOut> _encoding;
    stringify::v0::encoding_policy _epoli;
};

template <typename CharOut>
stringify::v0::alignment_format::fn<void>
dynamic_join_printer<CharOut>::formatting() const
{
    return {};
}

template <typename CharOut>
std::size_t dynamic_join_printer<CharOut>::necessary_size() const
{
    std::size_t fill_len = 0;
    const auto fmt = formatting();
    if(fmt.width() > 0)
    {
        stringify::v0::detail::width_subtracter<CharOut> wds{fmt.width()};
        compose(wds);
        std::size_t fillcount = wds.remaining_width();
        fill_len = _encoding.char_size(fmt.fill(), _epoli.err_hdl()) * fillcount;
    }

    stringify::v0::detail::necessary_size_sum<CharOut> s;
    compose(s);
    return s.accumulated_length() + fill_len;
}

template <typename CharOut>
int dynamic_join_printer<CharOut>::remaining_width(int w) const
{
    const auto fmt_width = formatting().width();
    if (fmt_width > w)
    {
        return 0;
    }

    stringify::v0::detail::width_subtracter<CharOut> s{w};
    compose(s);
    int rw = s.remaining_width();
    return (w - rw < fmt_width) ? (w - fmt_width) : rw;
}

template <typename CharOut>
bool dynamic_join_printer<CharOut>::write
    ( stringify::v0::output_buffer<CharOut>& ob ) const
{
    auto fmt = formatting();
    auto fillcount = fmt.width();
    if(fillcount > 0)
    {
        stringify::v0::detail::width_subtracter<CharOut> wds{fillcount};
        compose(wds);
        fillcount = wds.remaining_width();
    }
    if(fillcount > 0)
    {
        return write_with_fill(fillcount, ob);
    }
    else
    {
        return write_without_fill(ob);
    }
}

template <typename CharOut>
bool dynamic_join_printer<CharOut>::write_without_fill
    ( stringify::v0::output_buffer<CharOut>& ob ) const
{
    stringify::v0::detail::serial_writer<CharOut> s(ob);
    return compose(s);
}

template <typename CharOut>
bool dynamic_join_printer<CharOut>::write_with_fill
    ( int fillcount
    , stringify::v0::output_buffer<CharOut>& ob ) const
{
    auto fmt = formatting();
    char32_t fill_char = fmt.fill();
    switch (fmt.alignment())
    {
        case stringify::v0::alignment::left:
        {
            return write_without_fill(ob)
                && write_fill(fillcount, fill_char, ob);
        }
        case stringify::v0::alignment::center:
        {
            auto halfcount = fillcount >> 1;
            return write_fill(halfcount, fill_char, ob)
                && write_without_fill(ob)
                && write_fill(fillcount - halfcount, fill_char, ob);
        }
        //case stringify::v0::alignment::internal:
        //case stringify::v0::alignment::right:
        default:
        {
            return write_fill(fillcount, fill_char, ob)
                && write_without_fill(ob);
        }
    }
}

template <typename CharOut>
bool dynamic_join_printer<CharOut>::write_fill
    ( int count
    , char32_t ch
    , stringify::v0::output_buffer<CharOut>& ob ) const
{
    return stringify::v0::detail::write_fill
        ( _encoding, ob, count, ch, _epoli );
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DYNAMIC_JOIN_PRINTER_HPP

