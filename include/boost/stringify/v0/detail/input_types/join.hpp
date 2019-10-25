#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/stringify/v0/detail/printers_tuple.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename ... Args>
struct join_t
{
    stringify::v0::detail::simple_tuple<Args...> args;
};

struct aligned_join_t;

template <typename ... Args>
struct aligned_joined_args
{
    const stringify::v0::aligned_join_t& join;
    stringify::v0::detail::simple_tuple<Args...> args;
};


struct aligned_join_t
{
    int width = 0;
    stringify::v0::text_alignment align = stringify::v0::text_alignment::right;
    char32_t fillchar = U' ';
    int num_leading_args = 1;

    template <typename ... Args>
    constexpr stringify::v0::aligned_joined_args
        < stringify::v0::detail::opt_val_or_cref<Args>... >
    operator()(const Args& ... args) const
    {
        return {*this, stringify::v0::detail::make_simple_tuple(args ...)};
    }
};

namespace detail {

template <typename CharT>
void print_split
    ( stringify::v0::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , int fillcount
    , char32_t fillchar
    , int split_pos
    , stringify::v0::encoding_error enc_err
    , stringify::v0::surrogate_policy allow_surr)
{
    (void) split_pos;
    enc.encode_fill(ob, fillcount, fillchar, enc_err, allow_surr);
}

template <typename CharT, typename Printer, typename ... Printers>
void print_split
    ( stringify::v0::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , int fillcount
    , char32_t fillchar
    , int split_pos
    , stringify::v0::encoding_error enc_err
    , stringify::v0::surrogate_policy allow_surr
    , const Printer& p
    , const Printers& ... printers )
{
    if (split_pos > 0)
    {
        p.print_to(ob);
        print_split( ob, enc, fillcount, fillchar, split_pos - 1
                   , enc_err, allow_surr, printers... );
    }
    else
    {
        enc.encode_fill(ob, fillcount, fillchar, enc_err, allow_surr);
        stringify::v0::detail::write_args(ob, p, printers...);
    }
}


template< typename CharT, std::size_t ... I, typename ... Printers >
void print_split
    ( const stringify::v0::detail::printers_tuple_impl
        < CharT, std::index_sequence<I...>, Printers... >& printers
    , stringify::v0::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , int fillcount
    , char32_t fillchar
    , int split_pos
    , stringify::v0::encoding_error enc_err
    , stringify::v0::surrogate_policy allow_surr )
{
    stringify::v0::detail::print_split
        ( ob, enc, fillcount, fillchar, split_pos, enc_err, allow_surr
        , printers.template get<I>()... );
}


template <typename CharT, typename ... Printers>
class aligned_join_printer_impl: public printer<CharT>
{
public:

    template <typename FPack, typename ... Args>
    aligned_join_printer_impl
        ( const FPack& fp
        , const stringify::v0::aligned_joined_args<Args...>& ja )
        : _printers{fp, ja.args}
        , _fmt(ja.join)
        , _encoding(_get_facet<stringify::v0::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<stringify::v0::encoding_error_c>(fp))
        , _allow_surr(_get_facet<stringify::v0::surrogate_policy_c>(fp))
    {
        auto w = stringify::v0::detail::width(_printers, _fmt.width);
        _fillcount = ( _fmt.width > w
                     ? _fmt.width - w
                     : 0 );
    }

    aligned_join_printer_impl( const aligned_join_printer_impl& cp ) = default;

    ~aligned_join_printer_impl()
    {
    }

    std::size_t necessary_size() const override
    {
        return stringify::v0::detail::necessary_size(_printers)
            + _fill_length();
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override
    {
        if (_fillcount <= 0)
        {
            return stringify::v0::detail::write(ob, _printers);;
        }
        else
        {
            switch(_fmt.align)
            {
                case stringify::v0::text_alignment::left:
                {
                    stringify::v0::detail::write(ob, _printers);;
                    _write_fill(ob, _fillcount);
                    break;
                }
                case stringify::v0::text_alignment::right:
                {
                    _write_fill(ob, _fillcount);
                    stringify::v0::detail::write(ob, _printers);
                    break;
                }
                case stringify::v0::text_alignment::split:
                {
                    _print_split(ob);
                    break;
                }
                default:
                {
                    BOOST_ASSERT(_fmt.align == stringify::v0::text_alignment::center);
                    auto half_fillcount = _fillcount / 2;
                    _write_fill(ob, half_fillcount);
                    stringify::v0::detail::write(ob, _printers);;
                    _write_fill(ob, _fillcount - half_fillcount);
                    break;
                }
            }
        }
    }

    int width(int limit) const override
    {
        if (_fillcount > 0)
        {
            return _fmt.width;
        }
        return stringify::v0::detail::width(_printers, limit);;
    }

private:

    stringify::v0::detail::printers_tuple<CharT, Printers...> _printers;
    stringify::v0::aligned_join_t _fmt;
    const stringify::v0::encoding<CharT> _encoding;
    int _fillcount = 0;
    stringify::v0::encoding_error _enc_err;
    stringify::v0::surrogate_policy _allow_surr;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, stringify::v0::aligned_join_t>();
    }

    std::size_t _fill_length() const
    {
        if(_fillcount > 0)
        {
            return _fillcount * _encoding.char_size( _fmt.fillchar
                                                   , _enc_err);
        }
        return 0;
    }

    void _write_fill(stringify::v0::basic_outbuf<CharT>& ob, int count) const
    {
        _encoding.encode_fill( ob, count, _fmt.fillchar
                             , _enc_err, _allow_surr );
    }

    void _print_split(stringify::v0::basic_outbuf<CharT>& ob) const;
};

template <typename CharT, typename ... Printers>
void aligned_join_printer_impl<CharT, Printers...>::_print_split
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    stringify::v0::detail::print_split
        ( _printers, ob, _encoding, _fillcount, _fmt.fillchar
        , _fmt.num_leading_args, _enc_err, _allow_surr );
}

template <typename CharT, typename FPack, typename ... Args>
using aligned_join_printer_impl_of
= aligned_join_printer_impl
    < CharT
    , decltype(make_printer<CharT>( std::declval<const FPack&>()
                                  , std::declval<const Args&>() )) ... >;

template <typename CharT, typename FPack, typename ... Args>
class aligned_join_printer
    : public stringify::v0::detail::aligned_join_printer_impl_of
        <CharT, FPack, Args...>
{
    using _aligned_join_impl
    = stringify::v0::detail::aligned_join_printer_impl_of
        <CharT, FPack, Args...>;

public:

    aligned_join_printer
        ( const FPack& fp
        , const stringify::v0::aligned_joined_args<Args...>& ja )
        : _aligned_join_impl(fp, ja)
    {
    }

    aligned_join_printer(const aligned_join_printer&) = default;

    virtual ~aligned_join_printer()
    {
    }
};


template <typename CharT, typename ... Printers>
class join_printer_impl: public printer<CharT>
{
public:

    template <typename FPack, typename ... Args>
    join_printer_impl(const FPack& fp, const stringify::v0::join_t<Args...>& args)
        : _printers{fp, args.args}
    {
    }

    ~join_printer_impl()
    {
    }

    std::size_t necessary_size() const override
    {
        return stringify::v0::detail::necessary_size(_printers);
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override
    {
        stringify::v0::detail::write(ob, _printers);
    }

    int width(int limit) const override
    {
        return stringify::v0::detail::width(_printers, limit);
    }

private:

    stringify::v0::detail::printers_tuple<CharT, Printers...> _printers;
};

template <typename CharT, typename FPack, typename ... Args>
class join_printer
    : public stringify::v0::detail::join_printer_impl
        < CharT
        , decltype(make_printer<CharT>( std::declval<const FPack&>()
                                      , std::declval<const Args&>() )) ... >
{
    using _join_impl
    = stringify::v0::detail::join_printer_impl
        < CharT
        , decltype(make_printer<CharT>( std::declval<const FPack&>()
                                      , std::declval<const Args&>() )) ... >;
public:

    join_printer(const FPack& fp, const stringify::v0::join_t<Args...>& j)
        : _join_impl(fp, j)
    {
    }

    join_printer(const join_printer& cp) = default;
    virtual ~join_printer()
    {
    }
};

} // namespace detail

template <typename CharT, typename FPack, typename... Args>
inline stringify::v0::detail::join_printer<CharT, FPack, Args...>
make_printer( const FPack& fp
            , const stringify::v0::join_t<Args...>& args )
{
    return {fp, args};
}

template <typename ... Args>
stringify::v0::join_t< stringify::v0::detail::opt_val_or_cref<Args>... >
join(const Args& ... args)
{
    return {stringify::v0::detail::make_simple_tuple(args...)};
}

template <typename CharT, typename FPack, typename ... Args>
inline stringify::v0::detail::aligned_join_printer<CharT, FPack, Args...>
make_printer
    ( const FPack& fp
    , const stringify::v0::aligned_joined_args<Args...>& x )
{
    return {fp, x};
}

constexpr stringify::v0::aligned_join_t
join_align( int width
          , stringify::v0::text_alignment align
          , char32_t fillchar = U' '
          , int num_leading_args = 1 )
{
    return {width, align, fillchar, num_leading_args};
}

constexpr stringify::v0::aligned_join_t
join_center(int width, char32_t fillchar = U' ') noexcept
{
    return {width, stringify::v0::text_alignment::center, fillchar, 0};
}

constexpr stringify::v0::aligned_join_t
join_left(int width, char32_t fillchar = U' ') noexcept
{
    return {width, stringify::v0::text_alignment::left, fillchar, 0};
}


constexpr stringify::v0::aligned_join_t
join_right(int width, char32_t fillchar = U' ') noexcept
{
    return {width, stringify::v0::text_alignment::right, fillchar, 0};
}

constexpr stringify::v0::aligned_join_t
join_split(int width, char32_t fillchar, int num_leading_args) noexcept
{
    return {width, stringify::v0::text_alignment::split, fillchar, num_leading_args};
}

constexpr stringify::v0::aligned_join_t
join_split(int width, int num_leading_args) noexcept
{
    return {width, stringify::v0::text_alignment::split, U' ', num_leading_args};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_JOIN_HPP

