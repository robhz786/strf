#ifndef STRF_JOIN_HPP
#define STRF_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/encoding.hpp>
#include <strf/detail/printers_tuple.hpp>
#include <strf/detail/format_functions.hpp>

#if defined(_MSC_VER)
#include <tuple>
#endif

STRF_NAMESPACE_BEGIN

template <bool Active>
struct split_pos_format;

template <bool Active, typename T>
class split_pos_format_fn;

template <typename T>
class split_pos_format_fn<true, T>
{
public:

    constexpr split_pos_format_fn() noexcept = default;
    constexpr split_pos_format_fn(const split_pos_format_fn&) noexcept = default;
    constexpr explicit split_pos_format_fn(std::ptrdiff_t pos) noexcept
        : _pos(pos)
    {
    }

    template <bool B, typename U>
    constexpr explicit split_pos_format_fn
        ( const split_pos_format_fn<B,U>& r ) noexcept
        : _pos(r.split_pos())
    {
    }

    constexpr T&& split_pos(std::ptrdiff_t pos) && noexcept
    {
        _pos = pos;
        return static_cast<T&&>(*this);
    }

    constexpr std::ptrdiff_t split_pos() const noexcept
    {
        return _pos;
    }

private:

    std::ptrdiff_t _pos = 0;
};

template <typename T>
class split_pos_format_fn<false, T>
{
    using _adapted_derived_type = strf::fmt_replace
            < T
            , strf::split_pos_format<false>
            , strf::split_pos_format<true> >;
public:

    constexpr split_pos_format_fn() noexcept = default;
    constexpr split_pos_format_fn(const split_pos_format_fn&) noexcept = default;

    template <typename U>
    constexpr explicit split_pos_format_fn
        ( const strf::split_pos_format_fn<false,U>& ) noexcept
    {
    }

    constexpr _adapted_derived_type split_pos(std::ptrdiff_t pos) const noexcept
    {
        return { static_cast<const T&>(*this)
               , strf::tag<strf::split_pos_format<true>>{}
               , pos };
    }

    constexpr std::ptrdiff_t split_pos() const noexcept
    {
        return 0;
    }
};

template <bool Active>
struct split_pos_format
{
    template <typename T>
    using fn = strf::split_pos_format_fn<Active, T>;
};

struct aligned_join_t
{
    std::int16_t width = 0;
    strf::text_alignment align = strf::text_alignment::right;
    char32_t fillchar = U' ';
    std::ptrdiff_t split_pos = 1;

    template <typename ... Args>
    constexpr strf::value_with_format
        < strf::detail::simple_tuple<strf::detail::opt_val_or_cref<Args>... >
        , strf::split_pos_format<true>
        , strf::alignment_format_q<true> >
    operator()(const Args& ... args) const
    {
        return { strf::detail::make_simple_tuple<Args...>(args...)
               , strf::tag< strf::split_pos_format<true>
                               , strf::alignment_format_q<true> >{}
               , split_pos
               , strf::alignment_format_data{fillchar, width, align} };
    }
};

namespace detail {

template <typename CharT>
void print_split
    ( strf::basic_outbuf<CharT>& ob
    , strf::encoding<CharT> enc
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr)
{
    (void) split_pos;
    enc.encode_fill(ob, fillcount, fillchar, enc_err, allow_surr);
}

template <typename CharT, typename Printer, typename ... Printers>
void print_split
    ( strf::basic_outbuf<CharT>& ob
    , strf::encoding<CharT> enc
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr
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
        strf::detail::write_args(ob, p, printers...);
    }
}


template< typename CharT, std::size_t ... I, typename ... Printers >
void print_split
    ( const strf::detail::printers_tuple_impl
        < CharT, std::index_sequence<I...>, Printers... >& printers
    , strf::basic_outbuf<CharT>& ob
    , strf::encoding<CharT> enc
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr )
{
    strf::detail::print_split
        ( ob, enc, fillcount, fillchar, split_pos, enc_err, allow_surr
        , printers.template get<I>()... );
}


template <typename CharT, typename ... Printers>
class aligned_join_printer_impl: public printer<CharT>
{
    using _printers_tuple = strf::detail::printers_tuple<CharT, Printers...>;

public:

    template <typename FPack, bool ReqSize, typename ... Args>
    aligned_join_printer_impl
        ( const FPack& fp
        , strf::print_preview<ReqSize, false>& preview
        , const strf::detail::simple_tuple<Args...>& args
        , std::ptrdiff_t split_pos
        , strf::alignment_format_data afmt )
        : _split_pos(split_pos)
        , _afmt(afmt)
        , _encoding(_get_facet<strf::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        strf::print_preview<ReqSize, true> p{_afmt.width};
        new (_printers_ptr()) _printers_tuple{fp, p, args};
        if (p.remaining_width() > 0)
        {
            _fillcount = p.remaining_width().round();
        }
        STRF_IF_CONSTEXPR (ReqSize)
        {
            preview.add_size(p.get_size());
            if (_fillcount > 0)
            {
                auto fcharsize = _encoding.char_size(_afmt.fill);
                preview.add_size(_fillcount * fcharsize);
            }
        }
        (void)preview;
    }

    template <typename FPack, bool ReqSize, typename ... Args>
    aligned_join_printer_impl
        ( const FPack& fp
        , strf::print_preview<ReqSize, true>& preview
        , const strf::detail::simple_tuple<Args...>& args
        , std::ptrdiff_t split_pos
        , strf::alignment_format_data afmt )
        : _split_pos(split_pos)
        , _afmt(afmt)
        , _encoding(_get_facet<strf::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        if (_afmt.width < 0)
        {
            _afmt.width = 0;
        }
        strf::width_t wmax = _afmt.width;
        strf::width_t diff = 0;
        if (preview.remaining_width() > _afmt.width)
        {
            wmax = preview.remaining_width();
            diff = preview.remaining_width() - _afmt.width;
        }
        strf::print_preview<ReqSize, true> p{wmax};
        new (_printers_ptr()) _printers_tuple{fp, p, args}; // todo: what if this throws ?
        if (p.remaining_width() > diff)
        {
           _fillcount = (p.remaining_width() - diff).round();
        }
        width_t width = _fillcount + wmax - p.remaining_width();
        preview.subtract_width(width);
        STRF_IF_CONSTEXPR (ReqSize)
        {
            preview.add_size(p.get_size());
            if (_fillcount > 0)
            {
                preview.add_size
                    ( _fillcount
                    * _encoding.char_size(_afmt.fill) );
            }
        }
    }

    ~aligned_join_printer_impl()
    {
        _printers_ptr()->~_printers_tuple();
    }

    void print_to(strf::basic_outbuf<CharT>& ob) const override
    {
        switch(_afmt.alignment)
        {
            case strf::text_alignment::left:
            {
                strf::detail::write(ob, _printers());
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::right:
            {
                _write_fill(ob, _fillcount);
                strf::detail::write(ob, _printers());
                break;
            }
            case strf::text_alignment::split:
            {
                _print_split(ob);
                break;
            }
            default:
            {
                STRF_ASSERT(_afmt.alignment == strf::text_alignment::center);
                auto half_fillcount = _fillcount >> 1;
                _write_fill(ob, half_fillcount);
                strf::detail::write(ob, _printers());;
                _write_fill(ob, _fillcount - half_fillcount);
                break;
            }
        }
    }

private:

    using _printers_tuple_storage = typename std::aligned_storage_t
#if defined(_MSC_VER)
        <sizeof(std::tuple<Printers...>), alignof(strf::printer<CharT>)>;
#else
        <sizeof(_printers_tuple), alignof(_printers_tuple)>;
#endif
    _printers_tuple_storage _pool;
    std::ptrdiff_t _split_pos;
    strf::alignment_format_data _afmt;
    const strf::encoding<CharT> _encoding;
    strf::width_t _width;
    std::int16_t _fillcount = 0;
    strf::encoding_error _enc_err;
    strf::surrogate_policy _allow_surr;

    _printers_tuple * _printers_ptr()
    {
        return reinterpret_cast<_printers_tuple*>(&_pool);
    }
    const _printers_tuple& _printers() const
    {
        return *reinterpret_cast<const _printers_tuple*>(&_pool);
    }

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, strf::aligned_join_t>();
    }

    std::size_t _fill_length() const
    {
        if(_fillcount > 0)
        {
            return _fillcount * _encoding.char_size(_afmt.fill);
        }
        return 0;
    }

    void _write_fill(strf::basic_outbuf<CharT>& ob, int count) const
    {
        _encoding.encode_fill( ob, count, _afmt.fill
                             , _enc_err, _allow_surr );
    }

    void _print_split(strf::basic_outbuf<CharT>& ob) const;
};

template <typename CharT, typename ... Printers>
void aligned_join_printer_impl<CharT, Printers...>::_print_split
    ( strf::basic_outbuf<CharT>& ob ) const
{
    strf::detail::print_split
        ( _printers(), ob, _encoding, _fillcount, _afmt.fill
        , _split_pos, _enc_err, _allow_surr );
}

template <typename CharT, typename FPack, typename Preview, typename ... Args>
using aligned_join_printer_impl_of
= aligned_join_printer_impl
    < CharT
    , decltype(make_printer<CharT>
                  ( std::declval<const FPack&>()
                  , std::declval<strf::print_preview<Preview::size_required, true>&>()
                  , std::declval<const Args&>() )) ... >;

template <typename CharT, typename FPack, typename Preview, typename ... Args>
class aligned_join_printer
    : public strf::detail::aligned_join_printer_impl_of
        <CharT, FPack, Preview, Args...>
{
    using _aligned_join_impl
    = strf::detail::aligned_join_printer_impl_of
        <CharT, FPack, Preview, Args...>;

public:

    aligned_join_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::detail::simple_tuple<Args...>& args
        , std::ptrdiff_t split_pos
        , strf::alignment_format_data afmt )
        : _aligned_join_impl(fp, preview, args, split_pos, afmt)
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

    template <typename FPack, typename Preview, typename ... Args>
    join_printer_impl( const FPack& fp
                     , Preview& preview
                     , const strf::detail::simple_tuple<Args...>& args)
        : _printers{fp, preview, args}
    {
    }

    ~join_printer_impl()
    {
    }

    void print_to(strf::basic_outbuf<CharT>& ob) const override
    {
        strf::detail::write(ob, _printers);
    }

private:

    strf::detail::printers_tuple<CharT, Printers...> _printers;
};

template <typename CharT, typename FPack, typename Preview, typename ... Args>
class join_printer
    : public strf::detail::join_printer_impl
        < CharT
        , decltype(make_printer<CharT>( std::declval<const FPack&>()
                                      , std::declval<Preview&>()
                                      , std::declval<const Args&>() )) ... >
{
    using _join_impl
    = strf::detail::join_printer_impl
        < CharT
        , decltype(make_printer<CharT>( std::declval<const FPack&>()
                                      , std::declval<Preview&>()
                                      , std::declval<const Args&>() )) ... >;
public:

    join_printer( const FPack& fp
                , Preview& preview
                , const strf::detail::simple_tuple<Args...>& args )
        : _join_impl(fp, preview, args)
    {
    }

    join_printer(const join_printer& cp) = default;
    virtual ~join_printer()
    {
    }
};

} // namespace detail

template < typename CharT
         , typename FPack
         , typename Preview
         , bool SplitPosActive
         , typename... Args >
inline strf::detail::join_printer<CharT, FPack, Preview, Args...>
make_printer( const FPack& fp
            , Preview& preview
            , const strf::value_with_format
                < strf::detail::simple_tuple<Args...>
                , strf::split_pos_format<SplitPosActive>
                , strf::alignment_format_q<false> > input )
{
    return { fp, preview, input.value() };
}

template <typename ... Args>
constexpr strf::value_with_format
    < strf::detail::simple_tuple<strf::detail::opt_val_or_cref<Args>...>
    , strf::split_pos_format<false>
    , strf::alignment_format_q<false> >
join(const Args& ... args)
{
    return strf::value_with_format
        < strf::detail::simple_tuple<strf::detail::opt_val_or_cref<Args>...>
        , strf::split_pos_format<false>
        , strf::alignment_format_q<false> >
        { strf::detail::make_simple_tuple(args...) };
}

template < typename CharT
         , typename FPack
         , typename Preview
         , bool SplitPosActive
         , typename... Args >
inline strf::detail::aligned_join_printer<CharT, FPack, Preview, Args...>
make_printer( const FPack& fp
            , Preview& preview
            , const strf::value_with_format
                < strf::detail::simple_tuple<Args...>
                , strf::split_pos_format<SplitPosActive>
                , strf::alignment_format_q<true> > input )
{
    return { fp, preview, input.value(), input.split_pos()
           , input.get_alignment_format_data() };
}

constexpr strf::aligned_join_t join_align( std::int16_t width
                                         , strf::text_alignment align
                                         , char32_t fillchar = U' '
                                         , int split_pos = 0 )
{
    return {width, align, fillchar, split_pos};
}

constexpr strf::aligned_join_t
join_center(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::center, fillchar, 0};
}

constexpr strf::aligned_join_t
join_left(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::left, fillchar, 0};
}

constexpr strf::aligned_join_t
join_right(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::right, fillchar, 0};
}

constexpr strf::aligned_join_t
join_split( std::int16_t width
          , char32_t fillchar
          , std::ptrdiff_t split_pos ) noexcept
{
    return {width, strf::text_alignment::split, fillchar, split_pos};
}

constexpr strf::aligned_join_t
join_split(std::int16_t width, std::ptrdiff_t split_pos) noexcept
{
    return {width, strf::text_alignment::split, U' ', split_pos};
}

STRF_NAMESPACE_END

#endif  // STRF_JOIN_HPP

