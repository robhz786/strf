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

namespace strf {

template<bool Active>
struct split_pos_format;

template<bool Active, typename T>
class split_pos_format_fn;

template<typename T>
class split_pos_format_fn<true, T> {
public:

    constexpr STRF_HD split_pos_format_fn() noexcept
    {
    }

    constexpr STRF_HD explicit split_pos_format_fn(std::ptrdiff_t pos) noexcept
        : _pos(pos)
    {
    }

    template <bool B, typename U>
    constexpr STRF_HD explicit split_pos_format_fn
        ( const split_pos_format_fn<B,U>& r ) noexcept
        : _pos(r.split_pos())
    {
    }

    constexpr STRF_HD T&& split_pos(std::ptrdiff_t pos) && noexcept
    {
        _pos = pos;
        return static_cast<T&&>(*this);
    }

    constexpr STRF_HD std::ptrdiff_t split_pos() const noexcept
    {
        return _pos;
    }

private:

    std::ptrdiff_t _pos = 0;
};

template<typename T>
class split_pos_format_fn<false, T>
{
    using _adapted_derived_type = strf::fmt_replace
        < T
        , strf::split_pos_format<false>
        , strf::split_pos_format<true> >;
public:

    constexpr STRF_HD split_pos_format_fn() noexcept
    {
    }

    template<typename U>
    constexpr STRF_HD explicit split_pos_format_fn(const strf::split_pos_format_fn<false, U>&) noexcept
    {
    }

    constexpr STRF_HD _adapted_derived_type split_pos(std::ptrdiff_t pos) const noexcept
    {
        return { static_cast<const T&>(*this)
               , strf::tag<strf::split_pos_format<true>> {}
               , pos};
    }

    constexpr STRF_HD std::ptrdiff_t split_pos() const noexcept
    {
        return 0;
    }
};

template<bool Active>
struct split_pos_format
{
    template<typename T>
    using fn = strf::split_pos_format_fn<Active, T>;
};

struct aligned_join_t
{
    std::int16_t width = 0;
    strf::text_alignment align = strf::text_alignment::right;
    char32_t fillchar = U' ';
    std::ptrdiff_t split_pos = 1;

    template<typename ... Args>
    constexpr STRF_HD strf::value_with_format
        < strf::detail::simple_tuple< strf::detail::opt_val_or_cref<Args>...>
                                    , strf::split_pos_format<true>
                                    , strf::alignment_format_q<true> >
    operator()(const Args& ... args) const
    {
        return { strf::detail::make_simple_tuple<Args...>(args...)
               , strf::tag< strf::split_pos_format<true>
               , strf::alignment_format_q<true> > {}
               , split_pos
               , strf::alignment_format_data {fillchar, width, align}};
    }
};

namespace detail {

template<std::size_t CharSize>
STRF_HD void print_split
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::encode_fill_func<CharSize> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr)
{
    (void) split_pos;
    encode_fill(ob, fillcount, fillchar, enc_err, allow_surr);
}

template<std::size_t CharSize, typename Printer, typename ... Printers>
STRF_HD void print_split
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::encode_fill_func<CharSize> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr
    , const Printer& p
    , const Printers& ... printers )
{
    if (split_pos > 0) {
        p.print_to(ob);
        print_split( ob, encode_fill, fillcount, fillchar, split_pos - 1
                   , enc_err, allow_surr, printers... );
    } else {
        encode_fill(ob, fillcount, fillchar, enc_err, allow_surr);
        strf::detail::write_args(ob, p, printers...);
    }
}

template<std::size_t CharSize, std::size_t ... I, typename ... Printers>
STRF_HD void print_split
    ( const strf::detail::printers_tuple_impl
        < CharSize, std::index_sequence<I...>, Printers... >& printers
    , strf::underlying_outbuf<CharSize>& ob
    , strf::encode_fill_func<CharSize> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr )
{
    strf::detail::print_split( ob, encode_fill, fillcount, fillchar, split_pos, enc_err
                             , allow_surr, printers.template get<I>()... );
}

template<std::size_t CharSize, typename ... Printers>
class aligned_join_printer_impl: public printer<CharSize>
{
    using _printers_tuple = strf::detail::printers_tuple<CharSize, Printers...>;

public:

    template<typename FPack, bool ReqSize, typename CharT, typename ... Args>
    STRF_HD aligned_join_printer_impl
        ( const FPack& fp
        , strf::print_preview<ReqSize, false>& preview
        , const strf::detail::simple_tuple<Args...>& args
        , std::ptrdiff_t split_pos
        , strf::alignment_format_data afmt
        , strf::tag<CharT> tag_char)
        : _split_pos(split_pos)
        , _afmt(afmt)
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        decltype(auto) enc = _get_facet<strf::encoding_c<CharT>>(fp);
        _encode_fill_func = enc.encode_fill;
        strf::print_preview<ReqSize, true> p { _afmt.width };
        new (_printers_ptr()) _printers_tuple { fp, p, args, tag_char };
        if (p.remaining_width() > 0) {
            _fillcount = p.remaining_width().round();
        }
        STRF_IF_CONSTEXPR (ReqSize) {
            preview.add_size(p.get_size());
            if (_fillcount > 0) {
                auto fcharsize = enc.encoded_char_size(_afmt.fill);
                preview.add_size(_fillcount * fcharsize);
            }
        }
        (void) preview;
    }

    template<typename FPack, bool ReqSize, typename CharT, typename ... Args>
    STRF_HD aligned_join_printer_impl
        ( const FPack& fp, strf::print_preview<ReqSize, true>& preview
        , const strf::detail::simple_tuple<Args...>& args
        , std::ptrdiff_t split_pos
        , strf::alignment_format_data afmt
        , strf::tag<CharT> )
        : _split_pos(split_pos)
        , _afmt(afmt)
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        decltype(auto) enc = _get_facet<strf::encoding_c<CharT>>(fp);
        _encode_fill_func = enc.encode_fill;
        if (_afmt.width < 0) {
            _afmt.width = 0;
        }
        strf::width_t wmax = _afmt.width;
        strf::width_t diff = 0;
        if (preview.remaining_width() > _afmt.width) {
            wmax = preview.remaining_width();
            diff = preview.remaining_width() - _afmt.width;
        }
        strf::print_preview<ReqSize, true> p{wmax};
        // todo: what if the line below throws ?
        new (_printers_ptr()) _printers_tuple{fp, p, args, strf::tag<CharT>()};
        if (p.remaining_width() > diff) {
            _fillcount = (p.remaining_width() - diff).round();
        }
        width_t width = _fillcount + wmax - p.remaining_width();
        preview.subtract_width(width);
        STRF_IF_CONSTEXPR (ReqSize) {
            preview.add_size(p.get_size());
            if (_fillcount > 0) {
                auto fcharsize = enc.encoded_char_size(_afmt.fill);
                preview.add_size( _fillcount * fcharsize);
            }
        }
    }

    STRF_HD ~aligned_join_printer_impl()
    {
        _printers_ptr()->~_printers_tuple();
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override
    {
        switch (_afmt.alignment) {
            case strf::text_alignment::left: {
                strf::detail::write(ob, _printers());
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::right: {
                _write_fill(ob, _fillcount);
                strf::detail::write(ob, _printers());
                break;
            }
            case strf::text_alignment::split: {
                _print_split(ob);
                break;
            }
            default: {
                STRF_ASSERT(_afmt.alignment == strf::text_alignment::center);
                auto half_fillcount = _fillcount >> 1;
                _write_fill(ob, half_fillcount);
                strf::detail::write(ob, _printers());
                _write_fill(ob, _fillcount - half_fillcount);
                break;
            }
        }
    }

private:

    using _printers_tuple_storage = typename std::aligned_storage_t
#if defined(_MSC_VER)
    <sizeof(std::tuple<Printers...>), alignof(strf::printer<CharSize>)>;
#else
    <sizeof(_printers_tuple), alignof(_printers_tuple)>;
#endif
    _printers_tuple_storage _pool;
    std::ptrdiff_t _split_pos;
    strf::alignment_format_data _afmt;
    strf::encode_fill_func<CharSize> _encode_fill_func;
    strf::width_t _width;
    std::int16_t _fillcount = 0;
    strf::encoding_error _enc_err;
    strf::surrogate_policy _allow_surr;

    STRF_HD _printers_tuple * _printers_ptr()
    {
        return reinterpret_cast<_printers_tuple*>(&_pool);
    }
    STRF_HD const _printers_tuple& _printers() const
    {
        return *reinterpret_cast<const _printers_tuple*>(&_pool);
    }

    template <typename Category, typename FPack>
    static decltype(auto) STRF_HD _get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, strf::aligned_join_t>();
    }

    STRF_HD void _write_fill(strf::underlying_outbuf<CharSize>& ob, int count) const
    {
        _encode_fill_func(ob, count, _afmt.fill, _enc_err, _allow_surr);
    }

    STRF_HD void _print_split(strf::underlying_outbuf<CharSize>& ob) const;
};

template<std::size_t CharSize, typename ... Printers>
STRF_HD void aligned_join_printer_impl<CharSize, Printers...>::_print_split
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    strf::detail::print_split( _printers(), ob, _encode_fill_func, _fillcount
                             , _afmt.fill, _split_pos, _enc_err, _allow_surr );
}

template<typename CharT, typename FPack, typename Preview, typename ... Args>
using aligned_join_printer_impl_of
= aligned_join_printer_impl
    < sizeof(CharT)
    , decltype
        ( make_printer<CharT>
            ( strf::rank<5>()
            , std::declval<const FPack&>()
            , std::declval<strf::print_preview<Preview::size_required, true>&>()
            , std::declval<const Args&>() )) ... >;

template<typename CharT, typename FPack, typename Preview, typename ... Args>
class aligned_join_printer
    : public strf::detail::aligned_join_printer_impl_of<CharT, FPack, Preview, Args...>
{
    using _aligned_join_impl = strf::detail::aligned_join_printer_impl_of
        <CharT, FPack, Preview, Args...>;

public:

    STRF_HD aligned_join_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::detail::simple_tuple<Args...>& args
        , std::ptrdiff_t split_pos
        , strf::alignment_format_data afmt )
        : _aligned_join_impl( fp, preview, args, split_pos, afmt
                            , strf::tag<CharT>() )
    {
    }

    virtual STRF_HD ~aligned_join_printer()
    {
    }
};

template<std::size_t CharSize, typename ... Printers>
class join_printer_impl: public printer<CharSize> {
public:

    template<typename FPack, typename Preview, typename CharT, typename ... Args>
    STRF_HD join_printer_impl
        ( const FPack& fp
        , Preview& preview
        , const strf::detail::simple_tuple<Args...>& args
        , strf::tag<CharT> tag_char )
        : _printers{fp, preview, args, tag_char}
    {
    }

    STRF_HD ~join_printer_impl()
    {
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override
    {
        strf::detail::write(ob, _printers);
    }

private:

    strf::detail::printers_tuple<CharSize, Printers...> _printers;
};

template<typename CharT, typename FPack, typename Preview, typename ... Args>
class join_printer
    : public strf::detail::join_printer_impl
        < sizeof(CharT)
        , decltype(make_printer<CharT>( strf::rank<5>()
                                      , std::declval<const FPack&>()
                                      , std::declval<Preview&>()
                                      , std::declval<const Args&>() )) ... >
{
    using _join_impl = strf::detail::join_printer_impl
        < sizeof(CharT)
        , decltype(make_printer<CharT>( strf::rank<5>()
                                      , std::declval<const FPack&>()
                                      , std::declval<Preview&>()
                                      , std::declval<const Args&>() )) ... >;
public:

    STRF_HD join_printer( const FPack& fp
                        , Preview& preview
                        , const strf::detail::simple_tuple<Args...>& args )
        : _join_impl(fp, preview, args, strf::tag<CharT>())
    {
    }

    virtual STRF_HD ~join_printer()
    {
    }
};

} // namespace detail

template< typename CharT
        , typename FPack
        , typename Preview
        , bool SplitPosActive
        , typename ... Args >
inline STRF_HD strf::detail::join_printer<CharT, FPack, Preview, Args...> make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , const strf::value_with_format< strf::detail::simple_tuple<Args...>
                                   , strf::split_pos_format<SplitPosActive>
                                   , strf::alignment_format_q<false> > input )
{
    return {fp, preview, input.value()};
}

template<typename ... Args>
constexpr STRF_HD strf::value_with_format
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

template< typename CharT
        , typename FPack
        , typename Preview
        , bool SplitPosActive
        , typename ... Args >
inline STRF_HD strf::detail::aligned_join_printer<CharT, FPack, Preview, Args...> make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , const strf::value_with_format
        < strf::detail::simple_tuple<Args...>
        , strf::split_pos_format<SplitPosActive>
        , strf::alignment_format_q<true> > input )
{
    return { fp, preview, input.value(), input.split_pos()
           , input.get_alignment_format_data() };
}

constexpr STRF_HD strf::aligned_join_t join_align
    ( std::int16_t width
    , strf::text_alignment align
    , char32_t fillchar = U' '
    , int split_pos = 0 )
{
    return {width, align, fillchar, split_pos};
}

constexpr STRF_HD strf::aligned_join_t join_center(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::center, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_t join_left(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::left, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_t join_right(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::right, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_t join_split(std::int16_t width, char32_t fillchar,
    std::ptrdiff_t split_pos) noexcept
{
    return {width, strf::text_alignment::split, fillchar, split_pos};
}

constexpr STRF_HD strf::aligned_join_t join_split(std::int16_t width, std::ptrdiff_t split_pos) noexcept
{
    return {width, strf::text_alignment::split, U' ', split_pos};
}

} // namespace strf

#endif  // STRF_JOIN_HPP

