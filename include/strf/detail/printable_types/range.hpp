#ifndef STRF_DETAIL_INPUT_TYPES_RANGE_HPP
#define STRF_DETAIL_INPUT_TYPES_RANGE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <strf/detail/facets/char_encoding.hpp>

namespace strf {

template <typename It>
struct range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
};

template <typename It, typename CharIn>
struct separated_range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
    const CharIn* sep_begin;
    std::size_t sep_len;
};

template <typename It, typename UnaryOp>
struct transformed_range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
    UnaryOp op;
};

template <typename It, typename CharIn, typename UnaryOp>
struct separated_transformed_range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
    const CharIn* sep_begin;
    std::size_t sep_len;
    UnaryOp op;
};

namespace detail {

template <typename List, typename T>
struct mp_replace_front_impl;

template < typename NewFirst
         , template <typename...> class List
         , typename First
         , typename ... Rest >
struct mp_replace_front_impl<List<First, Rest...>, NewFirst>
{
    using type = List<NewFirst, Rest...>;
};

template <typename T, typename List>
using mp_replace_front
    = typename strf::detail::mp_replace_front_impl<T, List>::type;

} // namespace detail

template < typename Iterator
         , typename V  = strf::detail::iterator_value_type<Iterator>
         , typename VF = strf::fmt_type<V> >
using range_with_format
    = strf::detail::mp_replace_front
        < VF, strf::print_traits<strf::range_p<Iterator>> >;

template < typename Iterator
         , typename CharT
         , typename V  = strf::detail::iterator_value_type<Iterator>
         , typename VF = strf::fmt_type<V> >
using sep_range_with_format
    = strf::detail::mp_replace_front
        < VF, strf::print_traits<strf::separated_range_p<Iterator, CharT>> >;

namespace detail {

template <typename CharT, typename FPack, typename It>
class range_printer;

template <typename CharT, typename FPack, typename It>
class separated_range_printer;

template <typename CharT, typename FPack, typename It, typename... Fmts>
class fmt_range_printer;

template <typename CharT, typename FPack, typename It, typename... Fmts>
class fmt_separated_range_printer;

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class transformed_range_printer;

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class sep_transformed_range_printer;

} // namespace detail

template <typename It>
struct print_traits<strf::range_p<It>>
{
    using facet_tag = void;
    using forwarded_type = strf::range_p<It>;
    using fmt_type = strf::range_with_format<It>;

    template <typename CharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        (Preview& preview, const FPack& fp,  forwarded_type x)
        -> strf::usual_printer_input
            < CharT, Preview, FPack, forwarded_type
            , strf::detail::range_printer<CharT, FPack, It> >
    {
        return {preview, fp, x};
    }

    template <typename CharT, typename Preview, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , strf::value_with_format<strf::print_traits<strf::range_p<It>>, Fmts...> x )
        ->  strf::usual_printer_input
            < CharT
            , Preview, FPack
            , strf::value_with_format<strf::print_traits<strf::range_p<It>>, Fmts ...>
            , strf::detail::fmt_range_printer<CharT, FPack, It, Fmts...> >
    {
        return {preview, fp, x};
    }
};

template <typename It, typename SepCharT>
struct print_traits<strf::separated_range_p<It, SepCharT>>
{
    using facet_tag = void;
    using forwarded_type = strf::separated_range_p<It, SepCharT>;
    using fmt_type = strf::sep_range_with_format<It, SepCharT>;

    template <typename DestCharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        (Preview& preview, const FPack& fp,  forwarded_type x)
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack, forwarded_type
            , strf::detail::separated_range_printer<DestCharT, FPack, It> >
    {
        static_assert( std::is_same<SepCharT, DestCharT>::value
                     , "Character type of range separator string is different." );
        return {preview, fp, x};
    }

    template <typename DestCharT, typename Preview, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , strf::value_with_format<strf::print_traits<strf::separated_range_p<It, SepCharT>>, Fmts...> x )
        ->  strf::usual_printer_input
            < DestCharT
            , Preview, FPack
            , strf::value_with_format
                < strf::print_traits<strf::separated_range_p<It, SepCharT>>, Fmts... >
            , strf::detail::fmt_separated_range_printer<DestCharT, FPack, It, Fmts...> >
    {
        static_assert( std::is_same<SepCharT, DestCharT>::value
                     , "Character type of range separator string is different." );
        return {preview, fp, x};
    }
};

template <typename It, typename UnaryOp>
struct print_traits<strf::transformed_range_p<It, UnaryOp>>
{
    using facet_tag = void;
    using forwarded_type = strf::transformed_range_p<It, UnaryOp>;

    template <typename CharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        (Preview& preview, const FPack& fp,  forwarded_type x)
        -> strf::usual_printer_input
            < CharT, Preview, FPack, forwarded_type
            , strf::detail::transformed_range_printer<CharT, FPack, It, UnaryOp> >
    {
        return {preview, fp, x};
    }
};

template <typename It, typename SepCharT, typename UnaryOp>
struct print_traits<strf::separated_transformed_range_p<It, SepCharT, UnaryOp>>
{
    using facet_tag = void;
    using forwarded_type = strf::separated_transformed_range_p<It, SepCharT, UnaryOp>;

    template <typename DestCharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        (Preview& preview, const FPack& fp,  forwarded_type x)
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack, forwarded_type
            , strf::detail::sep_transformed_range_printer<DestCharT, FPack, It, UnaryOp> >
    {
        static_assert( std::is_same<SepCharT, DestCharT>::value
                     , "Character type of range separator string is different." );
        return {preview, fp, x};
    }
};

namespace detail {

template <typename CharT, typename FPack, typename It>
class range_printer: public strf::printer<CharT>
{
public:

    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    STRF_HD range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.fp)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, std::remove_cv_t<value_type> >;

    STRF_HD void preview_
        (strf::print_preview<strf::preview_size::no, strf::preview_width::no>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
};

template <typename CharT, typename FPack, typename It>
template <typename Preview, typename >
STRF_HD void range_printer<CharT, FPack, It>::preview_(Preview& preview) const
{
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>(preview, fp_, *it) );
    }
}

template <typename CharT, typename FPack, typename It>
STRF_HD void range_printer<CharT, FPack, It>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    using preview_type
        = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;
    preview_type no_preview;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<preview_type>
            ( strf::make_printer_input<CharT>(no_preview, fp_, *it) ).print_to(ob);
    }
}

template <typename CharT, typename FPack, typename It>
class separated_range_printer: public strf::printer<CharT>
{
public:

    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    STRF_HD separated_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.fp)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , sep_begin_(input.arg.sep_begin)
        , sep_len_(input.arg.sep_len)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, std::remove_cv_t<value_type> >;

    constexpr STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::size_t sep_len_;

    template <typename Category>
    decltype(auto) get_facet_(const FPack& fp) const
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename It>
template <typename Preview, typename>
STRF_HD void separated_range_printer<CharT, FPack, It>::preview_(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>(strf::make_printer_input<CharT>(preview, fp_, *it));
        ++ count;
        STRF_IF_CONSTEXPR (!Preview::size_required) {
            if (preview.remaining_width() <= 0) {
                return;
            }
        }
    }
    if (count < 2) {
        return;
    }
    if (Preview::width_required) {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( get_facet_<strf::char_encoding_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , sep_begin_
                                 , sep_len_
                                 , get_facet_<strf::surrogate_policy_c>(fp_) );
        if (dw != 0) {
            if (count > UINT32_MAX) {
                preview.clear_remaining_width();
            } else {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * sep_len_);
    }
}

template <typename CharT, typename FPack, typename It>
STRF_HD void separated_range_printer<CharT, FPack, It>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    using preview_type
        = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;
    preview_type no_preview;
    auto it = begin_;
    if (it != end_) {
        printer_type_<preview_type>
            ( strf::make_printer_input<CharT>(no_preview, fp_, *it) )
            .print_to(ob);
        while (++it != end_) {
            strf::write(ob, sep_begin_, sep_len_);
            printer_type_<preview_type>
                ( strf::make_printer_input<CharT>(no_preview, fp_, *it) )
                .print_to(ob);
        }
    }
}

template < typename CharT
         , typename FPack
         , typename It
         , typename ... Fmts >
class fmt_range_printer: public strf::printer<CharT>
{
    using value_type_ = strf::detail::iterator_value_type<It>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::print_traits<strf::range_p<It>> >;

public:

    template <typename... T>
    STRF_HD fmt_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.fp)
        , fmt_(input.arg)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, value_fmt_type_adapted_ >;

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    fmt_type_adapted_ fmt_;
};


template < typename CharT
         , typename FPack
         , typename It
         , typename ... Fmts >
template <typename Preview, typename >
STRF_HD void fmt_range_printer<CharT, FPack, It, Fmts ...>::preview_
    ( Preview& preview ) const
{
    auto r = fmt_.value();
    for(auto it = r.begin; it != r.end; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>
                ( preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) );
    }
}

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
STRF_HD void fmt_range_printer<CharT, FPack, It, Fmts ...>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    using preview_type
        = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;
    preview_type no_preview;
    auto r = fmt_.value();
    for(auto it = r.begin; it != r.end; ++it) {
        printer_type_<preview_type>
            ( strf::make_printer_input<CharT>
                ( no_preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
            .print_to(ob);
    }
}

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
class fmt_separated_range_printer: public strf::printer<CharT>
{
    using value_type_ = strf::detail::iterator_value_type<It>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::print_traits<strf::separated_range_p<It, CharT>> >;

public:

    template <typename... T>
    STRF_HD fmt_separated_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.fp)
        , fmt_(input.arg)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, value_fmt_type_adapted_ >;

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    fmt_type_adapted_ fmt_;

    template <typename Category>
    static inline STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
template <typename Preview, typename>
STRF_HD void fmt_separated_range_printer<CharT, FPack, It, Fmts ...>::preview_
    ( Preview& preview ) const
{
    auto r = fmt_.value();
    std::size_t count = 0;
    for(auto it = r.begin; it != r.end; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>
                ( preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) );
        ++ count;
        STRF_IF_CONSTEXPR (!Preview::size_required) {
            if (preview.remaining_width() <= 0) {
                return;
            }
        }
    }
    if (count < 2) {
        return;
    }
    if (Preview::width_required) {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( get_facet_<strf::char_encoding_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , r.sep_begin
                                 , r.sep_len
                                 , get_facet_<strf::surrogate_policy_c>(fp_) );
        if (dw != 0) {
            if (count > UINT32_MAX) {
                preview.clear_remaining_width();
            } else {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * r.sep_len);
    }
}

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
STRF_HD void fmt_separated_range_printer<CharT, FPack, It, Fmts ...>
::print_to( strf::basic_outbuff<CharT>& ob ) const
{
    using preview_type = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;
    preview_type no_preview;
    auto r = fmt_.value();
    auto it = r.begin;
    if (it != r.end) {
        printer_type_<preview_type>
            ( strf::make_printer_input<CharT>
                ( no_preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
            .print_to(ob);
        while(++it != r.end) {
            strf::write(ob, r.sep_begin, r.sep_len);
            printer_type_<preview_type>
                ( strf::make_printer_input<CharT>
                    ( no_preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
                .print_to(ob);
        }
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class transformed_range_printer: public strf::printer<CharT>
{
public:

    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    transformed_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.fp)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , op_(input.arg.op)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT
        , Preview, FPack
        , std::remove_reference_t
            < decltype(std::declval<UnaryOp>()(*std::declval<iterator>())) > >;

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    UnaryOp op_;
};

template <typename CharT, typename FPack, typename It, typename UnaryOp>
template <typename Preview, typename >
STRF_HD void transformed_range_printer<CharT, FPack, It, UnaryOp>
    ::preview_(Preview& preview) const
{
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>(preview, fp_, op_(*it)) );
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
STRF_HD void transformed_range_printer<CharT, FPack, It, UnaryOp>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    using preview_type
        = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;
    preview_type no_preview;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<preview_type>
            ( strf::make_printer_input<CharT>(no_preview, fp_, op_(*it)) )
            .print_to(ob);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class sep_transformed_range_printer: public strf::printer<CharT>
{
public:
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    STRF_HD sep_transformed_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.fp)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , sep_begin_(input.arg.sep_begin)
        , sep_len_(input.arg.sep_len)
        , op_(input.arg.op)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT
        , Preview, FPack
        , std::remove_reference_t
            < decltype(std::declval<UnaryOp>()(*std::declval<iterator>())) > >;

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::size_t sep_len_;
    UnaryOp op_;

    template <typename Category>
    STRF_HD decltype(auto) get_facet_(const FPack& fp) const
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename It, typename UnaryOp>
template <typename Preview, typename>
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>
    ::preview_(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>(preview, fp_, op_(*it)) );
        ++ count;
        STRF_IF_CONSTEXPR (!Preview::size_required) {
            if (preview.remaining_width() <= 0) {
                return;
            }
        }
    }
    if (count < 2) {
        return;
    }
    if (Preview::width_required) {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( get_facet_<strf::char_encoding_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , sep_begin_
                                 , sep_len_
                                 , get_facet_<strf::surrogate_policy_c>(fp_) );
        if (dw != 0) {
            if (count > UINT32_MAX) {
                preview.clear_remaining_width();
            } else {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * sep_len_);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    using preview_type = strf::print_preview
        < strf::preview_size::no, strf::preview_width::no >;
    preview_type no_preview;
    auto it = begin_;
    if (it != end_) {
        printer_type_<preview_type>
            ( strf::make_printer_input<CharT>(no_preview, fp_, op_(*it)) )
            .print_to(ob);
        while (++it != end_) {
            strf::write(ob, sep_begin_, sep_len_);
            printer_type_<preview_type>
                ( strf::make_printer_input<CharT>(no_preview, fp_, op_(*it)) )
                .print_to(ob);
        }
    }
}

} // namespace detail

template <typename It>
inline STRF_HD auto range(It begin, It end)
{
    return strf::range_p<It>{begin, end};
}

template <typename It, typename CharT>
inline STRF_HD auto separated_range(It begin, It end, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::separated_range_p<It, CharT>
        {begin, end, sep, sep_len};
}

template <typename Range, typename It = typename Range::const_iterator>
inline STRF_HD auto range(const Range& range)
{
    using namespace std;
    return strf::range_p<It>{begin(range), end(range)};
}

template <typename T, std::size_t N>
inline STRF_HD auto range(T (&array)[N])
{
    return strf::range_p<const T*>{&array[0], &array[0] + N};
}

template <typename Range, typename CharT>
inline STRF_HD auto separated_range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    using namespace std;
    return strf::separated_range_p
        <typename Range::const_iterator, CharT>
        {begin(range), end(range), sep, sep_len};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto separated_range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::separated_range_p<const T*, CharT>
        {&array[0], &array[0] + N, sep, sep_len};
}

template <typename It>
inline STRF_HD auto fmt_range(It begin, It end)
{
    return strf::range_with_format<It>{{begin, end}};
}

template <typename It, typename CharT>
inline STRF_HD auto fmt_separated_range(It begin, It end, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::sep_range_with_format<It, CharT>
        {{begin, end, sep, sep_len}};
}

template <typename Range, typename It = typename Range::const_iterator>
inline STRF_HD
strf::range_with_format<It> fmt_range(const Range& range)
{
    using namespace std;
    strf::range_p<It> r{begin(range), end(range)};
    return strf::range_with_format<It>{r};
}

template <typename T, std::size_t N>
inline STRF_HD auto fmt_range(T (&array)[N])
{
    using namespace std;
    using fmt_type = strf::range_with_format<const T*>;
    return fmt_type{{&array[0], &array[0] + N}};
}

template < typename Range
         , typename CharT
         , typename It = typename Range::const_iterator >
inline STRF_HD auto fmt_separated_range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    using namespace std;
    strf::separated_range_p<It, CharT> r
    { begin(range), end(range), sep, sep_len };
    return strf::sep_range_with_format<It, CharT>{r};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto fmt_separated_range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    using namespace std;
    return strf::sep_range_with_format<const T*, CharT>
        { {&array[0], &array[0] + N, sep, sep_len} };
}

template < typename It
         , typename UnaryOp
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto range(It begin, It end, UnaryOp op)
{
    return strf::transformed_range_p<It, UnaryOp>{begin, end, op};
}

template < typename Range
         , typename UnaryOp
         , typename It = typename Range::const_iterator
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto range(const Range& range, UnaryOp op)
{
    using namespace std;
    return strf::transformed_range_p<It, UnaryOp>{begin(range), end(range), op};
}

template < typename T
         , std::size_t N
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*(T*)0)) >
inline STRF_HD auto range(T (&array)[N], UnaryOp op)
{
    return strf::transformed_range_p<const T*, UnaryOp>
        {&array[0], &array[0] + N, op};
}

template < typename It
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto separated_range(It begin, It end, const CharT* sep, UnaryOp op)
{
    std::size_t sep_len = strf::detail::str_length(sep);
    return strf::separated_transformed_range_p<It, CharT, UnaryOp>
        { begin, end, sep, sep_len, op };
}

template < typename Range
         , typename CharT
         , typename UnaryOp
         , typename It = typename Range::const_iterator
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto separated_range(const Range& range, const CharT* sep, UnaryOp op)
{
    std::size_t sep_len = strf::detail::str_length(sep);
    using namespace std;
    return strf::separated_transformed_range_p<It, CharT, UnaryOp>
        { begin(range), end(range), sep, sep_len, op };
}

template < typename T
         , std::size_t N
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*(T*)0)) >
inline STRF_HD auto separated_range(T (&array)[N], const CharT* sep, UnaryOp op)
{
    std::size_t sep_len = strf::detail::str_length(sep);
    return strf::separated_transformed_range_p<const T*, CharT, UnaryOp>
        { &array[0], &array[0] + N, sep, sep_len, op };
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_RANGE_HPP

