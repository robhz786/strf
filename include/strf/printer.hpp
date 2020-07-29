#ifndef STRF_PRINTER_HPP
#define STRF_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuff.hpp>
#include <strf/width_t.hpp>
#include <strf/facets_pack.hpp>

namespace strf {

template <typename CharT>
class printer
{
public:

    using char_type = CharT;

    STRF_HD virtual ~printer()
    {
    }

    STRF_HD virtual void print_to(strf::basic_outbuff<CharT>& ob) const = 0;
};

struct string_input_tag_base
{
};

template <typename CharIn>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;

template <typename CharIn>
struct tr_string_input_tag: strf::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct range_separator_input_tag: strf::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct is_tr_string_of
{
    template <typename T>
    using fn = std::is_same<strf::tr_string_input_tag<CharIn>, T>;
};

template <typename T>
struct is_tr_string: std::false_type
{
};

template <typename CharIn>
struct is_tr_string<strf::is_tr_string_of<CharIn>> : std::true_type
{
};

template <bool Active>
class width_preview;

template <>
class width_preview<true>
{
public:

    explicit constexpr STRF_HD width_preview(strf::width_t initial_width) noexcept
        : width_(initial_width)
    {}

    STRF_HD width_preview(const width_preview&) = delete;

    constexpr STRF_HD void subtract_width(strf::width_t w) noexcept
    {
        width_ -= w;
    }

    constexpr STRF_HD void checked_subtract_width(strf::width_t w) noexcept
    {
        if (w < width_) {
            width_ -= w;
        } else {
            width_ = 0;
        }
    }

    constexpr STRF_HD void checked_subtract_width(std::ptrdiff_t w) noexcept
    {
        if (w < width_.ceil()) {
            width_ -= static_cast<std::int16_t>(w);
        } else {
            width_ = 0;
        }
    }

    constexpr STRF_HD void clear_remaining_width() noexcept
    {
        width_ = 0;
    }

    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return width_;
    }

private:

    strf::width_t width_;
};

template <>
class width_preview<false>
{
public:

    constexpr STRF_HD width_preview() noexcept
    {
    }

    constexpr STRF_HD void subtract_width(strf::width_t) noexcept
    {
    }

    constexpr STRF_HD void checked_subtract_width(strf::width_t) noexcept
    {
    }

    constexpr STRF_HD void checked_subtract_width(std::ptrdiff_t) noexcept
    {
    }

    constexpr STRF_HD void clear_remaining_width() noexcept
    {
    }

    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return 0;
    }
};

template <bool Active>
class size_preview;

template <>
class size_preview<true>
{
public:
    explicit constexpr STRF_HD size_preview(std::size_t initial_size = 0) noexcept
        : size_(initial_size)
    {
    }

    STRF_HD size_preview(const size_preview&) = delete;

    constexpr STRF_HD void add_size(std::size_t s) noexcept
    {
        size_ += s;
    }

    constexpr STRF_HD std::size_t get_size() const noexcept
    {
        return size_;
    }

private:

    std::size_t size_;
};

template <>
class size_preview<false>
{
public:

    constexpr STRF_HD size_preview() noexcept
    {
    }

    constexpr STRF_HD void add_size(std::size_t) noexcept
    {
    }

    constexpr STRF_HD std::size_t get_size() const noexcept
    {
        return 0;
    }
};

enum class preview_width: bool { no = false, yes = true };
enum class preview_size : bool { no = false, yes = true };

template <strf::preview_size SizeRequired, strf::preview_width WidthRequired>
class print_preview
    : public strf::size_preview<static_cast<bool>(SizeRequired)>
    , public strf::width_preview<static_cast<bool>(WidthRequired)>
{
public:

    static constexpr bool size_required = static_cast<bool>(SizeRequired);
    static constexpr bool width_required = static_cast<bool>(WidthRequired);
    static constexpr bool nothing_required = ! size_required && ! width_required;

    template <strf::preview_width W = WidthRequired>
    STRF_HD constexpr explicit print_preview
        ( std::enable_if_t<static_cast<bool>(W), strf::width_t> initial_width ) noexcept
        : strf::width_preview<true>{initial_width}
    {
    }

    constexpr STRF_HD print_preview() noexcept
    {
    }
};

using no_print_preview = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;

namespace detail {

#if defined(__cpp_fold_expressions)

template <typename CharT, typename ... Printers>
inline STRF_HD void write_args( strf::basic_outbuff<CharT>& ob
                              , const Printers& ... printers )
{
    (... , printers.print_to(ob));
}

#else // defined(__cpp_fold_expressions)

template <typename CharT>
inline STRF_HD void write_args(strf::basic_outbuff<CharT>&)
{
}

template <typename CharT, typename Printer, typename ... Printers>
inline STRF_HD void write_args
    ( strf::basic_outbuff<CharT>& ob
    , const Printer& printer
    , const Printers& ... printers )
{
    printer.print_to(ob);
    if (ob.good()) {
        write_args<CharT>(ob, printers ...);
    }
}

#endif // defined(__cpp_fold_expressions)

} // namespace detail

namespace detail{

template
    < class From
    , class To
    , template <class ...> class List
    , class ... T >
struct fmt_replace_impl2
{
    template <class U>
    using f = std::conditional_t<std::is_same<From, U>::value, To, U>;

    using type = List<f<T> ...>;
};

template <class From, class List>
struct fmt_replace_impl;

template
    < class From
    , template <class ...> class List
    , class ... T>
struct fmt_replace_impl<From, List<T ...> >
{
    template <class To>
    using type_tmpl =
        typename strf::detail::fmt_replace_impl2
            < From, To, List, T...>::type;
};

template <typename FmtA, typename FmtB, typename ValueWithFormat>
struct fmt_forward_switcher
{
    template <typename FmtAInit>
    static STRF_HD const typename FmtB::template fn<ValueWithFormat>&
    f(const FmtAInit&, const ValueWithFormat& v)
    {
        return v;
    }

    template <typename FmtAInit>
    static STRF_HD typename FmtB::template fn<ValueWithFormat>&&
    f(const FmtAInit&, ValueWithFormat&& v)
    {
        return v;
    }
};

template <typename FmtA, typename ValueWithFormat>
struct fmt_forward_switcher<FmtA, FmtA, ValueWithFormat>
{
    template <typename FmtAInit>
    static constexpr STRF_HD FmtAInit&&
    f(std::remove_reference_t<FmtAInit>& fa,  const ValueWithFormat&)
    {
        return static_cast<FmtAInit&&>(fa);
    }

    template <typename FmtAInit>
    static constexpr STRF_HD FmtAInit&&
    f(std::remove_reference_t<FmtAInit>&& fa, const ValueWithFormat&)
    {
        return static_cast<FmtAInit&&>(fa);
    }
};


} // namespace detail

template <typename List, typename From, typename To>
using fmt_replace
    = typename strf::detail::fmt_replace_impl<From, List>
    ::template type_tmpl<To>;

template <typename ValueType, class ... Fmts>
class value_with_format;

template <typename ValueType, class ... Fmts>
class value_with_format
    : public Fmts::template fn<value_with_format<ValueType, Fmts ...>> ...
{
public:

    using value_type = ValueType;

    template <typename ... OhterFmts>
    using replace_fmts = strf::value_with_format<ValueType, OhterFmts ...>;

    explicit constexpr STRF_HD value_with_format(const ValueType& v)
        : value_(v)
    {
    }

    template <typename OtherValueType>
    constexpr STRF_HD value_with_format
        ( const ValueType& v
        , const strf::value_with_format<OtherValueType, Fmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>>& >(f) )
        ...
        , value_(v)
    {
    }

    template <typename OtherValueType>
    constexpr STRF_HD value_with_format
        ( const ValueType& v
        , strf::value_with_format<OtherValueType, Fmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>> &&>(f) )
        ...
        , value_(static_cast<ValueType&&>(v))
    {
    }

    template <typename ... F, typename ... FInit>
    constexpr STRF_HD value_with_format
        ( const ValueType& v
        , strf::tag<F...>
        , FInit&& ... finit )
        : F::template fn<value_with_format<ValueType, Fmts...>>
            (std::forward<FInit>(finit))
        ...
        , value_(v)
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD value_with_format
        ( const strf::value_with_format<ValueType, OtherFmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>& >(f) )
        ...
        , value_(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD value_with_format
        ( strf::value_with_format<ValueType, OtherFmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>&& >(f) )
        ...
        , value_(static_cast<ValueType&&>(f.value()))
    {
    }

    template <typename Fmt, typename FmtInit, typename ... OtherFmts>
    constexpr STRF_HD value_with_format
        ( const strf::value_with_format<ValueType, OtherFmts...>& f
        , strf::tag<Fmt>
        , FmtInit&& fmt_init )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( strf::detail::fmt_forward_switcher
                  < Fmt
                  , Fmts
                  , strf::value_with_format<ValueType, OtherFmts...> >
              :: template f<FmtInit>(fmt_init, f) )
            ...
        , value_(f.value())
    {
    }

    constexpr STRF_HD const ValueType& value() const
    {
        return value_;
    }

    constexpr STRF_HD ValueType& value()
    {
        return value_;
    }

private:

    ValueType value_;
};

template <typename CharT>
struct printer_input_tag
{
private:
    static const printer_input_tag<CharT>& tag_();

public:

    template <typename Arg, typename Preview, typename FPack>
    constexpr STRF_HD auto operator()(Arg&& arg, Preview& preview, const FPack& fp) const
        noexcept(noexcept(strf::detail::tag_invoke(tag_(), arg, preview, fp)))
        -> decltype(strf::detail::tag_invoke(tag_(), arg, preview, fp))
    {
        return strf::detail::tag_invoke(*this, arg, preview, fp);
    }
};


struct printing_c;

struct default_printing_facet
{
#if defined(__cpp_inline_variable)

    template <typename CharT>
    constexpr static printer_input_tag<CharT> make_printer_input = {};

#else

private:

    template <typename CharT>
    using tag_ = printer_input_tag<CharT>;

public:

    template <typename CharT, typename Arg, typename Preview, typename FPack>
    constexpr STRF_HD auto make_printer_input(Arg&& arg, Preview& preview, const FPack& fp) const
        noexcept(noexcept(strf::detail::tag_invoke(tag_<CharT>(), arg, preview, fp)))
        -> decltype(strf::detail::tag_invoke(tag_<CharT>(), arg, preview, fp))
    {
        return strf::detail::tag_invoke(tag_<CharT>(), arg, preview, fp);
    }

#endif
};

struct printing_c
{
    static constexpr STRF_HD strf::default_printing_facet get_default() noexcept
    {
        return {};
    }
};

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename CharT, typename Arg, typename Preview, typename FPack>
constexpr STRF_HD decltype(auto) make_default_printer_input
    (Arg&& arg, Preview& preview, const FPack& fp)
{
    strf::printer_input_tag<CharT> tag;
    return tag(arg, preview, fp);
}


template <typename CharT, typename Arg, typename Preview, typename FPack>
constexpr STRF_HD decltype(auto) make_printer_input
    (Arg&& arg, Preview& preview, const FPack& fp)
{
    return strf::get_facet<strf::printing_c, Arg>(fp)
        .template make_printer_input<CharT>(arg, preview, fp);
}

#else

namespace detail {

template <typename CharT>
struct make_printer_input_impl
{
    template <typename Arg, typename Preview, typename FPack>
    constexpr STRF_HD decltype(auto) operator()
        (const Arg& arg, Preview& preview, const FPack& fp) const
        noexcept(noexcept(strf::get_facet<strf::printing_c, Arg>(fp)
                          .template make_printer_input<CharT>(arg, preview, fp)))
    {
        return strf::get_facet<strf::printing_c, Arg>(fp)
            .template make_printer_input<CharT>(arg, preview, fp);
    }
};

} // namespace detail

template <typename CharT>
constexpr strf::printer_input_tag<CharT> make_default_printer_input = {};

template <typename CharT>
constexpr strf::detail::make_printer_input_impl<CharT> make_printer_input = {};

#endif // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

namespace detail {

template <typename CharT, typename Arg, typename Preview, typename FPack>
struct printer_impl_helper
{
    static const FPack& fp();
    static Preview& preview();
    static const Arg& arg();

    using default_printer_input = decltype
        ( strf::make_default_printer_input<CharT>(arg(), preview(), fp()) );

    using printer_input = decltype
        ( strf::make_printer_input<CharT>(arg(), preview(), fp()) );

    using default_printer = typename default_printer_input::printer_type;
    using printer = typename printer_input::printer_type;
};

} // namespace detail

template <typename CharT, typename Arg, typename Preview, typename FPack>
using default_printer_impl =
    typename strf::detail::printer_impl_helper<CharT, Arg, Preview, FPack>::default_printer;

template <typename CharT, typename Arg, typename Preview, typename FPack>
using printer_impl =
    typename strf::detail::printer_impl_helper<CharT, Arg, Preview, FPack>::printer;

template <typename CharT, typename Arg, typename Preview, typename FPack, typename Printer>
struct usual_printer_input;

template< typename CharT
        , typename Arg
        , strf::preview_size PreviewSize
        , strf::preview_width PreviewWidth
        , typename FPack
        , typename Printer >
struct usual_printer_input
    <CharT, Arg, strf::print_preview<PreviewSize, PreviewWidth>, FPack, Printer>
{
    using char_type = CharT;
    using arg_type = Arg;
    using preview_type = strf::print_preview<PreviewSize, PreviewWidth>;
    using fpack_type = FPack;
    using printer_type = Printer;

    Arg arg;
    preview_type& preview;
    FPack fp;
};

struct fmt_tag {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::detail::tag_invoke(*(const fmt_tag*)0, value)))
        -> decltype(strf::detail::tag_invoke(*(const fmt_tag*)0, value))
    {
        return strf::detail::tag_invoke(*this, value);
    }
};

inline namespace format_functions {

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename T>
constexpr STRF_HD auto fmt(T&& value)
    noexcept(noexcept(strf::detail::tag_invoke(strf::fmt_tag{}, value)))
    -> decltype(strf::detail::tag_invoke(strf::fmt_tag{}, value))
{
    return strf::detail::tag_invoke(strf::fmt_tag{}, value);
}

#else //defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

constexpr fmt_tag fmt {};

#endif

} // inline namespace format_functions

template <typename T>
using fmt_type = std::remove_cv_t<std::remove_reference_t<decltype(strf::fmt(std::declval<T>()))>>;

template <bool HasAlignment>
struct alignment_format_q;

enum class text_alignment {left, right, split, center};

struct alignment_format_data
{
    char32_t fill = U' ';
    std::int16_t width = 0;
    strf::text_alignment alignment = strf::text_alignment::right;
};

constexpr STRF_HD bool operator==( strf::alignment_format_data lhs
                                 , strf::alignment_format_data rhs ) noexcept
{
    return lhs.fill == rhs.fill
        && lhs.width == rhs.width
        && lhs.alignment == rhs.alignment ;
}

constexpr STRF_HD bool operator!=( strf::alignment_format_data lhs
                                 , strf::alignment_format_data rhs ) noexcept
{
    return ! (lhs == rhs);
}

template <class T, bool HasAlignment>
class alignment_format_fn
{
    STRF_HD T& as_derived_ref()
    {
        T* d =  static_cast<T*>(this);
        return *d;
    }

    STRF_HD T&& as_derived_rval_ref()
    {
        T* d =  static_cast<T*>(this);
        return static_cast<T&&>(*d);
    }

public:

    constexpr STRF_HD alignment_format_fn() noexcept
    {
    }

    constexpr STRF_HD explicit alignment_format_fn
        ( strf::alignment_format_data data) noexcept
        : data_(data)
    {
    }

    template <typename U, bool B>
    constexpr STRF_HD explicit alignment_format_fn
        ( const strf::alignment_format_fn<U, B>& u ) noexcept
        : data_(u.get_alignment_format_data())
    {
    }

    constexpr STRF_HD T&& operator<(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::left;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& operator>(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::right;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& operator^(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::center;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& operator%(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::split;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& fill(char32_t ch) && noexcept
    {
        data_.fill = ch;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& set(alignment_format_data data) && noexcept
    {
        data_ = data;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD std::int16_t width() const noexcept
    {
        return data_.width;
    }
    constexpr STRF_HD strf::text_alignment alignment() const noexcept
    {
        return data_.alignment;
    }
    constexpr STRF_HD char32_t fill() const noexcept
    {
        return data_.fill;
    }

    constexpr STRF_HD alignment_format_data get_alignment_format_data() const noexcept
    {
        return data_;
    }

private:

    strf::alignment_format_data data_;
};

template <class T>
class alignment_format_fn<T, false>
{
    using derived_type = T;
    using adapted_derived_type = strf::fmt_replace
            < T
            , strf::alignment_format_q<false>
            , strf::alignment_format_q<true> >;

    constexpr STRF_HD adapted_derived_type make_adapted() const
    {
        return adapted_derived_type{static_cast<const T&>(*this)};
    }

public:

    constexpr STRF_HD alignment_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit alignment_format_fn(const alignment_format_fn<U, false>&) noexcept
    {
    }

    constexpr STRF_HD adapted_derived_type operator<(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::left } };
    }
    constexpr STRF_HD adapted_derived_type operator>(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::right } };
    }
    constexpr STRF_HD adapted_derived_type operator^(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::center } };
    }
    constexpr STRF_HD adapted_derived_type operator%(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::split } };
    }
    constexpr STRF_HD auto fill(char32_t ch) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ ch } };
    }
    constexpr STRF_HD auto set(strf::alignment_format_data data) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::alignment_format_q<true>>{}
            , data };
    }
    constexpr STRF_HD std::int16_t width() const noexcept
    {
        return 0;
    }
    constexpr STRF_HD strf::text_alignment alignment() const noexcept
    {
        return strf::text_alignment::right;
    }
    constexpr STRF_HD char32_t fill() const noexcept
    {
        return U' ';
    }
    constexpr STRF_HD alignment_format_data get_alignment_format_data() const noexcept
    {
        return {};
    }
};

template <bool HasAlignment>
struct alignment_format_q
{
    template <class T>
    using fn = strf::alignment_format_fn<T, HasAlignment>;
};

using alignment_format = strf::alignment_format_q<true>;
using empty_alignment_format = strf::alignment_format_q<false>;


template <class T>
class quantity_format_fn
{
public:

    constexpr STRF_HD quantity_format_fn(std::size_t count) noexcept
        : count_(count)
    {
    }

    constexpr STRF_HD quantity_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit quantity_format_fn(const quantity_format_fn<U>& u) noexcept
        : count_(u.count())
    {
    }

    constexpr STRF_HD T&& multi(std::size_t count) && noexcept
    {
        count_ = count;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD std::size_t count() const noexcept
    {
        return count_;
    }

private:

    std::size_t count_ = 1;
};

struct quantity_format
{
    template <class T>
    using fn = strf::quantity_format_fn<T>;
};


inline namespace format_functions {

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename T>
constexpr STRF_HD auto hex(T&& value)
    noexcept(noexcept(strf::fmt(value).hex()))
    -> std::remove_reference_t<decltype(strf::fmt(value).hex())>
{
    return strf::fmt(value).hex();
}

template <typename T>
constexpr STRF_HD auto dec(T&& value)
    noexcept(noexcept(strf::fmt(value).dec()))
    -> std::remove_reference_t<decltype(strf::fmt(value).dec())>
{
    return strf::fmt(value).dec();
}

template <typename T>
constexpr STRF_HD auto oct(T&& value)
    noexcept(noexcept(strf::fmt(value).oct()))
    -> std::remove_reference_t<decltype(strf::fmt(value).oct())>
{
    return strf::fmt(value).oct();
}

template <typename T>
constexpr STRF_HD auto bin(T&& value)
    noexcept(noexcept(strf::fmt(value).bin()))
    -> std::remove_reference_t<decltype(strf::fmt(value).bin())>
{
    return strf::fmt(value).bin();
}

template <typename T>
constexpr STRF_HD auto fixed(T&& value)
    noexcept(noexcept(strf::fmt(value).fixed()))
    -> std::remove_reference_t<decltype(strf::fmt(value).fixed())>
{
    return strf::fmt(value).fixed();
}

template <typename T>
    constexpr STRF_HD auto fixed(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt(value).fixed().p(precision)))
    -> std::remove_reference_t<decltype(strf::fmt(value).fixed().p(precision))>
{
    return strf::fmt(value).fixed().p(precision);
}

template <typename T>
constexpr STRF_HD auto sci(T&& value)
    noexcept(noexcept(strf::fmt(value).sci()))
    -> std::remove_reference_t<decltype(strf::fmt(value).sci())>
{
    return strf::fmt(value).sci();
}

template <typename T>
constexpr STRF_HD auto sci(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt(value).sci().p(precision)))
    -> std::remove_reference_t<decltype(strf::fmt(value).sci().p(precision))>
{
    return strf::fmt(value).sci().p(precision);
}

template <typename T>
constexpr STRF_HD auto gen(T&& value)
    noexcept(noexcept(strf::fmt(value).gen()))
    -> std::remove_reference_t<decltype(strf::fmt(value).gen())>
{
    return strf::fmt(value).gen();
}

template <typename T>
constexpr STRF_HD auto gen(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt(value).gen().p(precision)))
    -> std::remove_reference_t<decltype(strf::fmt(value).gen().p(precision))>
{
    return strf::fmt(value).gen().p(precision);
}

template <typename T, typename C>
constexpr STRF_HD auto multi(T&& value, C&& count)
    noexcept(noexcept(strf::fmt(value).multi(count)))
    -> std::remove_reference_t<decltype(strf::fmt(value).multi(count))>
{
    return strf::fmt(value).multi(count);
}

template <typename T>
constexpr STRF_HD auto conv(T&& value)
    noexcept(noexcept(strf::fmt(value).convert_encoding()))
    -> std::remove_reference_t<decltype(strf::fmt(value).convert_encoding())>
{
    return strf::fmt(value).convert_encoding();
}

template <typename T, typename E>
    constexpr STRF_HD auto conv(T&& value, E&& enc)
    noexcept(noexcept(strf::fmt(value).convert_from_encoding(enc)))
    -> std::remove_reference_t<decltype(strf::fmt(value).convert_from_encoding(enc))>
{
    return strf::fmt(value).convert_from_encoding(enc);
}

template <typename T>
constexpr STRF_HD auto sani(T&& value)
    noexcept(noexcept(strf::fmt(value).sanitize_encoding()))
    -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_encoding())>
{
    return strf::fmt(value).sanitize_encoding();
}

template <typename T, typename E>
    constexpr STRF_HD auto sani(T&& value, E&& enc)
    noexcept(noexcept(strf::fmt(value).sanitize_from_encoding(enc)))
    -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_from_encoding(enc))>
{
    return strf::fmt(value).sanitize_from_encoding(enc);
}

template <typename T>
constexpr STRF_HD auto right(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) > width))
    -> std::remove_reference_t<decltype(strf::fmt(value) > width)>
{
    return strf::fmt(value) > width;
}

template <typename T>
constexpr STRF_HD auto right(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) > width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) > width)>
{
    return strf::fmt(value).fill(fill) > width;
}

template <typename T>
constexpr STRF_HD auto left(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) < width))
    -> std::remove_reference_t<decltype(strf::fmt(value) < width)>
{
    return strf::fmt(value) < width;
}

template <typename T>
constexpr STRF_HD auto left(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) < width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) < width)>
{
    return strf::fmt(value).fill(fill) < width;
}

template <typename T>
constexpr STRF_HD auto center(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) ^ width))
    -> std::remove_reference_t<decltype(strf::fmt(value) ^ width)>
{
    return strf::fmt(value) ^ width;
}

template <typename T>
constexpr STRF_HD auto center(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) ^ width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) ^ width)>
{
    return strf::fmt(value).fill(fill) ^ width;
}

template <typename T>
constexpr STRF_HD auto split(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) % width))
    -> std::remove_reference_t<decltype(strf::fmt(value) % width)>
{
    return strf::fmt(value) % width;
}

template <typename T>
constexpr STRF_HD auto split(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) % width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) % width)>
{
    return strf::fmt(value).fill(fill) % width;
}

#else  // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

namespace detail_format_functions {

struct hex_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).hex()))
        -> std::remove_reference_t<decltype(strf::fmt(value).hex())>
    {
        return strf::fmt(value).hex();
    }
};

struct dec_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).dec()))
        -> std::remove_reference_t<decltype(strf::fmt(value).dec())>
    {
        return strf::fmt(value).dec();
    }
};

struct oct_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).oct()))
        -> std::remove_reference_t<decltype(strf::fmt(value).oct())>
    {
        return strf::fmt(value).oct();
    }
};

struct bin_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).bin()))
        -> std::remove_reference_t<decltype(strf::fmt(value).bin())>
    {
        return strf::fmt(value).bin();
    }
};

struct fixed_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).fixed()))
        -> std::remove_reference_t<decltype(strf::fmt(value).fixed())>
    {
        return strf::fmt(value).fixed();
    }
    template <typename T>
        constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt(value).fixed().p(precision)))
        -> std::remove_reference_t<decltype(strf::fmt(value).fixed().p(precision))>
    {
        return strf::fmt(value).fixed().p(precision);
    }
};

struct sci_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).sci()))
        -> std::remove_reference_t<decltype(strf::fmt(value).sci())>
    {
        return strf::fmt(value).sci();
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt(value).sci().p(precision)))
        -> std::remove_reference_t<decltype(strf::fmt(value).sci().p(precision))>
    {
        return strf::fmt(value).sci().p(precision);
    }
};

struct gen_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).gen()))
        -> std::remove_reference_t<decltype(strf::fmt(value).gen())>
    {
        return strf::fmt(value).gen();
    }
    template <typename T>
        constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt(value).gen().p(precision)))
        -> std::remove_reference_t<decltype(strf::fmt(value).gen().p(precision))>
    {
        return strf::fmt(value).gen().p(precision);
    }
};

struct multi_fn {
    template <typename T, typename C>
    constexpr STRF_HD auto operator()(T&& value, C&& count) const
        noexcept(noexcept(strf::fmt(value).multi(count)))
        -> std::remove_reference_t<decltype(strf::fmt(value).multi(count))>
    {
        return strf::fmt(value).multi(count);
    }
};

struct conv_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).convert_encoding()))
        -> std::remove_reference_t<decltype(strf::fmt(value).convert_encoding())>
    {
        return strf::fmt(value).convert_encoding();
    }
    template <typename T, typename E>
        constexpr STRF_HD auto operator()(T&& value, E&& enc) const
        noexcept(noexcept(strf::fmt(value).convert_from_encoding(enc)))
        -> std::remove_reference_t<decltype(strf::fmt(value).convert_from_encoding(enc))>
    {
        return strf::fmt(value).convert_from_encoding(enc);
    }
};

struct sani_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).sanitize_encoding()))
        -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_encoding())>
    {
        return strf::fmt(value).sanitize_encoding();
    }
    template <typename T, typename E>
    constexpr STRF_HD auto operator()(T&& value, E&& enc) const
        noexcept(noexcept(strf::fmt(value).sanitize_from_encoding(enc)))
        -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_from_encoding(enc))>
    {
        return strf::fmt(value).sanitize_from_encoding(enc);
    }
};

struct right_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) > width))
        -> std::remove_reference_t<decltype(strf::fmt(value) > width)>
    {
        return strf::fmt(value) > width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) > width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) > width)>
    {
        return strf::fmt(value).fill(fill) > width;
    }
};

struct left_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) < width))
        -> std::remove_reference_t<decltype(strf::fmt(value) < width)>
    {
        return strf::fmt(value) < width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) < width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) < width)>
    {
        return strf::fmt(value).fill(fill) < width;
    }
};

struct center_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) ^ width))
        -> std::remove_reference_t<decltype(strf::fmt(value) ^ width)>
    {
        return strf::fmt(value) ^ width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) ^ width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) ^ width)>
    {
        return strf::fmt(value).fill(fill) ^ width;
    }
};

struct split_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) % width))
        -> std::remove_reference_t<decltype(strf::fmt(value) % width)>
    {
        return strf::fmt(value) % width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) % width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) % width)>
    {
        return strf::fmt(value).fill(fill) % width;
    }
};

} // namespace detail_format_functions

constexpr strf::detail_format_functions::hex_fn    hex {};
constexpr strf::detail_format_functions::dec_fn    dec {};
constexpr strf::detail_format_functions::oct_fn    oct {};
constexpr strf::detail_format_functions::bin_fn    bin {};
constexpr strf::detail_format_functions::fixed_fn  fixed {};
constexpr strf::detail_format_functions::sci_fn    sci {};
constexpr strf::detail_format_functions::gen_fn    gen {};
constexpr strf::detail_format_functions::multi_fn  multi {};
constexpr strf::detail_format_functions::conv_fn   conv {};
constexpr strf::detail_format_functions::sani_fn   sani {};
constexpr strf::detail_format_functions::right_fn  right {};
constexpr strf::detail_format_functions::left_fn   left {};
constexpr strf::detail_format_functions::center_fn center {};
constexpr strf::detail_format_functions::split_fn  split {};

#endif // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

} // inline namespace format_functions

} // namespace strf

#endif // STRF_PRINTER_HPP
