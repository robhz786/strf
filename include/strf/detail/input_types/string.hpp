#ifndef STRF_DETAIL_INPUT_TYPES_STRING
#define STRF_DETAIL_INPUT_TYPES_STRING

#include <limits>
#include <string>
#include <strf/detail/facets/width_calculator.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/facets_pack.hpp>

namespace strf {
namespace detail {

template <typename CharIn>
class simple_string_view
{
public:

    using iterator = const CharIn*;
    using const_iterator = const CharIn*;

#if defined(STRF_HAS_STD_STRING_VIEW)

    template <typename Traits>
    constexpr STRF_HD simple_string_view(std::basic_string_view<CharIn, Traits> sv)
        : begin_(sv.data())
        , len_(sv.size())
    {
    }

#endif //defined(STRF_HAS_STD_STRING_VIEW)

    template <typename Traits, typename Allocator>
    STRF_HD simple_string_view(const std::basic_string<CharIn, Traits, Allocator>& s)
        : begin_(s.data())
        , len_(s.size())
    {
    }

    constexpr STRF_HD simple_string_view(const CharIn* begin, const CharIn* end) noexcept
        : begin_(begin)
        , len_(end - begin)
    {
    }
    constexpr STRF_HD simple_string_view(const CharIn* str, std::size_t len) noexcept
        : begin_(str)
        , len_(len)
    {
    }

    STRF_CONSTEXPR_CHAR_TRAITS
    STRF_HD simple_string_view(const CharIn* str) noexcept
        : begin_(str)
        , len_(strf::detail::str_length<CharIn>(str))
    {
    }
    constexpr STRF_HD const CharIn* begin() const
    {
        return begin_;
    }
    constexpr STRF_HD const CharIn* end() const
    {
        return begin_ + len_;
    }
    constexpr STRF_HD std::size_t size() const
    {
        return len_;
    }
    constexpr STRF_HD std::size_t length() const
    {
        return len_;
    }

private:

    const CharIn* begin_;
    const std::size_t len_;
};

} // namespace detail

template <typename CharT>
struct no_cv_format;

template <typename CharT>
struct cv_format;

template <typename CharT, typename Charset>
struct cv_format_with_charset;

template <typename CharT>
struct sani_format;

template <typename CharT, typename Charset>
struct sani_format_with_charset;

template <typename CharT, typename T>
class no_cv_format_fn
{
public:

    constexpr STRF_HD no_cv_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit no_cv_format_fn
        ( const no_cv_format_fn<CharT, U>& ) noexcept
    {
    }

    constexpr STRF_HD auto convert_charset() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_cv_format<CharT>
                                             , strf::cv_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }

    template <typename Charset>
    constexpr STRF_HD auto convert_charset(const Charset& cs) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::cv_format_with_charset<CharT, Charset> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::cv_format_with_charset<CharT, Charset>>{}
            , cs };
    }
    constexpr STRF_HD auto cv() const
    {
        return convert_charset();
    }
    template <typename Charset>
    constexpr STRF_HD auto cv(const Charset& cs) const
    {
        return convert_charset(cs);
    }

    constexpr STRF_HD auto sanitize_charset() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_cv_format<CharT>
                                             , strf::sani_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }
    template <typename Charset>
    constexpr STRF_HD auto sanitize_charset(const Charset& cs) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::sani_format_with_charset<CharT, Charset> >;

        return return_type
            { static_cast<const T&>(*this)
             , strf::tag<strf::sani_format_with_charset<CharT, Charset>>{}
            , cs };
    }
    constexpr auto sani() const
    {
        return sanitize_charset();
    }
    template <typename Charset>
    constexpr auto sani(const Charset& cs) const
    {
        return sanitize_charset(cs);
    }
};

template <typename CharT, typename T>
struct cv_format_fn
{
    constexpr STRF_HD cv_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit cv_format_fn
        ( const cv_format_fn<CharT, U>& ) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit cv_format_fn
        ( const strf::no_cv_format_fn<CharT, U>& ) noexcept
    {
    }
};

template <typename CharT, typename Charset, typename T>
class cv_format_with_charset_fn
{
public:

    cv_format_with_charset_fn(const Charset& e)
        : charset_(e)
    {
    }

    cv_format_with_charset_fn
        ( const cv_format_with_charset_fn& other ) noexcept = default;

    template <typename U>
    explicit cv_format_with_charset_fn
        ( const strf::cv_format_with_charset_fn<CharT, Charset, U>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    template <typename U>
    explicit cv_format_with_charset_fn
        ( const strf::no_cv_format_fn<CharT, U>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    const Charset& get_charset() const
    {
        return charset_;
    }

private:

    const Charset& charset_;
};

template <typename CharT>
struct no_cv_format
{
    template <typename T>
    using fn = strf::no_cv_format_fn<CharT, T>;
};

template <typename CharT>
struct cv_format
{
    template <typename T>
    using fn = strf::cv_format_fn<CharT, T>;
};

template <typename CharT, typename Charset>
struct cv_format_with_charset
{
    template <typename T>
    using fn = strf::cv_format_with_charset_fn<CharT, Charset, T>;
};

template <typename CharT>
struct sani_format
{
    template <typename T>
    using fn = strf::cv_format_fn<CharT, T>;
};

template <typename CharT, typename Charset>
struct sani_format_with_charset
{
    template <typename T>
    using fn = strf::cv_format_with_charset_fn<CharT, Charset, T>;
};

template <typename T, bool Active>
class string_precision_format_fn;

template <bool Active>
struct string_precision_format
{
    template <typename T>
    using fn = strf::string_precision_format_fn<T, Active>;
};

template <bool Active>
struct string_precision
{
};

template <>
struct string_precision<true>
{
    strf::width_t precision;
};

template <typename T>
class string_precision_format_fn<T, true>
{
public:
    constexpr STRF_HD string_precision_format_fn(strf::width_t p) noexcept
        : precision_(p)
    {
    }
    template <typename U>
    constexpr STRF_HD string_precision_format_fn
        ( strf::string_precision_format_fn<U, true> other ) noexcept
        : precision_(other.precision_)
    {
    }
    constexpr STRF_HD T&& p(strf::width_t _) && noexcept
    {
        precision_ = _;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD strf::width_t precision() const noexcept
    {
        return precision_;
    }
    constexpr STRF_HD auto get_string_precision() const noexcept
    {
        return strf::string_precision<true>{precision_};
    }

private:

    strf::width_t precision_;
};


template <typename T>
class string_precision_format_fn<T, false>
{
    using adapted_derived_type_
        = strf::fmt_replace< T
                           , strf::string_precision_format<false>
                           , strf::string_precision_format<true> >;
public:

    constexpr STRF_HD string_precision_format_fn() noexcept
    {
    }
    template <typename U>
    constexpr STRF_HD string_precision_format_fn
        ( strf::string_precision_format_fn<U, false> ) noexcept
    {
    }
    constexpr STRF_HD adapted_derived_type_ p(strf::width_t precision) const noexcept
    {
        return { static_cast<const T&>(*this)
               , strf::tag<string_precision_format<true> >{}
               , precision };
    }
    constexpr STRF_HD auto get_string_precision() const noexcept
    {
        return strf::string_precision<false>{};
    }
};

template <typename CharIn, bool HasPrecision = false, bool HasAlignment = false>
using string_with_format = strf::value_with_format
    < strf::detail::simple_string_view<CharIn>
    , strf::string_precision_format<HasPrecision>
    , strf::alignment_format_q<HasAlignment>
    , strf::no_cv_format<CharIn> >;

template <typename CharIn>
constexpr STRF_HD auto make_fmt
    ( strf::rank<1>
    , const strf::detail::simple_string_view<CharIn>& str) noexcept
{
    return strf::string_with_format<CharIn>{str};
}

template <typename CharIn, typename Traits, typename Allocator>
STRF_HD auto make_fmt
    ( strf::rank<1>
    , const std::basic_string<CharIn, Traits, Allocator>& str) noexcept
{
    return strf::string_with_format<CharIn>{{str.data(), str.size()}};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr STRF_HD auto make_fmt
    ( strf::rank<1>
    , const std::basic_string_view<CharIn, Traits>& str) noexcept
{
    return strf::string_with_format<CharIn>{{str.data(), str.size()}};
}

#endif // defined(STRF_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto make_fmt(strf::rank<1>, const char8_t* str)
{
    auto len = strf::detail::str_length<char8_t>(str);
    return strf::string_with_format<char8_t>{{str, len}};
}

#endif

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const char* str)
{
    auto len = strf::detail::str_length<char>(str);
    return strf::string_with_format<char>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const wchar_t* str)
{
    auto len = strf::detail::str_length<wchar_t>(str);
    return strf::string_with_format<wchar_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const char16_t* str)
{
    auto len = strf::detail::str_length<char16_t>(str);
    return strf::string_with_format<char16_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const char32_t* str)
{
    auto len = strf::detail::str_length<char32_t>(str);
    return strf::string_with_format<char32_t>{{str, len}};
}

namespace detail {

template <std::size_t CharSize>
class string_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT, bool HasPrecision>
    STRF_HD string_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::value_with_format
            < strf::detail::simple_string_view<CharT>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<false>
            , strf::no_cv_format<CharT> > input
        , strf::tag<CharT> t = strf::tag<CharT>{} )
        : string_printer(fp, preview, input.value(), input.get_string_precision(), t)
    {
    }

    template < typename FPack, typename Preview, typename Str, typename CharT
             , typename = decltype(strf::detail::simple_string_view<CharT>(std::declval<const Str&>())) >
    STRF_HD string_printer
        ( const FPack& fp
        , Preview& preview
        , const Str& str
        , strf::tag<CharT> t ) noexcept
        : string_printer
            ( fp, preview, strf::detail::simple_string_view<CharT>(str)
            ,  strf::string_precision<false>(), t )
    {
    }

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::string_precision<false>
        , strf::tag<CharT> ) noexcept
        : str_(reinterpret_cast<const char_type*>(str.begin()))
        , len_(str.size())
    {
        (void)fp;
        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c, CharT>(fp);
            auto w = wcalc.str_width( get_facet_<strf::charset_c<CharT>, CharT>(fp)
                                    , preview.remaining_width(), str_, len_
                                    , get_facet_<strf::surrogate_policy_c, CharT>(fp) );
            preview.subtract_width(w);
        }
        preview.add_size(len_);
    }

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::string_precision<true> sp
        , strf::tag<CharT> ) noexcept
        : str_(reinterpret_cast<const char_type*>(str.begin()))
    {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, CharT>(fp);
        decltype(auto) cs = get_facet_<strf::charset_c<CharT>, CharT>(fp);
        decltype(auto) surr_poli = get_facet_<strf::surrogate_policy_c, CharT>(fp);
        auto res = wcalc.str_width_and_pos(cs, sp.precision, str_, str.size(), surr_poli);
        len_ = res.pos;
        preview.subtract_width(res.width);
    }

    template <typename FPack, typename Preview, bool HasP, typename CharT>
    STRF_HD string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::string_precision<HasP> sp
        , strf::alignment_format_data
        , strf::tag<CharT> t) noexcept
        : string_printer(fp, preview, str, sp, t)
    {
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* str_;
    const std::size_t len_;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<std::size_t CharSize>
STRF_HD void string_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    strf::write(ob, str_, len_);
}

template <std::size_t CharSize>
class aligned_string_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT, bool HasPrecision>
    STRF_HD aligned_string_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::value_with_format
            < strf::detail::simple_string_view<CharT>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<true>
            , strf::no_cv_format<CharT> > input
        , strf::tag<CharT> t )
        : aligned_string_printer( fp, preview, input.value(), input.get_string_precision()
                                , input.get_alignment_format_data(), t )
    {
    }

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD aligned_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::alignment_format_data text_alignment
        , strf::tag<CharT> )
        : str_(reinterpret_cast<const char_type*>(str.begin()))
        , len_(str.size())
        , afmt_(text_alignment)
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, CharT>(fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, CharT>(fp))
    {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, CharT>(fp);
        decltype(auto) cs = get_facet_<strf::charset_c<CharT>, CharT>(fp);
        strf::width_t limit = ( Preview::width_required && preview.remaining_width() > afmt_.width
                              ? preview.remaining_width()
                              : afmt_.width );
        auto strw = wcalc.str_width(cs, limit, str_, len_, surr_poli_);
        encode_fill_ = cs.encode_fill_func();
        auto fillcount = init_(preview, strw);
        preview_size_(preview, cs, fillcount);
    }

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD aligned_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::string_precision<false>
        , strf::alignment_format_data text_alignment
        , strf::tag<CharT> )
        : aligned_string_printer(fp, preview, str, text_alignment, strf::tag<CharT>{})
    {
    }

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD aligned_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::string_precision<true> sp
        , strf::alignment_format_data text_alignment
        , strf::tag<CharT> )
        : str_(reinterpret_cast<const char_type*>(str.begin()))
        , afmt_(text_alignment)
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, CharT>(fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, CharT>(fp))
    {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, CharT>(fp);
        decltype(auto) cs = get_facet_<strf::charset_c<CharT>, CharT>(fp);
        auto res = wcalc.str_width_and_pos(cs, sp.precision, str_, str.size(), surr_poli_);
        len_ = res.pos;
        encode_fill_ = cs.encode_fill_func();
        auto fillcount = init_(preview, res.width);
        preview_size_(preview, cs, fillcount);
    }

    STRF_HD ~aligned_string_printer();

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* str_;
    std::size_t len_;
    strf::encode_fill_f<CharSize> encode_fill_;
    strf::alignment_format_data afmt_;
    std::int16_t left_fillcount_;
    std::int16_t right_fillcount_;
    const strf::invalid_seq_policy inv_seq_poli_;
    const strf::surrogate_policy surr_poli_;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <typename Preview>
    STRF_HD std::uint16_t init_(Preview&, strf::width_t strw);

    template <typename Charset>
    STRF_HD void preview_size_( strf::size_preview<true>& preview
                              , Charset cs, std::uint16_t fillcount )
    {
        preview.add_size(len_);
        if (fillcount > 0) {
            preview.add_size(fillcount * cs.encoded_char_size(afmt_.fill));
        }
    }

    template <typename Charset>
    STRF_HD void preview_size_(strf::size_preview<false>&, Charset, std::uint16_t)
    {
    }
};

template<std::size_t CharSize>
STRF_HD aligned_string_printer<CharSize>::~aligned_string_printer()
{
}

template<std::size_t CharSize>
template <typename Preview>
inline STRF_HD std::uint16_t aligned_string_printer<CharSize>::init_
    ( Preview& preview, strf::width_t strw )
{
    if (afmt_.width > strw) {
        std::uint16_t fillcount = (afmt_.width - strw).round();
        switch(afmt_.alignment) {
            case strf::text_alignment::left:
                left_fillcount_ = 0;
                right_fillcount_ = fillcount;
                break;
            case strf::text_alignment::center: {
                std::uint16_t halfcount = fillcount >> 1;
                left_fillcount_ = halfcount;
                right_fillcount_ = fillcount - halfcount;
                break;
            }
            default:
                left_fillcount_ = fillcount;
                right_fillcount_ = 0;
        }
        preview.subtract_width(strw + fillcount);
        return fillcount;
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        preview.subtract_width(strw);
        return 0;
    }
}

template<std::size_t CharSize>
void STRF_HD aligned_string_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_( ob, left_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }
    strf::write(ob, str_, len_);
    if (right_fillcount_ > 0) {
        encode_fill_( ob, right_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE class string_printer<1>;
STRF_EXPLICIT_TEMPLATE class string_printer<2>;
STRF_EXPLICIT_TEMPLATE class string_printer<4>;

STRF_EXPLICIT_TEMPLATE class aligned_string_printer<1>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<2>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<4>;

#endif // defined(STRF_SEPARATE_COMPILATION)

template <typename DestCharT, typename SrcCharT = DestCharT>
class string_printer_traits
{
public:
    static_assert( std::is_same<DestCharT, SrcCharT>::value
                 , "Character type mismatch. Use cv function." );
    template <typename>
    using printer_type = strf::detail::string_printer<sizeof(SrcCharT)>;
};

} // namespace detail

template <typename CharOut>
STRF_HD strf::detail::string_printer_traits<CharOut, char> get_printer_traits
(strf::tag<CharOut>, const char*);

#if defined(__cpp_char8_t)

template <typename CharOut>
STRF_HD strf::detail::string_printer_traits<CharOut, char8_t> get_printer_traits
( strf::tag<CharOut>, const char8_t* );

#endif // defined(__cpp_char8_t)

template <typename CharOut>
STRF_HD strf::detail::string_printer_traits<CharOut, char16_t> get_printer_traits
( strf::tag<CharOut>, const char16_t* );

template <typename CharOut>
STRF_HD strf::detail::string_printer_traits<CharOut, char32_t> get_printer_traits
( strf::tag<CharOut>, const char32_t* );

template <typename CharOut>
STRF_HD strf::detail::string_printer_traits<CharOut, wchar_t> get_printer_traits
( strf::tag<CharOut>, const wchar_t* );

template <typename CharOut, typename CharIn, typename Traits, typename Allocator>
STRF_HD strf::detail::string_printer_traits<CharOut, CharIn> get_printer_traits
( strf::tag<CharOut>, const std::basic_string<CharIn, Traits, Allocator>& );

template <typename CharOut, typename CharIn>
STRF_HD strf::detail::string_printer_traits<CharOut, CharIn> get_printer_traits
( strf::tag<CharOut>, strf::detail::simple_string_view<CharIn> );

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharOut, typename CharIn, typename Traits>
STRF_HD strf::detail::string_printer_traits<CharOut, CharIn> get_printer_traits
( strf::tag<CharOut>, std::basic_string_view<CharIn, Traits> );

#endif //defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharOut, typename CharIn, bool HasPrecision, bool HasAlignment>
class printer_traits
    < CharOut
    , strf::value_with_format
        < strf::detail::simple_string_view<CharIn>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<HasAlignment>
        , strf::no_cv_format<CharIn> > >
{
public:
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );

    template <typename>
    using printer_type = std::conditional_t
        < HasAlignment
        , strf::detail::aligned_string_printer<sizeof(CharOut)>
        , strf::detail::string_printer<sizeof(CharOut)> >;

};

} // namespace strf

#endif  /* STRF_DETAIL_INPUT_TYPES_CHAR_PTR */

