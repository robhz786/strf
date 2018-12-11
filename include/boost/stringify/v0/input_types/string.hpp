#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STRING
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>
#include <boost/stringify/v0/facets/encoding.hpp>
#include <boost/utility/string_view.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;

template <typename CharIn>
struct string_format
{
    template <typename T>
    class fn
    {
    public:

        template <typename>
        friend class fn;

        constexpr fn() = default;

        template <typename U>
        fn(const fn<U>& u)
            : _encoding(u._encoding)
            , _sani(u._sani)
        {
        }

        constexpr T& sani(bool s = true) &
        {
            _sani = s;
            return *this;
        }
        constexpr T&& sani(bool s = true) &&
        {
            _sani = s;
            return static_cast<T&&>(*this);
        }
        constexpr T& encoding(const stringify::v0::encoding<CharIn>& e) &
        {
            _encoding = & e;
            return *this;
        }
        constexpr T&& encoding(const stringify::v0::encoding<CharIn>& e) &&
        {
            _encoding = & e;
            return static_cast<T&&>(*this);
        }
        constexpr bool get_sani() const
        {
            return _sani;
        }
        bool has_encoding() const
        {
            return  _encoding != nullptr;
        }
        const stringify::v0::encoding<CharIn>& encoding() const
        {
            BOOST_ASSERT(has_encoding());
            return *_encoding;
        }

    private:

        const stringify::v0::encoding<CharIn>* _encoding = nullptr;
        bool _sani = false;
    };
};


template <typename CharIn>
class simple_string_view
{
public:

    simple_string_view(const CharIn* str, std::size_t len) noexcept
        : _begin(str)
        , _end(str + len)
    {
    }
    simple_string_view(const CharIn* str) noexcept
        : _begin(str)
        , _end(str + std::char_traits<CharIn>::length(str))
    {
    }
    constexpr const CharIn* begin() const
    {
        return _begin;
    }
    constexpr const CharIn* end() const
    {
        return _end;
    }

private:

    const CharIn* _begin;
    const CharIn* _end;
};

// namespace detail
// {

// template <CharIn, CharOut>
// stringify::v0::expected_buff_it<CharOut> decode_encode
//     ( const stringify::v0::encoding<CharIn>&  _src_encoding
//     , const stringify::v0::encoding<CharOut>& _dest_encoding
//     , const CharIn* src
//     , const CharIn* src_end
//     , stringify::v0::buff_it<CharOut> buff
//     , stringify::buffer_recycler<CharOut>& recycler
//     , const stringify::v0::error_handling err_hdl
//     , bool allow_surr )
// {
// }

// } // namespace detail



template <typename CharIn>
using string_with_format = stringify::v0::value_with_format
    < stringify::v0::simple_string_view<CharIn>
    , stringify::v0::string_format<CharIn>
    , stringify::v0::alignment_format >;

template<typename CharIn, typename CharOut>
class simple_string_printer: public stringify::v0::printer<CharOut>
{
    using input_tag = stringify::v0::string_input_tag<CharIn>;

public:

    template <typename FPack>
    simple_string_printer
        ( const FPack& fp
        , const CharIn* str
        , std::size_t len ) noexcept
        : simple_string_printer
            ( str
            , str + len
            , get_facet<stringify::v0::width_calculator_category>(fp)
            , get_facet<stringify::v0::encoding_category<CharIn>>(fp)
            , get_facet<stringify::v0::encoding_category<CharOut>>(fp)
            , get_facet<stringify::v0::encoding_policy_category>(fp) )
    {
    }

    simple_string_printer
        ( const CharIn* begin
        , const CharIn* end
        , const stringify::v0::width_calculator& wcalc
        , const stringify::v0::encoding<CharIn>& src_enc
        , const stringify::v0::encoding<CharOut>& dest_enc
        , const stringify::v0::encoding_policy epoli ) noexcept
        : _begin(begin)
        , _end(end)
        , _wcalc(wcalc)
        , _src_encoding(src_enc)
        , _dest_encoding(dest_enc)
        , _epoli(epoli)
    {
    }

    ~simple_string_printer() = default;

    std::size_t necessary_size() const override;

    stringify::v0::expected_buff_it<CharOut> write
        ( stringify::v0::buff_it<CharOut> buff
        , stringify::buffer_recycler<CharOut>& recycler ) const override;

    int remaining_width(int w) const override;

private:

    const CharIn* const _begin;
    const CharIn* const _end;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding<CharIn>&  _src_encoding;
    const stringify::v0::encoding<CharOut>& _dest_encoding;
    const stringify::v0::encoding_policy _epoli;

    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharIn, typename CharOut>
std::size_t simple_string_printer<CharIn, CharOut>::necessary_size() const
{
    const auto * cv = get_transcoder(_src_encoding, _dest_encoding);
    auto err_hdl = _epoli.err_hdl();
    bool allow_surr = _epoli.allow_surr();
    if (cv)
    {
        return cv->necessary_size(_begin, _end, err_hdl, allow_surr);
    }

    char32_t buff[16];
    char32_t * const buff_end = buff + sizeof(buff)/sizeof(buff[0]);
    std::size_t count = 0;
    stringify::v0::cv_result res_dec;
    auto src_it = _begin;
    do
    {
        auto buff_it = & buff[0];
        res_dec = _src_encoding.to_u32.transcode( &src_it, _end
                                                , &buff_it, buff_end
                                                , err_hdl, allow_surr );
        count += _dest_encoding.from_u32.necessary_size(buff, buff_it, err_hdl, allow_surr);
    } while(res_dec == stringify::v0::cv_result::insufficient_space);

    return count;
}

template<typename CharIn, typename CharOut>
stringify::v0::expected_buff_it<CharOut>
simple_string_printer<CharIn, CharOut>::write
    ( stringify::v0::buff_it<CharOut> buff
    , stringify::buffer_recycler<CharOut>& recycler ) const
{
    const auto * cv = get_transcoder(_src_encoding, _dest_encoding);
    auto err_hdl = _epoli.err_hdl();
    bool allow_surr = _epoli.allow_surr();
    if (cv)
    {
        auto src_it = _begin;
        stringify::v0::cv_result res;
        while(true)
        {
            res = cv->transcode(&src_it, _end, &buff.it, buff.end, err_hdl, allow_surr);
            if (res == stringify::v0::cv_result::success)
            {
                return { stringify::v0::in_place_t{}, buff };
            }
            if (res == stringify::v0::cv_result::invalid_char)
            {
                return { stringify::v0::unexpect_t{}
                       , std::make_error_code(std::errc::result_out_of_range) };
            }
            BOOST_ASSERT(res == stringify::v0::cv_result::insufficient_space);
        }
    }
    else
    {
        char32_t buff32[16];
        char32_t * const buff32_end = buff32 + sizeof(buff32)/sizeof(buff32[0]);

        stringify::v0::cv_result res1;
        auto src_it = _begin;
        do
        {
            auto buff32_it = & buff32[0];
            res1 = _src_encoding.to_u32.transcode( &src_it, _end
                                                 , &buff32_it, buff32_end
                                                 , err_hdl, allow_surr );
            if (res1 == stringify::v0::cv_result::invalid_char)
            {
                return { stringify::v0::unexpect_t{}
                       , std::make_error_code(std::errc::result_out_of_range) };
            }
            const auto* buff32_it2 = & buff32[0];
            auto res2 = _dest_encoding.from_u32.transcode( &buff32_it2, buff32_it
                                                         , &buff.it, buff.end
                                                         , err_hdl, allow_surr );
            while (res2 == stringify::v0::cv_result::insufficient_space)
            {
                auto x = recycler.recycle(buff.it);
                BOOST_STRINGIFY_RETURN_ON_ERROR(x);
                buff = *x;
                res2 = _dest_encoding.from_u32.transcode( &buff32_it2, buff32_it
                                                        , &buff.it, buff.end
                                                        , err_hdl, allow_surr );
            }
            if (res2 == stringify::v0::cv_result::invalid_char)
            {
                return { stringify::v0::unexpect_t{}
                       , std::make_error_code(std::errc::result_out_of_range) };
            }
        } while (res1 == stringify::v0::cv_result::insufficient_space);

        return { stringify::v0::in_place_t{}, buff };
    }
}

template<typename CharIn, typename CharOut>
int simple_string_printer<CharIn, CharOut>::remaining_width(int w) const
{
    return _wcalc.remaining_width(w, _begin, _end, _src_encoding, _epoli);
}

template <typename CharT>
class simple_string_printer<CharT, CharT>: public stringify::v0::printer<CharT>
{
    using CharIn = CharT;
    using CharOut = CharT;
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using traits = std::char_traits<CharT>;

public:

    template <typename FPack>
    simple_string_printer
        ( const FPack& fp
        , const CharIn* str
        , std::size_t len
        ) noexcept
        : _str(str)
        , _len(len)
        , _wcalc(get_facet<stringify::v0::width_calculator_category>(fp))
        , _encoding(get_facet<stringify::v0::encoding_category<CharT>>(fp))
        , _epoli(get_facet<stringify::v0::encoding_policy_category>(fp))
    {
    }

    ~simple_string_printer() = default;

    std::size_t necessary_size() const override;

    stringify::v0::expected_buff_it<CharT> write
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const override;

    int remaining_width(int w) const override;

private:

    const CharIn* _str;
    const std::size_t _len;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding<CharT>& _encoding;
    const stringify::v0::encoding_policy  _epoli;

    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharT>
std::size_t simple_string_printer<CharT, CharT>::necessary_size() const
{
    return _len;
}

template<typename CharT>
stringify::v0::expected_buff_it<CharT>
simple_string_printer<CharT, CharT>::write
    ( stringify::v0::buff_it<CharT> buff
    , stringify::buffer_recycler<CharT>& recycler ) const
{
    auto remaining_len = _len;
    while (true)
    {
        std::size_t space = buff.end - buff.it;
        if (remaining_len <= space)
        {
            traits::copy(buff.it, _str, remaining_len);
            return { stringify::v0::in_place_t{}
                   , stringify::v0::buff_it<CharT>{buff.it + remaining_len, buff.end} };
        }
        traits::copy(buff.it, _str, space);
        remaining_len -= space;
        auto x = recycler.recycle(buff.it + space);
        BOOST_STRINGIFY_RETURN_ON_ERROR(x);
        buff = *x;
    }
}

template<typename CharT>
int simple_string_printer<CharT, CharT>::remaining_width(int w) const
{
    return _wcalc.remaining_width(w, _str, _str + _len, _encoding, _epoli);
}

// template<typename CharIn, typename CharOut>
// class string_printer: public printer<CharOut>
// {
// private:

//     using input_tag = stringify::v0::string_input_tag<CharIn>;

// public:

//     template <typename FPack>
//     string_printer
//         ( stringify::v0::output_writer<CharOut>& out
//         , const FPack& fp
//         , const stringify::v0::string_with_format<CharIn>& input )
//         noexcept
//         : string_printer
//             ( out
//             , input
//             , get_facet<stringify::v0::encoding_category<CharIn>>(fp)
//             , get_facet<stringify::v0::width_calculator_category>(fp) )
//     {
//     }

//     string_printer
//         ( stringify::v0::output_writer<CharOut>& out
//         , const stringify::v0::string_with_format<CharIn>& input
//         , const stringify::v0::encoding<CharIn> input_enc
//         , const stringify::v0::width_calculator& wcalc )
//         noexcept;

//     ~string_printer();

//     std::size_t necessary_size() const override;

//     void write() const override;

//     int remaining_width(int w) const override;

// private:

//     const stringify::v0::string_with_format<CharIn> _fmt;
//     const stringify::v0::string_writer<CharIn, CharOut> _sw;
//     const stringify::v0::decoder<CharIn>& _decoder;
//     const stringify::v0::width_calculator _wcalc;
//     const int _fillcount = 0;

//     template <typename Category, typename FPack>
//     const auto& get_facet(const FPack& fp) const
//     {
//         return fp.template get_facet<Category, input_tag>();
//     }

//     void write_string() const
//     {
//         _sw.write(_fmt.value().begin(), _fmt.value().end());
//     }

//     void write_fill(int count) const
//     {
//         _sw.put32(count, _fmt.fill());
//     }
// };

// template<typename CharIn, typename CharOut>
// string_printer<CharIn, CharOut>::string_printer
//     ( stringify::v0::output_writer<CharOut>& out
//     , const stringify::v0::string_with_format<CharIn>& input
//     , const stringify::v0::encoding<CharIn> input_enc
//     , const stringify::v0::width_calculator& wcalc
//     ) noexcept
//     : _fmt(input)
//     , _sw
//         ( out
//         , input.has_encoding() ? input.encoding() : input_enc
//         , input.get_sani() )
//     , _decoder(input_enc.decoder())
//     , _wcalc(wcalc)
//     , _fillcount
//         ( input.width() > 0
//         ? wcalc.remaining_width
//             ( input.width()
//             , input.value().begin()
//             , input.value().end()
//             , input_enc.decoder()
//             , out.on_encoding_error()
//             , out.allow_surrogates() )
//         : 0 )
// {
// }

// template<typename CharIn, typename CharOut>
// string_printer<CharIn, CharOut>::~string_printer()
// {
// }


// template<typename CharIn, typename CharOut>
// std::size_t string_printer<CharIn, CharOut>::necessary_size() const
// {
//     std::size_t len = _sw.necessary_size( _fmt.value().begin()
//                                          , _fmt.value().end() );

//     if (_fillcount > 0)
//     {
//         len += _fillcount * _sw.necessary_size(_fmt.fill());
//     }
//     return len;
// }


// template<typename CharIn, typename CharOut>
// void string_printer<CharIn, CharOut>::write() const
// {
//     if (_fillcount > 0)
//     {
//         switch(_fmt.alignment())
//         {
//             case stringify::v0::alignment::left:
//             {
//                 write_string();
//                 write_fill(_fillcount);
//                 break;
//             }
//             case stringify::v0::alignment::center:
//             {
//                 int halfcount = _fillcount / 2;
//                 write_fill(halfcount);
//                 write_string();
//                 write_fill(_fillcount - halfcount);
//                 break;
//             }
//             default:
//             {
//                 write_fill(_fillcount);
//                 write_string();
//             }
//         }
//     }
//     else
//     {
//         write_string();
//     }
// }


// template<typename CharIn, typename CharOut>
// int string_printer<CharIn, CharOut>::remaining_width(int w) const
// {
//     if (w > _fillcount)
//     {
//         return _wcalc.remaining_width
//             ( w - _fillcount
//             , _fmt.value().begin()
//             , _fmt.value().end()
//             , _decoder
//             , _sw.on_encoding_error()
//             , _sw.allow_surrogates() );
//     }
//     return 0;
// }


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, wchar_t>;

// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, char>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, char16_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, char32_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, wchar_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, char>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, char16_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, char32_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, wchar_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, char>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, char16_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, char32_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, wchar_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, char>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, char16_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, char32_t>;
// BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

template
    < typename CharOut
    , typename FPack
    , typename CharIn
    , typename Traits
    , typename Allocator >
inline stringify::v0::simple_string_printer<CharIn, CharOut>
make_printer
   ( const FPack& fp
   , const std::basic_string<CharIn, Traits, Allocator>& str )
{
    return {fp, str.data(), str.size()};
}

template
    < typename CharOut
    , typename FPack
    , typename CharIn
    , typename Traits >
inline stringify::v0::simple_string_printer<CharIn, CharOut>
make_printer
   ( const FPack& fp
   , const boost::basic_string_view<CharIn, Traits>& str )
{
    return {fp, str.data(), str.size()};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<char, CharOut>
make_printer
   ( const FPack& fp
   , const char* str )
{
    return {fp, str, std::char_traits<char>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<wchar_t, CharOut>
make_printer
   ( const FPack& fp
   , const wchar_t* str )
{
    return {fp, str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<char16_t, CharOut>
make_printer
   ( const FPack& fp
   , const char16_t* str )
{
    return {fp, str, std::char_traits<char16_t>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<char32_t, CharOut>
make_printer
   ( const FPack& fp
   , const char32_t* str )
{
    return {fp, str, std::char_traits<char32_t>::length(str)};
}


template <typename CharIn, typename Traits>
inline auto
make_fmt(stringify::v0::tag, const std::basic_string<CharIn, Traits>& str)
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

template <typename CharIn, typename Traits>
inline auto
make_fmt(stringify::v0::tag, const boost::basic_string_view<CharIn, Traits>& str)
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

inline auto make_fmt(stringify::v0::tag, const char* str)
{
    auto len = std::char_traits<char>::length(str);
    return stringify::v0::string_with_format<char>{{str, len}};
}
inline auto make_fmt(stringify::v0::tag, const wchar_t* str)
{
    auto len = std::char_traits<wchar_t>::length(str);
    return stringify::v0::string_with_format<wchar_t>{{str, len}};
}
inline auto make_fmt(stringify::v0::tag, const char16_t* str)
{
    auto len = std::char_traits<char16_t>::length(str);
    return stringify::v0::string_with_format<char16_t>{{str, len}};
}
inline auto make_fmt(stringify::v0::tag, const char32_t* str)
{
    auto len = std::char_traits<char32_t>::length(str);
    return stringify::v0::string_with_format<char32_t>{{str, len}};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FPack, typename CharIn, typename Traits>
inline stringify::v0::simple_string_printer<CharIn, CharOut>
make_printer
   ( const FPack& fp
   , const std::basic_string_view<CharIn, Traits>& str )
{
    return {fp, str.data(), str.size()};
}

template <typename CharIn, typename Traits>
inline auto
make_fmt(stringify::v0::tag, const std::basic_string_view<CharIn, Traits>& str)
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

#endif //defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

// template <typename CharOut, typename FPack, typename CharIn>
// inline stringify::v0::string_printer<CharIn, CharOut>
// make_printer
//     ( const FPack& fp
//     , const stringify::v0::string_with_format<CharIn>& ch )
// {
//     return {fp, ch};
// }

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR */

