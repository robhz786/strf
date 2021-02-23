//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#include <array>
#include <tuple>

template <typename T>
constexpr STRF_TEST_FUNC auto as_signed(const T& value)
{
    return static_cast<typename std::make_signed<T>::type>(value);
}
STRF_TEST_FUNC strf::detail::simple_string_view<char>
valid_input_sample(const strf::utf<char>&)
{
    return {(const char*)u8"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 19};
}

STRF_TEST_FUNC strf::detail::simple_string_view<char16_t>
valid_input_sample(const strf::utf<char16_t>&)
{
    return {u"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 10};
}

STRF_TEST_FUNC strf::detail::simple_string_view<char32_t>
valid_input_sample(const strf::utf<char32_t>&)
{
    return {U"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 8};
}

STRF_TEST_FUNC strf::detail::simple_string_view<wchar_t>
valid_input_sample(const strf::utf<wchar_t>&)
{
    return {L"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", (sizeof(wchar_t) == 2 ? 10 : 8)};
}

template <typename SrcEncoding, typename DestEncoding>
STRF_TEST_FUNC void test_valid_input(SrcEncoding src_enc, DestEncoding dest_enc)
{
    TEST_SCOPE_DESCRIPTION("from ", src_enc.name(), " to ", dest_enc.name());

    auto input = valid_input_sample(src_enc);
    auto expected = valid_input_sample(dest_enc);
    TEST(expected).with(dest_enc) (strf::sani(input, src_enc));
}

STRF_TEST_FUNC strf::detail::simple_string_view<char>
sample_with_surrogates(const strf::utf<char>&)
{
    return " \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF";
}
STRF_TEST_FUNC strf::detail::simple_string_view<char16_t>
sample_with_surrogates(const strf::utf<char16_t>&)
{
    static const char16_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
STRF_TEST_FUNC strf::detail::simple_string_view<char32_t>
sample_with_surrogates(const strf::utf<char32_t>&)
{
    static const char32_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
STRF_TEST_FUNC strf::detail::simple_string_view<wchar_t>
sample_with_surrogates(const strf::utf<wchar_t>&)
{
    static const wchar_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}

template <typename SrcEncoding, typename DestEncoding>
STRF_TEST_FUNC void test_allowed_surrogates(SrcEncoding src_enc, DestEncoding dest_enc)
{
    TEST_SCOPE_DESCRIPTION("from ", src_enc.name()," to ", dest_enc.name());

    const auto input    = sample_with_surrogates(src_enc);
    const auto expected = sample_with_surrogates(dest_enc);

    TEST(expected)
        .with( dest_enc, strf::surrogate_policy::lax )
        (strf::sani(input, src_enc));
}

template <typename CharT>
struct invalid_seq
{
    int errors_count;
    strf::detail::simple_string_view<CharT> sequence;
};

template <typename T, std::size_t N>
struct array
{
    STRF_TEST_FUNC const T* begin() const noexcept { return &elements_[0]; };
    STRF_TEST_FUNC const T* end()  const noexcept  { return begin() + N; };

    T elements_[N];
};

STRF_TEST_FUNC auto invalid_sequences(const strf::utf<char>&)
{
    // based on https://www.unicode.org/versions/Unicode10.0.0/ch03.pdf
    // "Best Practices for Using U+FFFD"
    return array<invalid_seq<char>, 17>
       {{ {3, "\xF1\x80\x80\xE1\x80\xC0"} // sample from Tabble 3-8 of Unicode standard
        , {2, "\xC1\xBF"}                 // overlong sequence
        , {3, "\xE0\x9F\x80"}             // overlong sequence
        , {3, "\xC1\xBF\x80"}             // overlong sequence with extra continuation bytes
        , {4, "\xE0\x9F\x80\x80"}         // overlong sequence with extra continuation bytes
        , {1, "\xC2"}                     // missing continuation
        , {1, "\xE0\xA0"}                 // missing continuation
        , {3, "\xED\xA0\x80"}             // surrogate
        , {3, "\xED\xAF\xBF"}             // surrogate
        , {3, "\xED\xB0\x80"}             // surrogate
        , {3, "\xED\xBF\xBF"}             // surrogate
        , {5, "\xED\xBF\xBF\x80\x80"}     // surrogate with extra continuation bytes
        , {4, "\xF0\x8F\xBF\xBF" }        // overlong sequence
        , {5, "\xF0\x8F\xBF\xBF\x80" }    // overlong sequence with extra continuation bytes
        , {1, "\xF0\x90\xBF" }            // missing continuation
        , {4, "\xF4\xBF\xBF\xBF"}         // codepoint too big
        , {6, "\xF5\x90\x80\x80\x80\x80"} // codepoint too big with extra continuation bytes
        }};
}

STRF_TEST_FUNC auto invalid_sequences(const strf::utf<char16_t>&)
{
    static const char16_t ch[] = {0xDFFF, 0xDC00, 0xD800, 0xDBFF};
    return array<invalid_seq<char16_t>, 5>
       {{ {1, {&ch[0], 1}}
        , {1, {&ch[1], 1}}
        , {1, {&ch[2], 1}}
        , {1, {&ch[3], 1}}
        , {2, {&ch[1], 2}} }};
}

STRF_TEST_FUNC auto invalid_sequences(const strf::utf<char32_t>&)
{
    static const char32_t ch[] = {0xD800, 0xDBFF,0xDC00,0xDFFF, 0x110000};
    return array<invalid_seq<char32_t>, 5>
       {{ {1, {&ch[0], 1}}
        , {1, {&ch[1], 1}}
        , {1, {&ch[2], 1}}
        , {1, {&ch[3], 1}}
        , {1, {&ch[4], 1}} }};
}

STRF_TEST_FUNC auto invalid_sequences(const strf::utf<wchar_t>&)
{
    static const wchar_t ch[] = { 0xD800, 0xDBFF, 0xDC00, 0xDFFF
                                , (sizeof(wchar_t) == 4 ? 0x110000 : 0xDFFF) };
    return array<invalid_seq<wchar_t>, 5>
       {{ {1, {&ch[0], 1}}
        , {1, {&ch[1], 1}}
        , {1, {&ch[2], 1}}
        , {1, {&ch[3], 1}}
        , {1, {&ch[4], 1}} }};
}

inline STRF_TEST_FUNC strf::detail::simple_string_view<char>
replacement_char(const strf::utf<char>&){ return (const char*)u8"\uFFFD";}

inline STRF_TEST_FUNC strf::detail::simple_string_view<char16_t>
replacement_char(const strf::utf<char16_t>&){ return u"\uFFFD";}

inline STRF_TEST_FUNC strf::detail::simple_string_view<char32_t>
replacement_char(const strf::utf<char32_t>&){ return U"\uFFFD";}

inline STRF_TEST_FUNC strf::detail::simple_string_view<wchar_t>
replacement_char(const strf::utf<wchar_t>&){ return L"\uFFFD";}


template <typename CharT>
strf::detail::simple_string_view<CharT> STRF_TEST_FUNC concatenate
    ( CharT* buff
    , const CharT(&prefix)[3]
    , strf::detail::simple_string_view<CharT> str
    , std::size_t count
    , const CharT(&suffix)[3] )
{
    buff[0] = prefix[0];
    buff[1] = prefix[1];
    buff[2] = prefix[2];
    auto it = buff + 3;
    for (std::size_t i = 0; i < count; ++i) {
        strf::detail::copy_n(str.begin(), str.size(), it);
        it += str.size();
    }
    it[0] = suffix[0];
    it[1] = suffix[1];
    it[2] = suffix[2];

    return {buff, it + 3};
}

template <class>
struct get_first_template_parameter_impl;

template
    < class T0
    , strf::char_encoding_id T1
    , template <class, strf::char_encoding_id> class Tmpl >
struct get_first_template_parameter_impl<Tmpl<T0, T1>>
{
    using type = T0;
};

template <class T>
using get_first_template_parameter
= typename get_first_template_parameter_impl<T>::type;

static STRF_TEST_FUNC bool encoding_error_handler_called = false ;

void STRF_TEST_FUNC  encoding_error_handler()
{
    encoding_error_handler_called = true;
}

template <typename SrcEncoding, typename DestEncoding>
void STRF_TEST_FUNC test_invalid_input(SrcEncoding src_enc, DestEncoding dest_enc)
{
    TEST_SCOPE_DESCRIPTION("From invalid ", src_enc.name(), " to ", dest_enc.name());
    using src_char_type  = get_first_template_parameter<SrcEncoding>;
    using dest_char_type = get_first_template_parameter<DestEncoding>;

    const src_char_type  suffix_in  [] = { 'd', 'e', 'f' };
    const dest_char_type suffix_out [] = { 'd', 'e', 'f' };
    const src_char_type  prefix_in  [] = { 'a', 'b', 'c' };
    const dest_char_type prefix_out [] = { 'a', 'b', 'c' };

    for(const auto& s : invalid_sequences(src_enc))
    {
        const int err_count = s.errors_count;
        const auto& seq = s.sequence;

        auto f = [](auto ch){
            return *strf::hex((unsigned)(std::make_unsigned_t<src_char_type>)ch);
        };

#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

        TEST_SCOPE_DESCRIPTION
            .with(strf::lettercase::mixed)
            ( "Sequence = ", strf::separated_range(seq, " ", f) );

#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
#  pragma GCC diagnostic pop
#endif

        src_char_type buff_in[20];
        dest_char_type buff_out[80];
        auto input = concatenate(buff_in, prefix_in, seq, 1, suffix_in);

        {   // replace
            auto expected = concatenate( buff_out
                                       , prefix_out
                                       , replacement_char(dest_enc)
                                       , err_count
                                       , suffix_out );
            TEST(expected)
                .with(dest_enc)
                (strf::sani(input, src_enc));
        }

        {
            ::encoding_error_handler_called = false;
            strf::to(buff_out)
                .with(dest_enc, strf::invalid_seq_notifier{encoding_error_handler})
                (strf::sani(input, src_enc));
            TEST_TRUE(::encoding_error_handler_called);
        }
    }
}


template < typename Func
         , typename SrcEncoding >
void STRF_TEST_FUNC combine_3(Func, SrcEncoding)
{
}

template < typename Func
         , typename SrcEncoding
         , typename DestEncoding0
         , typename ... DestEncoding >
void STRF_TEST_FUNC combine_3
    ( Func func
    , SrcEncoding src_enc
    , DestEncoding0 dest_enc0
    , DestEncoding ... dest_enc )
{
    func(src_enc, dest_enc0);
    combine_3(func, src_enc, dest_enc...);
}

template < typename Func
         , typename Encoding0 >
void STRF_TEST_FUNC combine_2(Func func, Encoding0 cs0)
{
    combine_3(func, cs0, cs0);
}

template < typename Func
         , typename Encoding0
         , typename ... Enc >
void STRF_TEST_FUNC combine_2(Func func, Encoding0 cs0, Enc... enc)
{
    combine_3(func, cs0, cs0, enc...);
}

template < typename Func
         , typename Dest_EncTuple
         , std::size_t ... I >
void STRF_TEST_FUNC combine(Func, const Dest_EncTuple&, std::index_sequence<I...> )
{
}

template < typename Func
         , typename DestEncTuple
         , std::size_t ... I
         , typename Encoding0
         , typename ... Enc >
void STRF_TEST_FUNC combine
    ( Func func
    , const DestEncTuple& dest_encodings
    , std::index_sequence<I...> iseq
    , Encoding0 cs0
    , Enc ... enc )

{
    combine_2(func, cs0, std::get<I>(dest_encodings)...);
    combine(func, dest_encodings, iseq, enc...);
}

template < typename Func, typename Tuple, std::size_t ... I >
void STRF_TEST_FUNC for_all_combinations(Func func, const Tuple& encodings, std::index_sequence<I...> iseq)
{
    combine(func, encodings, iseq, std::get<I>(encodings)...);
}

template < typename Tuple, typename Func >
void STRF_TEST_FUNC for_all_combinations(const Tuple& encodings, Func func)
{
    constexpr std::size_t tsize = std::tuple_size<Tuple>::value;
    for_all_combinations(func, encodings, std::make_index_sequence<tsize>());
}

void STRF_TEST_FUNC test_utf_to_utf()
{
    const auto encodings = std::make_tuple
        ( strf::utf<char>(), strf::utf<char16_t>()
        , strf::utf<char32_t>(), strf::utf<wchar_t>());

    for_all_combinations
        ( encodings
        , [](auto src_enc, auto dest_enc){ test_valid_input(src_enc, dest_enc); } );

    for_all_combinations
        ( encodings
        , [](auto src_enc, auto dest_enc){ test_allowed_surrogates(src_enc, dest_enc); } );

    for_all_combinations
        ( encodings
        , [](auto src_enc, auto dest_enc){ test_invalid_input(src_enc, dest_enc); } );

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char, strf::eid_utf8, strf::eid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf<char>()
                                                   , strf::utf<char>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char16_t, strf::eid_utf8, strf::eid_utf16 >
                   , decltype(strf::find_transcoder( strf::utf<char>()
                                                   , strf::utf<char16_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char32_t, strf::eid_utf8, strf::eid_utf32 >
                   , decltype(strf::find_transcoder( strf::utf<char>()
                                                   , strf::utf<char32_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char, strf::eid_utf16, strf::eid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf<char16_t>()
                                                   , strf::utf<char>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char16_t, strf::eid_utf16, strf::eid_utf16 >
                   , decltype(strf::find_transcoder( strf::utf<char16_t>()
                                                   , strf::utf<char16_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char32_t, strf::eid_utf16, strf::eid_utf32 >
                   , decltype(strf::find_transcoder( strf::utf<char16_t>()
                                                   , strf::utf<char32_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char, strf::eid_utf32, strf::eid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf<char32_t>()
                                                   , strf::utf<char>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char16_t,  strf::eid_utf32, strf::eid_utf16 >
                   , decltype(strf::find_transcoder( strf::utf<char32_t>()
                                                   , strf::utf<char16_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char32_t, strf::eid_utf32, strf::eid_utf32 >
                   , decltype(strf::find_transcoder( strf::utf<char32_t>()
                                                   , strf::utf<char32_t>())) >
                  :: value));

}
