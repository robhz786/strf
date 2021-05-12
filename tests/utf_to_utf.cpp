//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#include <tuple>

namespace {

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
    // static const char arr[] = 
        // { ' ', 0xED, 0xA0, 0x80, ' ', 0xED, 0xAF, 0xBF, ' '
        // , 0xED, 0xB0, 0x80, ' ', 0xED, 0xBF, 0xBF };
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
    STRF_TEST_FUNC const T* end()   const noexcept { return begin() + N; };

    T elements_[N];
};

template <typename T>
struct array<T, 0>
{
    constexpr STRF_TEST_FUNC const T* begin() const noexcept { return nullptr; };
    constexpr STRF_TEST_FUNC const T* end()   const noexcept { return nullptr; };
};


STRF_TEST_FUNC auto invalid_sequences(strf::utf<char>)
{
    // based on https://www.unicode.org/versions/Unicode10.0.0/ch03.pdf
    // "Best Practices for Using U+FFFD"
    return array<invalid_seq<char>, 19>
       {{ {3, "\xF1\x80\x80\xE1\x80\xC0"} // sample from Tabble 3-8 of Unicode standard
        , {1, "\xBF"}                     // missing leading byte
        , {2, "\x80\x80"}                 // missing leading byte
        , {2, "\xC1\xBF"}                 // overlong sequence
        , {3, "\xE0\x9F\x80"}             // overlong sequence
        , {3, "\xC1\xBF\x80"}             // overlong sequence with extra continuation bytes
        , {4, "\xE0\x9F\x80\x80"}         // overlong sequence with extra continuation bytes
        , {1, "\xC2"}                     // missing continuation
        , {1, "\xE0"}                     // missing continuation
        , {1, "\xE0\xA0"}                 // missing continuation
        , {1, "\xE0\xA0"}                 // missing continuation
        , {1, "\xF1"}                     // missing continuation
        , {1, "\xF1\x81"}                 // missing continuation
        , {1, "\xF1\x81\x81"}             // missing continuation
        , {4, "\xF0\x8F\xBF\xBF" }        // overlong sequence
        , {5, "\xF0\x8F\xBF\xBF\x80" }    // overlong sequence with extra continuation bytes
        , {1, "\xF0\x90\xBF" }            // missing continuation
        , {4, "\xF4\xBF\xBF\xBF"}         // codepoint too big
        , {6, "\xF5\x90\x80\x80\x80\x80"} // codepoint too big with extra continuation bytes
        }};
}

STRF_TEST_FUNC auto surrogates_sequences(strf::utf<char>)
{
    return array<invalid_seq<char>, 6>
       {{ {3, "\xED\xA0\x80"}             // surrogate
        , {3, "\xED\xAF\xBF"}             // surrogate
        , {3, "\xED\xB0\x80"}             // surrogate
        , {3, "\xED\xBF\xBF"}             // surrogate
        , {5, "\xED\xBF\xBF\x80\x80"}
        , {2, "\xED\xBF"}     // missing continuation ( but could only be a surrogate )
       }}; // surrogate with extra continuation bytes
}

template <typename CharT, std::enable_if_t<sizeof(CharT) == 2, int> = 0>
STRF_TEST_FUNC auto invalid_sequences(const strf::utf<CharT>&)
{
    return array<invalid_seq<CharT>, 0>{};
}

template <typename CharT, std::enable_if_t<sizeof(CharT) == 4, int> = 0>
STRF_TEST_FUNC auto invalid_sequences(const strf::utf<CharT>&)
{
    static const CharT ch = 0x110000;
    return array<invalid_seq<CharT>, 1> {{ {1, {&ch, 1}} }};
}

template < typename CharT
         , std::enable_if_t<sizeof(CharT) == 2 || sizeof(CharT) == 4, int> = 0 >
STRF_TEST_FUNC auto surrogates_sequences(const strf::utf<CharT>&)
{
    static const CharT ch[] = {0xDFFF, 0xDC00, 0xD800, 0xDBFF};
    return array<invalid_seq<CharT>, 5>
       {{ {1, {&ch[0], 1}}
        , {1, {&ch[1], 1}}
        , {1, {&ch[2], 1}}
        , {1, {&ch[3], 1}}
        , {2, {&ch[1], 2}} }};
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
    , strf::detail::simple_string_view<CharT> str_to_be_repeated
    , std::size_t count )
{
    buff[0] = prefix[0];
    buff[1] = prefix[1];
    buff[2] = prefix[2];
    auto it = buff + 3;
    for (std::size_t i = 0; i < count; ++i) {
        strf::detail::copy_n(str_to_be_repeated.begin(), str_to_be_repeated.size(), it);
        it += str_to_be_repeated.size();
    }
    return {buff, it};
}

template <typename CharT>
strf::detail::simple_string_view<CharT> STRF_TEST_FUNC concatenate
    ( CharT* buff
    , const CharT(&prefix)[3]
    , strf::detail::simple_string_view<CharT> str_to_be_repeated
    , std::size_t count
    , const CharT(&suffix)[3] )
{
    auto r = concatenate(buff, prefix, str_to_be_repeated, count);
    buff [r.size()] = suffix[0];
    buff [r.size() + 1] = suffix[1];
    buff [r.size() + 2] = suffix[2];
    return {buff, r.size() + 3};
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

STRF_TEST_FUNC bool encoding_error_handler_called = false ;

void STRF_TEST_FUNC  encoding_error_handler()
{
    encoding_error_handler_called = true;
}

template <typename SrcEncoding, typename DestEncoding>
void STRF_TEST_FUNC test_invalid_input
    ( SrcEncoding src_enc
    , DestEncoding dest_enc
    , invalid_seq<get_first_template_parameter<SrcEncoding>> s
    , strf::surrogate_policy policy )
{
    using src_char_type  = get_first_template_parameter<SrcEncoding>;
    using dest_char_type = get_first_template_parameter<DestEncoding>;

    const int err_count = s.errors_count;
    const auto& seq = s.sequence;

#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

    auto f = [](auto ch){
        return *strf::hex((unsigned)(std::make_unsigned_t<src_char_type>)ch);
    };
    TEST_SCOPE_DESCRIPTION
        .with(strf::lettercase::mixed)
        ( "Sequence = ", strf::separated_range(seq, " ", f), " / policy = "
        , policy == strf::surrogate_policy::strict ? "strict" : "lax" );

#if defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)
#  pragma GCC diagnostic pop
#endif

    const src_char_type  prefix_in  [] = { 'a', 'b', 'c' };
    const dest_char_type prefix_out [] = { 'a', 'b', 'c' };
    src_char_type buff_in[20];
    dest_char_type buff_out[80];

    {
        auto input = concatenate(buff_in, prefix_in, seq, 1);
        {   // test if invalid sequences are replaced by U+FFFD
            auto expected = concatenate( buff_out
                                       , prefix_out
                                       , replacement_char(dest_enc)
                                       , err_count );

            TEST(expected) .with(policy, dest_enc) (strf::sani(input, src_enc));
        }
        {   // test invalid_seq_notifier
            ::encoding_error_handler_called = false;
            strf::to(buff_out)
                .with(policy, dest_enc, strf::invalid_seq_notifier{encoding_error_handler})
                (strf::sani(input, src_enc));
            TEST_TRUE(::encoding_error_handler_called);
        }
    }

    {   // same thing, but with some valid characters at the end.
        const src_char_type  suffix_in  [] = { 'd', 'e', 'f' };
        const dest_char_type suffix_out [] = { 'd', 'e', 'f' };
        auto input = concatenate(buff_in, prefix_in, seq, 1, suffix_in);

        {   // test if invalid sequences are replaced by U+FFFD
            auto expected = concatenate( buff_out
                                         , prefix_out
                                         , replacement_char(dest_enc)
                                         , err_count
                                         , suffix_out );

            TEST(expected) .with(policy, dest_enc) (strf::sani(input, src_enc));
        }

        {   // test invalid_seq_notifier
            ::encoding_error_handler_called = false;
            strf::to(buff_out)
                .with(policy, dest_enc, strf::invalid_seq_notifier{encoding_error_handler})
                (strf::sani(input, src_enc));
            TEST_TRUE(::encoding_error_handler_called);
        }
    }
}

template <typename SrcEncoding, typename DestEncoding>
void STRF_TEST_FUNC test_invalid_input(SrcEncoding src_enc, DestEncoding dest_enc)
{
    TEST_SCOPE_DESCRIPTION("From invalid ", src_enc.name(), " to ", dest_enc.name());
    for(const auto& s : invalid_sequences(src_enc))
    {
        test_invalid_input(src_enc, dest_enc, s, strf::surrogate_policy::strict);
        test_invalid_input(src_enc, dest_enc, s, strf::surrogate_policy::lax);
    }
    for(const auto& s : surrogates_sequences(src_enc))
    {
        test_invalid_input(src_enc, dest_enc, s, strf::surrogate_policy::strict);
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

void STRF_TEST_FUNC test_transcode_utf_to_utf()
{
    const auto encodings = std::make_tuple
        ( strf::utf<char>()
        , strf::utf<char16_t>()
        , strf::utf<char32_t>()
#if ! defined(__CUDACC__)
        // causes CUDACC to crash ( maybe because compilation unit is too big )
        , strf::utf<wchar_t>()
#endif
        );

    for_all_combinations
        ( encodings
        , [](auto src_enc, auto dest_enc){ test_valid_input(src_enc, dest_enc); } );

    for_all_combinations
        ( encodings
        , [](auto src_enc, auto dest_enc){ test_allowed_surrogates(src_enc, dest_enc); } );

    for_all_combinations
        ( encodings
        , [](auto src_enc, auto dest_enc){ test_invalid_input(src_enc, dest_enc); } );
}

template <typename Enc>
void STRF_TEST_FUNC test_codepoints_robust_count_invalid_sequences
    ( Enc enc
    , invalid_seq<get_first_template_parameter<Enc>> s
    , strf::surrogate_policy policy )
{
    using char_type = get_first_template_parameter<Enc>;
    const char_type prefix[] = { 'a', 'b', 'c' };
    char_type buff[20];
    {
        auto input = concatenate(buff, prefix, s.sequence, 1);

        const std::size_t expected_count = 3 + s.errors_count;
        auto result = enc.codepoints_robust_count
            (input.data(), input.size(), expected_count, policy);

        TEST_EQ(result.count, expected_count);
        TEST_EQ(result.pos, input.size());
    }
    {
        const char_type suffix[] = { 'd', 'e', 'f' };
        auto input = concatenate(buff, prefix, s.sequence, 1, suffix);

        const std::size_t expected_count = 6 + s.errors_count;
        auto result = enc.codepoints_robust_count
            (input.data(), input.size(), expected_count, policy);

        TEST_EQ(result.count, expected_count);
        TEST_EQ(result.pos, input.size());
    }
}

void STRF_TEST_FUNC test_codepoints_robust_count_valid_sequences(strf::utf<char> enc)
{
    constexpr auto strict = strf::surrogate_policy::strict;
    constexpr auto lax = strf::surrogate_policy::lax;
    {
        auto r = enc.codepoints_robust_count("", 0, 0, strict);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count("", 0, 1, strict);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count("x", 1, 0, strict);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count((const char*)u8"\u0080", 2, 1, strict);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_robust_count((const char*)u8"\u07FF", 2, 1, strict);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_robust_count((const char*)u8"\u0800", 3, 1, strict);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_robust_count((const char*)u8"\u8000", 3, 1, strict);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_robust_count((const char*)u8"\uFFFF", 3, 1, strict);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_robust_count((const char*)u8"\U00010000", 4, 1, strict);
        TEST_EQ(r.pos, 4);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_robust_count("\xED\xA0\x80", 3, 1, lax);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
}

void STRF_TEST_FUNC test_codepoints_robust_count_valid_sequences(strf::utf<char16_t> enc)
{
    constexpr auto lax = strf::surrogate_policy::lax;
    {
        auto r = enc.codepoints_robust_count(u"abc", 3, 0, lax);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count(u"", 0, 3, lax);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count(u"\U0010AAAA", 2, 1, lax);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 1);
    }
    {
        char16_t a_high_surrogate = 0xD800;
        auto r = enc.codepoints_robust_count(&a_high_surrogate, 1, 1, lax);
        TEST_EQ(r.pos, 1);
        TEST_EQ(r.count, 1);
    }
    {
        char16_t a_low_surrogate = 0xDC00;
        auto r = enc.codepoints_robust_count(&a_low_surrogate, 1, 1, lax);
        TEST_EQ(r.pos, 1);
        TEST_EQ(r.count, 1);
    }
}

void STRF_TEST_FUNC test_codepoints_robust_count_valid_sequences(strf::utf<char32_t> enc)
{
    constexpr auto lax = strf::surrogate_policy::lax;
    {
        auto r = enc.codepoints_robust_count(U"", 0, 0, lax);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count(U"", 0, 1, lax);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count(U"a", 1, 0, lax);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_robust_count(U"abc", 3, 3, lax);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 3);
    }
    {
        auto r = enc.codepoints_robust_count(U"abc", 3, 2, lax);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 2);
    }
}

template <typename Enc>
void STRF_TEST_FUNC test_codepoints_robust_count(Enc enc)
{
    test_codepoints_robust_count_valid_sequences(enc);

    constexpr auto strict = strf::surrogate_policy::strict;
    constexpr auto lax = strf::surrogate_policy::lax;

    for(const auto& s : invalid_sequences(enc))
    {
        test_codepoints_robust_count_invalid_sequences(enc, s, strict);
        test_codepoints_robust_count_invalid_sequences(enc, s, lax);
    }
    for(const auto& s : surrogates_sequences(enc))
    {
        test_codepoints_robust_count_invalid_sequences(enc, s, strict);
    }
}

void STRF_TEST_FUNC test_codepoints_fast_count(strf::utf<char> enc)
{
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\u0080", 2, 1);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\u07FF", 2, 1);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\u0800", 3, 1);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\u8000", 3, 1);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\uFFFF", 3, 1);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\U00010000", 4, 1);
        TEST_EQ(r.pos, 4);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\U0010FFFF", 4, 1);
        TEST_EQ(r.pos, 4);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"\U0010FFFF", 3, 1);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count((const char*)u8"_\U0010FFFF_", 6, 2);
        TEST_EQ(r.pos, 5);
        TEST_EQ(r.count, 2);
    }
}

void STRF_TEST_FUNC test_codepoints_fast_count(strf::utf<char16_t> enc)
{
    {
        auto r = enc.codepoints_fast_count(u"", 0, 0);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_fast_count(u"", 0, 1);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_fast_count(u" \U00010000 ", 4, 2);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 2);
    }
    {
        auto r = enc.codepoints_fast_count(u"\u0800", 1, 1);
        TEST_EQ(r.pos, 1);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count(u"\U00010000", 2, 1);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count(u"\U0010FFFF", 2, 1);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 1);
    }
    {
        auto r = enc.codepoints_fast_count(u"_\u0800_", 3, 3);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 3);
    }
    {
        auto r = enc.codepoints_fast_count(u"_\U00010000", 3, 2);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 2);
    }
    {
        auto r = enc.codepoints_fast_count(u"_\U00010000", 2, 2);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 2);
    }
}

void STRF_TEST_FUNC test_codepoints_fast_count(strf::utf<char32_t> enc)
{
    {
        auto r = enc.codepoints_fast_count(U"", 0, 0);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_fast_count(U"", 0, 1);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_fast_count(U"a", 1, 0);
        TEST_EQ(r.pos, 0);
        TEST_EQ(r.count, 0);
    }
    {
        auto r = enc.codepoints_fast_count(U"abc", 3, 3);
        TEST_EQ(r.pos, 3);
        TEST_EQ(r.count, 3);
    }
    {
        auto r = enc.codepoints_fast_count(U"abc", 3, 2);
        TEST_EQ(r.pos, 2);
        TEST_EQ(r.count, 2);
    }
}

void STRF_TEST_FUNC test_codepoints_count()
{
    test_codepoints_robust_count(strf::utf<char>{});
    test_codepoints_fast_count(strf::utf<char>{});

    test_codepoints_robust_count(strf::utf<char16_t>{});
    test_codepoints_fast_count(strf::utf<char16_t>{});

    test_codepoints_robust_count(strf::utf<char32_t>{});
    test_codepoints_fast_count(strf::utf<char32_t>{});
}

} // unamed namespace

void STRF_TEST_FUNC test_utf()
{
    test_transcode_utf_to_utf();
    test_codepoints_count();

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

    {
        auto tr = strf::utf<char>::find_transcoder_from(strf::tag<char16_t>{}, strf::eid_utf16);
        using expected_transcoder = strf::static_transcoder
            <char16_t, char, strf::eid_utf16, strf::eid_utf8>;
        TEST_TRUE(tr.transcode_func() == expected_transcoder::transcode_func());
    }
    {
        auto tr = strf::utf<char>::find_transcoder_from(strf::tag<char16_t>{}, strf::eid_utf32);
        TEST_TRUE(tr.transcode_func() == nullptr);
    }
    {
        auto tr = strf::utf<char16_t>::find_transcoder_from(strf::tag<char>{}, strf::eid_utf8);
        using expected_transcoder = strf::static_transcoder
            <char, char16_t, strf::eid_utf8, strf::eid_utf16>;
        TEST_TRUE(tr.transcode_func() == expected_transcoder::transcode_func());
    }
    {
        auto tr = strf::utf<char32_t>::find_transcoder_from(strf::tag<char>{}, strf::eid_utf8);
        using expected_transcoder = strf::static_transcoder
            <char, char32_t, strf::eid_utf8, strf::eid_utf32>;
        TEST_TRUE(tr.transcode_func() == expected_transcoder::transcode_func());
    }
    {
        auto tr = strf::utf<char32_t>::find_transcoder_from(strf::tag<char>{}, strf::eid_utf16);
        TEST_TRUE(tr.transcode_func() == nullptr);
    }
    {
        auto tr = strf::utf<char32_t>::find_transcoder_to(strf::tag<char>{}, strf::eid_utf8);
        using expected_transcoder = strf::static_transcoder
            <char32_t, char, strf::eid_utf32, strf::eid_utf8>;
        TEST_TRUE(tr.transcode_func() == expected_transcoder::transcode_func());
    }

    TEST_EQ(strf::utf<char32_t>::validate(0x123), 1);

    TEST(u8"\uFFFD").tr(u8"{10}"); // cover write_replacement_char(x);
    TEST(u"\uFFFD").tr(u"{10}");
    TEST(U"\uFFFD").tr(U"{10}");
}

REGISTER_STRF_TEST(test_utf);

