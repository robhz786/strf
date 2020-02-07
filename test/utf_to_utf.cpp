//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#include <array>
#include <tuple>

template <typename T>
constexpr auto as_signed(const T& value)
{
    return static_cast<typename std::make_signed<T>::type>(value);
}
strf::detail::simple_string_view<char>
valid_input_sample(const strf::utf<char>&)
{
    return {(const char*)u8"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 19};
}

strf::detail::simple_string_view<char16_t>
valid_input_sample(const strf::utf<char16_t>&)
{
    return {u"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 10};
}

strf::detail::simple_string_view<char32_t>
valid_input_sample(const strf::utf<char32_t>&)
{
    return {U"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 8};
}

strf::detail::simple_string_view<wchar_t>
valid_input_sample(const strf::utf<wchar_t>&)
{
    return {L"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", (sizeof(wchar_t) == 2 ? 10 : 8)};
}

template <typename SrcCharset, typename DestCharset>
void test_valid_input
    ( const SrcCharset& ein
    , const DestCharset& eout )
{
    TEST_SCOPE_DESCRIPTION("from ", ein.name(), " to ", eout.name());

    auto input = valid_input_sample(ein);
    auto expected = valid_input_sample(eout);
    TEST(expected).with(eout) (strf::sani(input, ein));
}

strf::detail::simple_string_view<char>
sample_with_surrogates(const strf::utf<char>&)
{
    return " \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF";
}
strf::detail::simple_string_view<char16_t>
sample_with_surrogates(const strf::utf<char16_t>&)
{
    static const char16_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
strf::detail::simple_string_view<char32_t>
sample_with_surrogates(const strf::utf<char32_t>&)
{
    static const char32_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
strf::detail::simple_string_view<wchar_t>
sample_with_surrogates(const strf::utf<wchar_t>&)
{
    static const wchar_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}

template <typename SrcCharset, typename DestCharset>
void test_allowed_surrogates
    ( const SrcCharset& ein
    , const DestCharset& eout )
{
    TEST_SCOPE_DESCRIPTION("from ", ein.name()," to ", eout.name());

    const auto input    = sample_with_surrogates(ein);
    const auto expected = sample_with_surrogates(eout);

    TEST(expected)
        .with( eout
               , strf::invalid_seq_policy::stop
               , strf::surrogate_policy::lax )
        (strf::sani(input, ein));
}

const auto& invalid_sequences(const strf::utf<char>&)
{
    // based on https://www.unicode.org/versions/Unicode10.0.0/ch03.pdf
    // "Best Practices for Using U+FFFD"
    using str_view = strf::detail::simple_string_view<char>;
    using pair = std::pair<int,str_view>;
    static std::array<std::pair<int, str_view>, 17> seqs
       {{ pair{3, "\xF1\x80\x80\xE1\x80\xC0"} // sample from Tabble 3-8 of Unicode standard
        , pair{2, "\xC1\xBF"}                 // overlong sequence
        , pair{3, "\xE0\x9F\x80"}             // overlong sequence
        , pair{3, "\xC1\xBF\x80"}             // overlong sequence with extra continuation bytes
        , pair{4, "\xE0\x9F\x80\x80"}         // overlong sequence with extra continuation bytes
        , pair{1, "\xC2"}                     // missing continuation
        , pair{1, "\xE0\xA0"}                 // missing continuation
        , pair{3, "\xED\xA0\x80"}             // surrogate
        , pair{3, "\xED\xAF\xBF"}             // surrogate
        , pair{3, "\xED\xB0\x80"}             // surrogate
        , pair{3, "\xED\xBF\xBF"}             // surrogate
        , pair{5, "\xED\xBF\xBF\x80\x80"}     // surrogate with extra continuation bytes
        , pair{4, "\xF0\x8F\xBF\xBF" }        // overlong sequence
        , pair{5, "\xF0\x8F\xBF\xBF\x80" }    // overlong sequence with extra continuation bytes
        , pair{1, "\xF0\x90\xBF" }            // missing continuation
        , pair{4, "\xF4\xBF\xBF\xBF"}         // codepoint too big
        , pair{6, "\xF5\x90\x80\x80\x80\x80"} // codepoint too big with extra continuation bytes
        }};

    return seqs;
}

const auto& invalid_sequences(const strf::utf<char16_t>&)
{
    using str_view = strf::detail::simple_string_view<char16_t>;
    using pair = std::pair<int,str_view>;
    static const char16_t ch[] = {0xD800, 0xDBFF, 0xDC00, 0xDFFF};
    static const std::array<std::pair<int,str_view>, 4> seqs
       {{ pair{1, {&ch[0], 1}}
        , pair{1, {&ch[1], 1}}
        , pair{1, {&ch[2], 1}}
        , pair{1, {&ch[3], 1}} }};

    return seqs;
}

const auto& invalid_sequences(const strf::utf<char32_t>&)
{
    using str_view = strf::detail::simple_string_view<char32_t>;
    using pair = std::pair<int,str_view>;
    static const char32_t ch[] = {0xD800, 0xDBFF,0xDC00,0xDFFF, 0x110000};
    static const std::array<pair, 5> seqs
       {{ pair{1, {&ch[0], 1}}
        , pair{1, {&ch[1], 1}}
        , pair{1, {&ch[2], 1}}
        , pair{1, {&ch[3], 1}}
        , pair{1, {&ch[4], 1}} }};

    return seqs;
}

const auto& invalid_sequences(const strf::utf<wchar_t>&)
{
    using str_view = strf::detail::simple_string_view<wchar_t>;
    using pair = std::pair<int,str_view>;
    static const wchar_t ch[] = { 0xD800, 0xDBFF, 0xDC00, 0xDFFF
                                , (sizeof(wchar_t) == 4 ? 0x110000 : 0xDFFF) };
    static const std::array<pair, 5> seqs
       {{ pair{1, {&ch[0], 1}}
        , pair{1, {&ch[1], 1}}
        , pair{1, {&ch[2], 1}}
        , pair{1, {&ch[3], 1}}
        , pair{1, {&ch[4], 1}} }};

    return seqs;
}

strf::detail::simple_string_view<char>
replacement_char(const strf::utf<char>&){ return (const char*)u8"\uFFFD";}
strf::detail::simple_string_view<char16_t>
replacement_char(const strf::utf<char16_t>&){ return u"\uFFFD";}
strf::detail::simple_string_view<char32_t>
replacement_char(const strf::utf<char32_t>&){ return U"\uFFFD";}
strf::detail::simple_string_view<wchar_t>
replacement_char(const strf::utf<wchar_t>&){ return L"\uFFFD";}


template <typename CharT>
strf::detail::simple_string_view<CharT> concatenate
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
        strf::detail::str_copy_n(it, str.begin(), str.size());
        it += str.size();
    }
    it[0] = suffix[0];
    it[1] = suffix[1];
    it[2] = suffix[2];

    return {buff, it + 3};
}

template <class>
struct get_template_parameter_impl;

template <class C, template <class> class Tmpl>
struct get_template_parameter_impl<Tmpl<C>>
{
    using type = C;
};

template <class T>
using get_template_parameter
= typename get_template_parameter_impl<T>::type;


template <typename SrcCharset, typename DestCharset>
void test_invalid_input
    ( const SrcCharset& ein
    , const DestCharset& eout )
{
    TEST_SCOPE_DESCRIPTION("From invalid ", ein.name(), " to ", eout.name());
    using src_char_type  = get_template_parameter<SrcCharset>;
    using dest_char_type = get_template_parameter<DestCharset>;

    const src_char_type  suffix_in  [] = { 'd', 'e', 'f' };
    const dest_char_type suffix_out [] = { 'd', 'e', 'f' };
    const src_char_type  prefix_in  [] = { 'a', 'b', 'c' };
    const dest_char_type prefix_out [] = { 'a', 'b', 'c' };

    for(const auto& s : invalid_sequences(ein))
    {
        const int err_count = s.first;
        const auto& seq = s.second;

        auto f = [](auto ch){
            return ~strf::hex((unsigned)(std::make_unsigned_t<src_char_type>)ch);
        };
        TEST_SCOPE_DESCRIPTION
            .with(strf::mixedcase)
            ( "Sequence = ", strf::separated_range(seq, " ", f) );

        src_char_type buff_in[20];
        dest_char_type buff_out[80];
        auto input = concatenate(buff_in, prefix_in, seq, 1, suffix_in);

        {   // replace
            auto expected = concatenate( buff_out
                                       , prefix_out
                                       , replacement_char(eout)
                                       , err_count
                                       , suffix_out );
            TEST(expected)
                .with(eout)
                .with(strf::invalid_seq_policy::replace)
                (strf::sani(input, ein));
        }

#if defined(__cpp_exceptions)

        // stop
        TEST_THROWS( (strf::to(buff_out)
                          .with(eout, strf::invalid_seq_policy::stop)
                          (strf::sani(input, ein)))
                   , strf::invalid_sequence );

#endif // defined(__cpp_exceptions)

    }
}


template < typename Func
         , typename SrcCharset >
void combine_3(Func, SrcCharset)
{
}

template < typename Func
         , typename SrcCharset
         , typename DestCharset0
         , typename ... DestCharset >
void combine_3(Func func, SrcCharset ein, DestCharset0 eout0, DestCharset ... eout)
{
    func(ein, eout0);
    combine_3(func, ein, eout...);
}

template < typename Func
         , typename Charset0 >
void combine_2(Func func, Charset0 cs0)
{
    combine_3(func, cs0, cs0);
}

template < typename Func
         , typename Charset0
         , typename ... Cs >
void combine_2(Func func, Charset0 cs0, Cs... cs)
{
    combine_3(func, cs0, cs0, cs...);
}

template < typename Func
         , typename EoutTuple
         , std::size_t ... I >
void combine(Func, const EoutTuple&, std::index_sequence<I...> )
{
}

template < typename Func
         , typename EoutTuple
         , std::size_t ... I
         , typename Charset0
         , typename ... Cs >
void combine( Func func
            , const EoutTuple& out_charsets
            , std::index_sequence<I...> iseq
            , Charset0 cs0
            , Cs ... cs )

{
    combine_2(func, cs0, std::get<I>(out_charsets)...);
    combine(func, out_charsets, iseq, cs...);
}

template < typename Func, typename Tuple, std::size_t ... I >
void for_all_combinations(Func func, const Tuple& charsets, std::index_sequence<I...> iseq)
{
    combine(func, charsets, iseq, std::get<I>(charsets)...);
}

template < typename Tuple, typename Func >
void for_all_combinations(const Tuple& charsets, Func func)
{
    constexpr std::size_t tsize = std::tuple_size<Tuple>::value;
    for_all_combinations(func, charsets, std::make_index_sequence<tsize>());
}


int main()
{
    const auto charsets = std::make_tuple
        ( strf::utf8<char>(), strf::utf16<char16_t>()
        , strf::utf32<char32_t>(), strf::utfw());

    for_all_combinations
        ( charsets
        , [](auto ein, auto eout){ test_valid_input(ein, eout); } );

    for_all_combinations
        ( charsets
        , [](auto ein, auto eout){ test_allowed_surrogates(ein, eout); } );

    for_all_combinations
        ( charsets
        , [](auto ein, auto eout){ test_invalid_input(ein, eout); } );

    return test_finish();
}
