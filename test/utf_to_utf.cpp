//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#include <vector>
#include <tuple>

template <typename T>
constexpr auto as_signed(const T& value)
{
    return static_cast<typename std::make_signed<T>::type>(value);
}

std::string valid_input_sample(const strf::encoding<char>&)
{
    return {(const char*)u8"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 19};
}
std::u16string valid_input_sample(const strf::encoding<char16_t>&)
{
    return {u"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 10};
}
std::u32string valid_input_sample(const strf::encoding<char32_t>&)
{
    return {U"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 8};
}
std::wstring valid_input_sample(const strf::encoding<wchar_t>&)
{
    return {L"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", (sizeof(wchar_t) == 2 ? 10 : 8)};
}

template <typename CharIn, typename CharOut>
void test_valid_input
    ( const strf::encoding<CharIn>& ein
    , const strf::encoding<CharOut>& eout )
{
    BOOST_TEST_LABEL << "from " << ein.name() << " to " << eout.name();

    auto input = valid_input_sample(ein);
    auto expected = valid_input_sample(eout);
    TEST(expected).with(eout) (strf::sani(input, ein));
}

std::string sample_with_surrogates(const strf::encoding<char>&)
{
    return " \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF";
}
std::u16string sample_with_surrogates(const strf::encoding<char16_t>&)
{
    static const char16_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
std::u32string sample_with_surrogates(const strf::encoding<char32_t>&)
{
    static const char32_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
std::wstring sample_with_surrogates(const strf::encoding<wchar_t>&)
{
    static const wchar_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}

template <typename CharIn, typename CharOut>
void test_allowed_surrogates
    ( const strf::encoding<CharIn>& ein
    , const strf::encoding<CharOut>& eout )
{
    BOOST_TEST_LABEL << "from " << ein.name() << " to " << eout.name();

    const auto input    = sample_with_surrogates(ein);
    const auto expected = sample_with_surrogates(eout);

    TEST(expected)
        .with( eout
               , strf::encoding_error::stop
               , strf::surrogate_policy::lax )
        (strf::sani(input, ein));
}

const auto& invalid_sequences(const strf::encoding<char>&)
{
    // based on https://www.unicode.org/versions/Unicode10.0.0/ch03.pdf
    // "Best Practices for Using U+FFFD"
    static std::vector<std::pair<int, std::string>> seqs =
        { {3, "\xF1\x80\x80\xE1\x80\xC0"} // sample from Tabble 3-8 of Unicode standard
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
        };

    return seqs;
}

const auto& invalid_sequences(const strf::encoding<char16_t>&)
{
    static const std::vector<std::pair<int, std::u16string>> seqs =
        { {1, {(char16_t)0xD800}}
        , {1, {(char16_t)0xDBFF}}
        , {1, {(char16_t)0xDC00}}
        , {1, {(char16_t)0xDFFF}} };

    return seqs;
}

const auto& invalid_sequences(const strf::encoding<char32_t>&)
{
    static const std::vector<std::pair<int, std::u32string>> seqs =
        { {1, {(char32_t)0xD800}}
        , {1, {(char32_t)0xDBFF}}
        , {1, {(char32_t)0xDC00}}
        , {1, {(char32_t)0xDFFF}}
        , {1, {(char32_t)0x110000}} };

    return seqs;
}

const auto& invalid_sequences(const strf::encoding<wchar_t>&)
{
    static const std::vector<std::pair<int, std::wstring>> seqs =
        { {1, {(wchar_t)0xD800}}
        , {1, {(wchar_t)0xDBFF}}
        , {1, {(wchar_t)0xDC00}}
        , {1, {(wchar_t)0xDFFF}}
        , {1, {wchar_t(sizeof(wchar_t) == 4 ? 0x110000 : 0xDFFF)}} };

    return seqs;
}

std::string    replacement_char(const strf::encoding<char>&){ return (const char*)u8"\uFFFD";}
std::u16string replacement_char(const strf::encoding<char16_t>&){ return u"\uFFFD";}
std::u32string replacement_char(const strf::encoding<char32_t>&){ return U"\uFFFD";}
std::wstring   replacement_char(const strf::encoding<wchar_t>&){ return L"\uFFFD";}


template <typename StrType, typename CharIn = typename StrType::value_type>
std::string stringify_invalid_char_sequence(const StrType& seq)
{
    std::vector<unsigned> vec(seq.begin(), seq.end());
    return strf::to_string(' ', ~strf::fmt_range(vec, " ").hex());
}

template <typename ChIn, typename ChOut>
void test_invalid_input
    ( const strf::encoding<ChIn>& ein
    , const strf::encoding<ChOut>& eout )
{
    BOOST_TEST_LABEL << "From invalid " << ein.name() << " to " << eout.name();

    const std::basic_string<ChIn>  suffix_in { (ChIn)'d', (ChIn)'e', (ChIn)'f' };
    const std::basic_string<ChOut> suffix_out{ (ChOut)'d', (ChOut)'e', (ChOut)'f' };
    const std::basic_string<ChIn>  prefix_in { (ChIn)'a', (ChIn)'b', (ChIn)'c' };
    const std::basic_string<ChOut> prefix_out{ (ChOut)'a', (ChOut)'b', (ChOut)'c' };

    for(const auto& s : invalid_sequences(ein))
    {
        const int err_count = s.first;
        const auto& seq = s.second;

        BOOST_TEST_LABEL << "Sequence = " << stringify_invalid_char_sequence(seq);

        const std::basic_string<ChIn> input = prefix_in + seq + suffix_in;

        {   // replace
            std::basic_string<ChOut> expected = prefix_out;
            for(int i = 0; i < err_count; i++)
                expected.append(replacement_char(eout));
            expected += suffix_out;

            TEST(expected)
                .with(eout)
                .with(strf::encoding_error::replace)
                (strf::sani(input, ein));
        }


        // stop
        BOOST_TEST_THROWS( (strf::to_string.with(eout)
                                           .with(strf::encoding_error::stop)
                                            (strf::sani(input, ein)) )
                          , strf::encoding_failure );
    }
}


template < typename Func
         , typename EncIn >
void combine_3(Func, EncIn)
{
}

template < typename Func
         , typename EncIn
         , typename EncOut0
         , typename ... EncOut >
void combine_3(Func func, EncIn ein, EncOut0 eout0, EncOut ... eout)
{
    func(ein, eout0);
    combine_3(func, ein, eout...);
}

template < typename Func
         , typename Enc0 >
void combine_2(Func func, Enc0 enc0)
{
    combine_3(func, enc0, enc0);
}

template < typename Func
         , typename Enc0
         , typename ... Enc >
void combine_2(Func func, Enc0 enc0, Enc... enc)
{
    combine_3(func, enc0, enc0, enc...);
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
         , typename Enc0
         , typename ... Enc >
void combine( Func func
            , const EoutTuple& out_encodings
            , std::index_sequence<I...> iseq
            , Enc0 enc0
            , Enc ... enc )

{
    combine_2(func, enc0, std::get<I>(out_encodings)...);
    combine(func, out_encodings, iseq, enc...);
}

template < typename Func, typename Tuple, std::size_t ... I >
void for_all_combinations(Func func, const Tuple& encodings, std::index_sequence<I...> iseq)
{
    combine(func, encodings, iseq, std::get<I>(encodings)...);
}

template < typename Tuple, typename Func >
void for_all_combinations(const Tuple& encodings, Func func)
{
    constexpr std::size_t tsize = std::tuple_size<Tuple>::value;
    for_all_combinations(func, encodings, std::make_index_sequence<tsize>());
}


int main()
{
    const auto encodings = std::make_tuple
        ( strf::utf8<char>(), strf::utf16<char16_t>()
        , strf::utf32<char32_t>(), strf::wchar_encoding());

    for_all_combinations
        ( encodings
        , [](auto ein, auto eout){ test_valid_input(ein, eout); } );

    for_all_combinations
        ( encodings
        , [](auto ein, auto eout){ test_allowed_surrogates(ein, eout); } );

    for_all_combinations
        ( encodings
        , [](auto ein, auto eout){ test_invalid_input(ein, eout); } );


    return boost::report_errors();
}
