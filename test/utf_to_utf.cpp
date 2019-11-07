//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#include <stringify.hpp>

#include <boost/hana/for_each.hpp>
#include <boost/hana/tuple.hpp>
#include <vector>

namespace hana = boost::hana;

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
    TEST(expected).facets(eout) (strf::cv(input, ein));
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
        .facets( eout
               , strf::encoding_error::stop
               , strf::surrogate_policy::lax )
        (strf::cv(input, ein));
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
                .facets(eout)
                .facets(strf::encoding_error::replace)
                (strf::cv(input, ein));
        }

        // ignore
        TEST(prefix_out + suffix_out)
            .reserve(6)
            .facets(eout)
            .facets(strf::encoding_error::ignore)
            (strf::cv(input, ein));

        // stop
        BOOST_TEST_THROWS( (strf::to_string.facets(eout)
                                           .facets(strf::encoding_error::stop)
                                            (strf::cv(input, ein)) )
                          , strf::encoding_failure );
    }
}

int main()
{
    auto encodings = hana::make_tuple
        ( strf::utf8<char>(), strf::utf16<char16_t>()
        , strf::utf32<char32_t>(), strf::wchar_encoding());

    hana::for_each(encodings, [&](const auto& ein){
            hana::for_each(encodings, [&](const auto& eout) {
                    test_valid_input(ein, eout);
                });
        });

    hana::for_each(encodings, [&](const auto& ein){
            hana::for_each(encodings, [&](const auto& eout) {
                    test_allowed_surrogates(ein, eout);
                });
        });

    hana::for_each(encodings, [&](const auto& ein){
            hana::for_each(encodings, [&](const auto& eout) {
                    test_invalid_input(ein, eout);
                });
        });

    return boost::report_errors();
}
