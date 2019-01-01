//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>

namespace boost {
namespace detail {


class test_identifier
{
public:

    test_identifier(std::string x)
    {
        _current_test_id = std::move(x);
    }

    ~test_identifier()
    {
        _current_test_id.clear();
    }

    static void write_current_id(std::ostream& out)
    {
        if( ! _current_test_id.empty())
        {
            out << " [ " << _current_test_id << " ] ";
        }
    }

private:

    static std::string _current_test_id;
};

std::string test_identifier::_current_test_id {};

inline void test_failed_impl_v2
    ( char const * expr
    , char const * file
    , int line
    , char const * function )
{
    BOOST_LIGHTWEIGHT_TEST_OSTREAM
        << file << "(" << line << "): test '" << expr << "' failed in function '"
        << function << "'";
    test_identifier::write_current_id(BOOST_LIGHTWEIGHT_TEST_OSTREAM);
    BOOST_LIGHTWEIGHT_TEST_OSTREAM << std::endl << std::endl;
    ++test_errors();
}


template<class BinaryPredicate, class T, class U>
inline void test_with_impl_v2(BinaryPredicate pred, char const * expr1, char const * expr2,
                           char const * file, int line, char const * function,
                           T const & t, U const & u)
{
    if( pred(t, u) )
    {
        report_errors_remind();
    }
    else
    {
        BOOST_LIGHTWEIGHT_TEST_OSTREAM
            << file << "(" << line << "): test '" << expr1 << " " << pred.op() << " " << expr2
            << "' ('" << test_output_impl(t) << "' " << pred.op() << " '" << test_output_impl(u)
            << "') failed in function '" << function << "'";
        test_identifier::write_current_id(BOOST_LIGHTWEIGHT_TEST_OSTREAM);
        BOOST_LIGHTWEIGHT_TEST_OSTREAM << std::endl << std::endl;
        ++test_errors();
    }
}



} // namespace detail
} // namespace boost

#define STR_CONCAT(str1, str2) str1 ## str2

#define BOOST_TEST_LABEL(str) \
::boost::detail::test_identifier STR_CONCAT(test_identifier_, __LINE__) {(str)};


#undef BOOST_TEST
#define BOOST_TEST(expr)                                                \
    ( (expr)                                                            \
    ? (void)0                                                           \
    : ::boost::detail::test_failed_impl_v2(#expr, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION))

#undef BOOST_TEST_EQ
#undef BOOST_TEST_NE
#undef BOOST_TEST_LT
#undef BOOST_TEST_LE
#undef BOOST_TEST_GT
#undef BOOST_TEST_GE

#define BOOST_TEST_EQ(expr1,expr2) ( ::boost::detail::test_with_impl_v2(::boost::detail::lw_test_eq(), #expr1, #expr2, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, expr1, expr2) )
#define BOOST_TEST_NE(expr1,expr2) ( ::boost::detail::test_with_impl_v2(::boost::detail::lw_test_ne(), #expr1, #expr2, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, expr1, expr2) )

#define BOOST_TEST_LT(expr1,expr2) ( ::boost::detail::test_with_impl_v2(::boost::detail::lw_test_lt(), #expr1, #expr2, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, expr1, expr2) )
#define BOOST_TEST_LE(expr1,expr2) ( ::boost::detail::test_with_impl_v2(::boost::detail::lw_test_le(), #expr1, #expr2, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, expr1, expr2) )
#define BOOST_TEST_GT(expr1,expr2) ( ::boost::detail::test_with_impl_v2(::boost::detail::lw_test_gt(), #expr1, #expr2, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, expr1, expr2) )
#define BOOST_TEST_GE(expr1,expr2) ( ::boost::detail::test_with_impl_v2(::boost::detail::lw_test_ge(), #expr1, #expr2, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, expr1, expr2) )




#include <boost/utility/string_view.hpp>
#include <boost/stringify.hpp>

#include <boost/hana/for_each.hpp>
#include <boost/hana/tuple.hpp>
#include <iostream>

namespace strf = boost::stringify::v0;
namespace hana = boost::hana;
using boost::basic_string_view;
using boost::string_view;
using boost::u16string_view;


namespace boost {
namespace stringify {
inline namespace v0 {

inline std::ostream& operator<<(std::ostream& dest, strf::cv_result r)
{
    return dest << ( r == strf::cv_result::success ? "success"
                   : r == strf::cv_result::insufficient_space ? "insufficient_space"
                   : r == strf::cv_result::invalid_char ? "invalid_char"
                   : "???" );
}

} } }


template <typename T>
constexpr auto as_signed(const T& value)
{
    return static_cast<typename std::make_signed<T>::type>(value);
}

constexpr std::size_t buff_size = 10000;

basic_string_view<char> valid_input_sample(const strf::encoding<char>&)
{
    return {u8"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 19};
}
basic_string_view<char16_t> valid_input_sample(const strf::encoding<char16_t>&)
{
    return {u"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 10};
}
basic_string_view<char32_t> valid_input_sample(const strf::encoding<char32_t>&)
{
    return {U"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", 8};
}
basic_string_view<wchar_t> valid_input_sample(const strf::encoding<wchar_t>&)
{
    return {L"a\0b\u0080\u0800\uD7FF\U00010000\U0010FFFF", (sizeof(wchar_t) == 2 ? 10 : 8)};
}


template <typename CharIn, typename CharOut>
void test_valid_input
    ( const strf::transcoder<CharIn, CharOut>* cv
    , const basic_string_view<CharIn> input
    , const basic_string_view<CharOut> expected )
{
    CharOut buff[buff_size];
    CharOut* const buff_end = buff + buff_size;

    BOOST_TEST(cv != nullptr);
    if (cv == nullptr)
    {
        return;
    }

    {   // calculate size

        auto calculated_size = cv->necessary_size
            ( input.begin(), input.end()
            , strf::error_handling::replace, false );

        BOOST_TEST_EQ(calculated_size, expected.size());
    }

    {   // convert

        std::fill(buff, buff_end, 'x');
        buff[expected.size()] = 'x';
        auto src_it = input.begin();
        auto dest_it = buff;
        auto res = cv->transcode( &src_it
                                , input.end()
                                , &dest_it
                                , buff_end
                                , strf::error_handling::replace
                                , false);

        BOOST_TEST_EQ(res, strf::cv_result::success);
        BOOST_TEST_EQ(src_it, input.end());
        BOOST_TEST((expected == std::basic_string<CharOut>(buff, dest_it)));
        BOOST_TEST_EQ(*dest_it, 'x');
    }


    {   // test insufficient space

        std::fill(buff, buff_end, 'x');

        auto src_it = input.begin();
        auto dest_it = buff;
        auto res = cv->transcode( &src_it
                                , input.end()
                                , &dest_it
                                , buff + expected.size() - 1
                                , strf::error_handling::replace, false);
        BOOST_TEST_EQ(res, strf::cv_result::insufficient_space);
        BOOST_TEST_LT(src_it, input.end());
        BOOST_TEST_LT((dest_it - buff), as_signed(expected.size()));
        BOOST_TEST(std::equal(buff, dest_it, expected.begin()));
        BOOST_TEST_EQ(buff[expected.size() - 1], 'x');

        // write the remaining part

        auto src_it2 = src_it;
        auto dest_it2 = dest_it;
        auto res2 = cv->transcode( &src_it2
                                 , input.end()
                                 , &dest_it2
                                 , buff_end
                                 , strf::error_handling::replace
                                 , false );
        BOOST_TEST_EQ(res2, strf::cv_result::success);
        BOOST_TEST_EQ(src_it2, input.end());
        BOOST_TEST_EQ(dest_it2 - buff, as_signed(expected.size()));
        BOOST_TEST(std::equal(expected.begin(), expected.end(), buff));
        BOOST_TEST_EQ(buff[expected.size()], 'x');

    }
}

template <typename CharIn, typename CharOut>
void test_valid_input
    ( const strf::encoding<CharIn>& ein
    , const strf::encoding<CharOut>& eout )
{
    auto test_id = strf::to_string("from ", ein.name, " to ", eout.name).value();
    BOOST_TEST_LABEL(test_id);

    test_valid_input( strf::get_transcoder(ein, eout)
                    , valid_input_sample(ein)
                    , valid_input_sample(eout) );

}

template <typename CharOut>
void test_overlong_sequence
    ( const strf::encoding<char>& ein
    , const strf::encoding<CharOut>& eout )
{
    auto test_id = strf::to_string("from ", ein.name, " to ", eout.name).value();
    BOOST_TEST_LABEL(test_id);

}


basic_string_view<char> sample_with_surrogates(const strf::encoding<char>&)
{
    return {u8" \xED\xA0\x80 \xED\xAF\xBF \xED\xB0\x80 \xED\xBF\xBF"};
}
basic_string_view<char16_t> sample_with_surrogates(const strf::encoding<char16_t>&)
{
    static const char16_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
basic_string_view<char32_t> sample_with_surrogates(const strf::encoding<char32_t>&)
{
    static const char32_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}
basic_string_view<wchar_t> sample_with_surrogates(const strf::encoding<wchar_t>&)
{
    static const wchar_t arr[] = {' ', 0xD800, ' ', 0xDBFF, ' ', 0xDC00, ' ', 0xDFFF, 0};
    return {arr, 8};
}

template <typename CharIn, typename CharOut>
void test_allowed_surrogates
    ( const strf::encoding<CharIn>& ein
    , const strf::encoding<CharOut>& eout )
{
    auto test_id = strf::to_string("from ", ein.name, " to ", eout.name).value();
    BOOST_TEST_LABEL(test_id);

    const auto input    = sample_with_surrogates(ein);
    const auto expected = sample_with_surrogates(eout);
    const strf::transcoder<CharIn, CharOut>* cv = strf::get_transcoder(ein, eout);

    CharOut buff[buff_size];
    CharOut* const buff_end = buff + buff_size;

    {   // calculate size
        auto s = cv->necessary_size( input.begin()
                                   , input.end()
                                   , strf::error_handling::stop
                                   , true );
        BOOST_TEST_EQ(s, expected.size());
    }

    {   // convert

        auto src_it = input.begin();
        auto dest_it = buff;
        auto res = cv->transcode( &src_it
                                , input.end()
                                , &dest_it
                                , buff_end
                                , strf::error_handling::stop
                                , true );
        BOOST_TEST_EQ(res, strf::cv_result::success);
        BOOST_TEST_EQ(src_it, input.end());
        BOOST_TEST_EQ((dest_it - buff), as_signed(expected.size()));
        BOOST_TEST(std::equal( expected.begin()
                             , expected.end()
                             , buff ));
    }
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

std::string    replacement_char(const strf::encoding<char>&){ return u8"\uFFFD";}
std::u16string replacement_char(const strf::encoding<char16_t>&){ return u"\uFFFD";}
std::u32string replacement_char(const strf::encoding<char32_t>&){ return U"\uFFFD";}
std::wstring   replacement_char(const strf::encoding<wchar_t>&){ return L"\uFFFD";}


template <typename CharIn, typename CharOut, typename StrType>
std::string invalid_input_test_label
    ( const strf::encoding<CharIn>& ein
    , const StrType& seq
    , const strf::encoding<CharOut>& eout )
{
    char buff[1000];
    char* it = buff;
    it += sprintf(it, "from %s to %s, when sequence is: ", ein.name, eout.name);
    for (auto ch : seq)
    {
        typename std::make_unsigned<CharIn>::type uch = ch;
        it += sprintf(it, " %X", (unsigned)uch);
    }
    return buff;
}

template <typename ChIn, typename ChOut>
void test_invalid_input
    ( const strf::encoding<ChIn>& ein
    , const strf::encoding<ChOut>& eout )
{
    const strf::transcoder<ChIn, ChOut>* cv = strf::get_transcoder(ein, eout);

    if (cv == nullptr)
    {
        return;
    }

    const std::basic_string<ChIn>  prefix_in { (ChIn)'a', (ChIn)'b', (ChIn)'c' };
    const std::basic_string<ChIn>  suffix_in { (ChIn)'d', (ChIn)'e', (ChIn)'f' };
    const std::basic_string<ChOut> prefix_out{ (ChOut)'a', (ChOut)'b', (ChOut)'c' };
    const std::basic_string<ChOut> suffix_out{ (ChOut)'d', (ChOut)'e', (ChOut)'f' };
    ChOut buff[buff_size];
    ChOut* const buff_end = buff + buff_size;

    for(const auto& s : invalid_sequences(ein))
    {
        const auto& seq = s.second;
        int err_count = s.first;

        BOOST_TEST_LABEL(invalid_input_test_label(ein, seq, eout));

        const std::basic_string<ChIn> input = prefix_in + seq + suffix_in;

        {   // calculate size when ignoring invalid seq
            const auto expected_size = prefix_out.size() + suffix_out.size();
            auto calculated_size = cv->necessary_size
                ( &*input.begin()
                , &*input.end()
                , strf::error_handling::ignore
                , false );

            BOOST_TEST_GE(calculated_size, expected_size);
        }

        {   // calculate size when stopping in invalid seq
            auto calculated_size = cv->necessary_size( &*input.begin()
                                                     , &*input.end()
                                                     , strf::error_handling::stop
                                                     , false );

            BOOST_TEST_GE(calculated_size, prefix_out.size());
        }

        {   // calculate size when replacing invalid seq
            auto calculated_size = cv->necessary_size( &*input.begin()
                                                     , &*input.end()
                                                     , strf::error_handling::replace
                                                     , false );
            auto expected_size = ( prefix_out.size()
                                 + suffix_out.size()
                                 + eout.replacement_char_size * err_count);

            BOOST_TEST_GE(calculated_size, expected_size);
        }

        {   // convert ignoring invalid sequence
            const auto expected = prefix_out + suffix_out;
            std::fill(buff, buff_end, 'x');

            auto src_it = &*input.begin();
            auto dest_it = buff;
            auto res = cv->transcode( &src_it
                                    , &*input.end()
                                    , &dest_it
                                    , buff_end
                                    , strf::error_handling::ignore
                                    , false );
            BOOST_TEST_EQ(res, strf::cv_result::success);
            BOOST_TEST_EQ(&*input.end() - src_it, 0);
            BOOST_TEST_EQ((dest_it - buff), as_signed(expected.size()));
            BOOST_TEST(std::equal( expected.begin()
                                 , expected.end()
                                 , buff ));
            BOOST_TEST_EQ(buff[expected.size()], 'x');
        }

        {   // convert stopping on invalid sequence
            std::fill(buff, buff_end, 'x');

            auto src_it = &*input.begin();
            auto dest_it = buff;
            strf::cv_result res;
            int calls_count = 0;
            do
            {
                ++calls_count;
                res = cv->transcode( &src_it
                                   , &*input.end()
                                   , &dest_it
                                   , buff_end
                                   , strf::error_handling::stop
                                   , false );
            } while(res == strf::cv_result::invalid_char && calls_count < 10);

            const auto expected = prefix_out + suffix_out;

            BOOST_TEST_EQ(calls_count, err_count + 1);
            BOOST_TEST_EQ(src_it, &*input.end());
            BOOST_TEST(std::basic_string<ChOut>(buff, dest_it) == expected);
            BOOST_TEST_EQ(*dest_it, 'x');
        }

        {   // convert replacing invalid sequence
            auto expected = prefix_out;
            for(int i = 0; i < err_count; ++i)
            {
                expected.append(replacement_char(eout));
            }
            expected.append(suffix_out);


            std::fill(buff, buff_end, 'x');
            auto src_it = &*input.begin();
            auto dest_it = buff;
            auto res = cv->transcode( &src_it
                                    , &*input.end()
                                    , &dest_it
                                    , buff_end
                                    , strf::error_handling::replace
                                    , false );

            BOOST_TEST_EQ(res, strf::cv_result::success);
            BOOST_TEST_EQ((&*input.end() - src_it), 0);
            BOOST_TEST_EQ((dest_it - buff), as_signed(expected.size()));
            BOOST_TEST(std::equal( expected.begin()
                                 , expected.end()
                                 , buff ));
            BOOST_TEST_EQ(buff[expected.size()], 'x');
        }

        {   // insufficent space after ignoring invalid seq

            std::fill(buff, buff_end, 'x');
            const auto& expected = prefix_out;

            auto src_it = &*input.begin();
            auto dest_it = buff;
            auto res = cv->transcode( &src_it
                                    , &*input.end()
                                    , &dest_it
                                    , buff + prefix_out.size()
                                    , strf::error_handling::ignore
                                    , false );

            BOOST_TEST_EQ(res, strf::cv_result::insufficient_space);
            BOOST_TEST_EQ((dest_it - buff), as_signed(expected.size()));
            BOOST_TEST(std::equal( expected.begin()
                                 , expected.end()
                                 , buff ));
            BOOST_TEST_EQ(buff[expected.size()], 'x');

            // write remaining part

            auto src_it2 = src_it;
            auto dest_it2 = dest_it;
            auto res2 = cv->transcode( &src_it2
                                     , &*input.end()
                                     , &dest_it2
                                     , buff_end
                                     , strf::error_handling::ignore
                                     , false );
            const auto expected2 = prefix_out + suffix_out;

            BOOST_TEST_EQ(res2, strf::cv_result::success);
            BOOST_TEST_EQ((dest_it2 - buff), expected2.size());
            BOOST_TEST(std::equal( expected2.begin()
                                 , expected2.end()
                                 , buff ));
        }

        {   // insufficent space before replacing invalid seq

            std::fill(buff, buff_end, 'x');
            const auto& expected = prefix_out;

            auto src_it = &*input.begin();
            auto dest_it = buff;
            auto res = cv->transcode( &src_it
                                    , &*input.end()
                                    , &dest_it
                                    , buff + prefix_out.size()
                                    , strf::error_handling::replace
                                    , false );

            BOOST_TEST_EQ(res, strf::cv_result::insufficient_space);
            BOOST_TEST_EQ((src_it - &*input.begin()), as_signed(prefix_in.size()));
            BOOST_TEST_EQ((dest_it - buff), as_signed(expected.size()));
            BOOST_TEST(std::equal( expected.begin()
                                 , expected.end()
                                 , buff ));
            BOOST_TEST_EQ(buff[expected.size()], 'x');

            // write remaining part

            auto src_it2 = src_it;
            auto dest_it2 = dest_it;
            auto res2 = cv->transcode( &src_it2
                                     , &*input.end()
                                     , &dest_it2
                                     , buff_end
                                     , strf::error_handling::replace
                                     , false );

            auto expected2 = prefix_out;
            for(int i = 0; i < err_count; ++i)
            {
                expected2.append(replacement_char(eout));
            }
            expected2.append(suffix_out);

            BOOST_TEST_EQ(res2, strf::cv_result::success);
            BOOST_TEST_EQ((dest_it2 - buff), as_signed(expected2.size()));
            BOOST_TEST(std::equal( expected2.begin()
                                 , expected2.end()
                                 , buff ));
        }
    }
}


int main()
{
    auto encodings = hana::make_tuple
        ( strf::utf8(), strf::utf16(), strf::utf32(), strf::wchar_encoding());

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
