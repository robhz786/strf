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
    return {u8"abc\0def\u0080ghi\u0800jkl\U00010000", 22};
}
basic_string_view<char16_t> valid_input_sample(const strf::encoding<char16_t>&)
{
    return {u"abc\0def\u0080ghi\u0800jkl\U00010000", 17};
}
basic_string_view<char32_t> valid_input_sample(const strf::encoding<char32_t>&)
{
    return {U"abc\0def\u0080ghi\u0800jkl\U00010000", 16};
}
basic_string_view<wchar_t> valid_input_sample(const strf::encoding<wchar_t>&)
{
    return {L"abc\0def\u0080ghi\u0800jkl\U00010000", (sizeof(wchar_t) == 2 ? 17 : 16)};
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
                                , strf::error_handling::replace, false);

        BOOST_TEST_EQ(res, strf::cv_result::success);
        BOOST_TEST_EQ(src_it, input.end());
        BOOST_TEST_EQ((dest_it - buff), as_signed(expected.size()));
        BOOST_TEST((expected == std::basic_string<CharOut>(buff, expected.size())));
        BOOST_TEST_EQ(buff[expected.size()], 'x');
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
    static std::vector<std::string> seqs =
        { "\xDF"                    // not enough continuation bytes
        , "\xEF\x80"                // not enough continuation bytes
        , "\xF3\x80\x80"            // not enough continuation bytes
        , "\xff"                    // invalid byte
        , "\xff\x80\x80"            // invalid byte with continuation bytes
        , "\x80\x80"                // continuation bytes without leading byte
        , "\xED\xA0\x80"            // surrogate
        , "\xED\xAF\xBF"            // surrogate
        , "\xED\xB0\x80"            // surrogate
        , "\xED\xBF\xBF"            // surrogate
        , "\xED\xBF\xBF\x80\x80"    // surrogate with extra continuation bytes
        , "\xF5\x90\x80\x80"        // codepoint too big
        , "\xF5\x90\x80\x80\x80\x80"// codepoint too big with extra continuation bytes
        };

    return seqs;
}

const auto& invalid_sequences(const strf::encoding<char16_t>&)
{
    static const std::vector<std::u16string> seqs =
        { {(char16_t)0xD800}
        , {(char16_t)0xDBFF}
        , {(char16_t)0xDC00}
        , {(char16_t)0xDFFF} };

    return seqs;
}

const auto& invalid_sequences(const strf::encoding<char32_t>&)
{
    static const std::vector<std::u32string> seqs =
        { {(char32_t)0xD800}
        , {(char32_t)0xDBFF}
        , {(char32_t)0xDC00}
        , {(char32_t)0xDFFF}
        , {(char32_t)0x110000} };

    return seqs;
}

const auto& invalid_sequences(const strf::encoding<wchar_t>&)
{
    static const std::vector<std::wstring> seqs =
        { {(wchar_t)0xD800}
        , {(wchar_t)0xDBFF}
        , {(wchar_t)0xDC00}
        , {(wchar_t)0xDFFF}
        , {wchar_t(sizeof(wchar_t) == 4 ? 0x110000 : 0xDFFF)} };

    return seqs;
}

std::string    replacement_char(const strf::encoding<char>&){ return u8"\uFFFD";}
std::u16string replacement_char(const strf::encoding<char16_t>&){ return u"\uFFFD";}
std::u32string replacement_char(const strf::encoding<char32_t>&){ return U"\uFFFD";}
std::wstring   replacement_char(const strf::encoding<wchar_t>&){ return L"\uFFFD";}


template <typename CharIn, typename CharOut>
std::string invalid_input_test_label
    ( const strf::encoding<CharIn>& ein
    , const auto& seq
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

    for(const auto& seq : invalid_sequences(ein))
    {
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
                                 + eout.replacement_char_size );

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
            const auto expected = prefix_out;
            std::fill(buff, buff_end, 'x');

            auto src_it = &*input.begin();
            auto dest_it = buff;
            auto res = cv->transcode( &src_it
                                    , &*input.end()
                                    , &dest_it
                                    , buff_end
                                    , strf::error_handling::stop
                                    , false );

            BOOST_TEST_EQ(res, strf::cv_result::invalid_char);
            BOOST_TEST_EQ((src_it - &*input.begin())
                         , as_signed(prefix_in.size() + seq.size()));
            BOOST_TEST_EQ((dest_it - buff), as_signed(expected.size()));
            BOOST_TEST(std::equal( expected.begin()
                                 , expected.end()
                                 , buff ));
            BOOST_TEST_EQ(buff[expected.size()], 'x');

            // convert remaining input

            auto src_it2 = src_it;
            auto dest_it2 = dest_it;
            auto res2 = cv->transcode( &src_it2
                                     , &*input.end()
                                     , &dest_it2
                                     , buff_end
                                     , strf::error_handling::stop
                                     , false );

            const auto expected2 = prefix_out + suffix_out;

            BOOST_TEST_EQ(res2, strf::cv_result::success);
            BOOST_TEST_EQ(src_it2, &*input.end());
            BOOST_TEST_EQ((dest_it2 - buff), as_signed(expected2.size()));
            BOOST_TEST(std::equal( expected2.begin()
                                 , expected2.end()
                                 , buff ));

        }

        {   // convert replacing invalid sequence
            const auto expected = prefix_out + replacement_char(eout) + suffix_out;
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
            const auto expected2 = prefix_out + replacement_char(eout) + suffix_out;

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
    auto encodings = hana::make_tuple( strf::utf8(), strf::utf16()
                                     , strf::utf32(), strf::wchar_encoding());

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


    // TODO: test utf8 input with overlong sequences

    return boost::report_errors();
}
