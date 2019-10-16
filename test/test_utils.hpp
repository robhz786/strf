#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/stringify.hpp>
#include <cctype>

#include "lightweight_test_label.hpp"

namespace test_utils {

std::string unique_tmp_file_name()
{

#if defined(_WIN32)

    char dirname[MAX_PATH];
    auto dirlen = GetTempPathA(MAX_PATH, dirname);
    char fullname[MAX_PATH];
    sprintf_s(fullname, MAX_PATH, "%s\\test_boost_outbuf_%x.txt", dirname, std::rand());
    return fullname;

#else // defined(_WIN32)

   char fullname[200];
   sprintf(fullname, "/tmp/test_boost_outbuf_%x.txt", std::rand());
   return fullname;

#endif  // defined(_WIN32)
}

std::wstring read_wfile(std::FILE* file)
{
    std::wstring result;
    wint_t ch = fgetwc(file);
    while(ch != WEOF)
    {
        result += static_cast<wchar_t>(ch);
        ch = fgetwc(file);
    };

    return result;
}

std::wstring read_wfile(const char* filename)
{
    std::wstring result;

#if defined(_WIN32)

    std::FILE* file = NULL;
    (void) fopen_s(&file, filename, "r");

#else // defined(_WIN32)

    std::FILE* file = std::fopen(filename, "r");

#endif  // defined(_WIN32)

    if(file != nullptr)
    {
        result = read_wfile(file);
        fclose(file);
    }
    return result;
}

template <typename CharT>
std::basic_string<CharT> read_file(std::FILE* file)
{
    constexpr std::size_t buff_size = 500;
    CharT buff[buff_size];
    std::basic_string<CharT> result;
    std::size_t read_size = 0;
    do
    {
        read_size = std::fread(buff, sizeof(buff[0]), buff_size, file);
        result.append(buff, read_size);
    }
    while(read_size == buff_size);

    return result;
}

template <typename CharT>
std::basic_string<CharT> read_file(const char* filename)
{
    std::basic_string<CharT> result;

#if defined(_WIN32)

    std::FILE* file = nullptr;
    (void) fopen_s(&file, filename, "r");

#else // defined(_WIN32)

    std::FILE* file = std::fopen(filename, "r");

#endif  // defined(_WIN32)


    if(file != nullptr)
    {
        result = read_file<CharT>(file);
    }
    if (file != nullptr)
    {
        fclose(file);
    }

    return result;
}

template <typename CharT>
struct char_generator
{
    CharT operator()()
    {
        ch = ch == 0x79 ? 0x20 : (ch + 1);
        return ch;
    }
    CharT ch = 0x20;
};

template <typename CharT>
std::basic_string<CharT> make_string(std::size_t size)
{
    std::basic_string<CharT> str(size, CharT('x'));
    char_generator<CharT> gen;
    std::generate(str.begin(), str.end(), gen);
    return str;
}

template <typename CharT>
inline std::basic_string<CharT> make_half_string()
{
    constexpr auto bufsize = boost::stringify::v0::min_size_after_recycle<CharT>();
    return make_string<CharT>(bufsize / 2);
}

template <typename CharT>
inline std::basic_string<CharT> make_full_string()
{
    constexpr auto bufsize = boost::stringify::v0::min_size_after_recycle<CharT>();
    return make_string<CharT>(bufsize);
}

template <typename CharT>
inline std::basic_string<CharT> make_double_string()
{
    constexpr auto bufsize = boost::stringify::v0::min_size_after_recycle<CharT>();
    return make_string<CharT>(2 * bufsize);
}

template <typename CharT>
std::basic_string<CharT> make_tiny_string()
{
    return make_string<CharT>(5);
}

template <typename CharT>
inline void turn_into_bad(boost::stringify::v0::basic_outbuf<CharT>& ob)
{
    boost::stringify::v0::detail::outbuf_test_tool::turn_into_bad(ob.as_underlying());
}

template <typename CharOut>
class input_tester
    : public boost::stringify::v0::basic_outbuf<CharOut>
{

public:

    input_tester
        ( std::basic_string<CharOut> expected
        , const char* src_filename
        , int src_line
        , const char* function
        , double reserve_factor );

    input_tester
        ( std::basic_string<CharOut> expected
        , const char* src_filename
        , int src_line
        , const char* function
        , std::error_code err
        , double reserve_factor );

    ~input_tester();

    using char_type = CharOut;

    void recycle() override;

    void finish();

    void reserve(std::size_t size);

private:

    template <typename ... MsgArgs>
    void _test_failure(const MsgArgs&... msg_args)
    {
        _test_failed = true;
        boost::stringify::v0::append(_failure_msg)(msg_args...);
    }

    bool _wrongly_reserved() const;

    bool _too_much_reserved() const;

    std::basic_string<CharOut> _result;
    std::basic_string<CharOut> _expected;
    std::string _failure_msg;
    std::size_t _reserved_size;
    const char* _src_filename;
    const char* _function;
    int _src_line;
    double _reserve_factor;

    std::error_code _expected_error;
    bool _expect_error = false;
    bool _recycle_called = false;
    bool _source_location_printed = false;
    bool _test_failed = false;
};


template <typename CharOut>
input_tester<CharOut>::input_tester
    ( std::basic_string<CharOut> expected
    , const char* src_filename
    , int src_line
    , const char* function
    , double reserve_factor )
    : boost::stringify::v0::basic_outbuf<CharOut>{nullptr, nullptr}
    , _expected(std::move(expected))
    , _reserved_size(0)
    , _src_filename(std::move(src_filename))
    , _function(function)
    , _src_line(src_line)
    , _reserve_factor(reserve_factor)
{
}

template <typename CharOut>
input_tester<CharOut>::~input_tester()
{
}

template <typename CharOut>
void input_tester<CharOut>::reserve(std::size_t size)
{
    _reserved_size = size;
    if (size != 0)
    {
        _result.resize(size, CharOut{'#'});
        this->set_pos(&*_result.begin());
        this->set_end(&*_result.begin() + size);
    }
}

template <typename CharOut>
void input_tester<CharOut>::recycle()
{
    _test_failure(" basic_outbuf::recycle() called "
                  "( return of printer::necessary_size() too small ).\n");

    std::size_t previous_size = 0;
    if (this->pos() != nullptr)
    {
       previous_size = this->pos() - &*_result.begin();
       _result.resize(previous_size);
    }
    _result.append(boost::stringify::v0::min_size_after_recycle<CharOut>(), CharOut{'#'});
    this->set_pos(&*_result.begin() + previous_size);
    this->set_end(&*_result.begin() + _result.size());
}

template <typename CharOut>
void input_tester<CharOut>::finish()
{
    _result.resize(this->pos() - &*_result.begin());

    if (_expected != _result)
    {
        namespace strf = boost::stringify::v0;

        _test_failure( "\n expected: \"", strf::cv(_expected), '\"'
                     , "\n obtained: \"", strf::cv(_result), "\"\n" );

    }
    if(_wrongly_reserved())
    {
        _test_failure( "\n reserved size  : ", _reserved_size
                     , "\n necessary size : ", _result.length(), '\n' );
    }

    if (_test_failed)
    {
        ::boost::detail::error_impl( _failure_msg.c_str(), _src_filename
                                   , _src_line, _function);
    }
}

template <typename CharOut>
bool input_tester<CharOut>::_wrongly_reserved() const
{
    return (_reserved_size < _result.length() || _too_much_reserved());
}

template <typename CharOut>
bool input_tester<CharOut>::_too_much_reserved() const
{
    return
        static_cast<double>(_reserved_size) /
        static_cast<double>(_result.length())
        > _reserve_factor;
}

template<typename CharT>
auto make_tester
   ( const CharT* expected
   , const char* filename
   , int line
   , const char* function
   , double reserve_factor = 1.0 )
{
   using writer = test_utils::input_tester<CharT>;
   return boost::stringify::v0::dispatcher
       < boost::stringify::v0::facets_pack<>
       , writer, const CharT*, const char*, int
       , const char*, double >
       ( expected, filename, line, function, reserve_factor);
}

template<typename CharT>
auto make_tester
   ( const std::basic_string<CharT>& expected
   , const char* filename
   , int line
   , const char* function
   , double reserve_factor = 1.0 )
{
   using writer = test_utils::input_tester<CharT>;
   return boost::stringify::v0::dispatcher
       < boost::stringify::v0::facets_pack<>
       , writer, const std::basic_string<CharT>&, const char*
       , int, const char*, double>
       ( expected, filename, line, function, reserve_factor);
}

} // namespace test_utils

#define TEST(EXPECTED)                                                  \
    test_utils::make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION) \
    .reserve_calc()

#define TEST_RF(EXPECTED, RF)                                           \
    test_utils::make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (RF)) \
    .reserve_calc()

#endif
