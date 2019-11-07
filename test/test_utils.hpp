#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <stringify.hpp>
#include <cctype>

#include "lightweight_test_label.hpp"

#if defined(_WIN32)
#include <windows.h>
#endif  // defined(_WIN32)

namespace test_utils {

std::string unique_tmp_file_name()
{

#if defined(_WIN32)

    char dirname[MAX_PATH];
    GetTempPathA(MAX_PATH, dirname);
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
constexpr std::size_t full_string_size
= strf::min_size_after_recycle<CharT>();

template <typename CharT>
constexpr std::size_t half_string_size = full_string_size<CharT> / 2;

template <typename CharT>
constexpr std::size_t double_string_size = full_string_size<CharT> * 2;

template <typename CharT>
inline std::basic_string<CharT> make_half_string()
{
    return make_string<CharT>(half_string_size<CharT>);
}

template <typename CharT>
inline std::basic_string<CharT> make_full_string()
{
    return make_string<CharT>(full_string_size<CharT>);
}

template <typename CharT>
inline std::basic_string<CharT> make_double_string()
{
    return make_string<CharT>(double_string_size<CharT>);
}

template <typename CharT>
std::basic_string<CharT> make_tiny_string()
{
    return make_string<CharT>(5);
}

template <typename CharT>
inline void turn_into_bad(strf::basic_outbuf<CharT>& ob)
{
    strf::detail::outbuf_test_tool::turn_into_bad(ob.as_underlying());
}

template <typename CharOut>
class input_tester
    : public strf::basic_outbuf<CharOut>
{

public:

    input_tester
        ( std::basic_string<CharOut> expected
        , const char* src_filename
        , int src_line
        , const char* function
        , double reserve_factor
        , std::size_t size = 0 );

    input_tester(input_tester&& r) = delete;

    input_tester(const input_tester& r) = delete;

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
        strf::append(_failure_msg)(msg_args...);
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
    , double reserve_factor
    , std::size_t size )
    : strf::basic_outbuf<CharOut>{nullptr, nullptr}
    , _result(size, CharOut{'#'})
    , _expected(std::move(expected))
    , _reserved_size(size)
    , _src_filename(std::move(src_filename))
    , _function(function)
    , _src_line(src_line)
    , _reserve_factor(reserve_factor)
{
    if ( ! _result.empty())
    {
        this->set_pos(&*_result.begin());
        this->set_end(&*_result.begin() + size);
    }
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
    _result.append(strf::min_size_after_recycle<CharOut>(), CharOut{'#'});
    this->set_pos(&*_result.begin() + previous_size);
    this->set_end(&*_result.begin() + _result.size());
}

template <typename CharOut>
void input_tester<CharOut>::finish()
{
    if (!_result.empty())
    {
        _result.resize(this->pos() - &*_result.begin());
    }
    if (_expected != _result)
    {
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

template <typename CharT>
class input_tester_creator
{
public:

    using char_type = CharT;

    input_tester_creator( std::basic_string<CharT> expected
                        , const char* filename
                        , int line
                        , const char* function
                        , double reserve_factor )
        : _expected(std::move(expected))
        , _filename(filename)
        , _function(function)
        , _line(line)
        , _reserve_factor(reserve_factor)
    {
    }

    input_tester_creator(const input_tester_creator& ) = default;
    input_tester_creator(input_tester_creator&& ) = default;

    template <typename ... Printers>
    void sized_write(std::size_t size, const Printers& ... printers) const
    {
        test_utils::input_tester<CharT> ob
            { _expected, _filename, _line, _function, _reserve_factor, size };
        strf::detail::write_args(ob, printers...);
        ob.finish();
    }

private:

    std::basic_string<CharT> _expected;
    const char* _filename;
    const char* _function;
    int _line;
    double _reserve_factor = 1.0;
};


template<typename CharT>
auto make_tester
   ( const CharT* expected
   , const char* filename
   , int line
   , const char* function
   , double reserve_factor = 1.0 )
{
   return strf::dispatcher_calc_size
       < test_utils::input_tester_creator<CharT> >
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
   return strf::dispatcher_calc_size
       < test_utils::input_tester_creator<CharT> >
       ( expected, filename, line, function, reserve_factor);
}

} // namespace test_utils

#define TEST(EXPECTED)                                                  \
    test_utils::make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION)

#define TEST_RF(EXPECTED, RF)                                           \
    test_utils::make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (RF))

#endif
