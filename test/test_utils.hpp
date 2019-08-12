#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/stringify.hpp>
#include <cctype>

#include "lightweight_test_label.hpp"

template <typename CharOut>
class input_tester
    : public boost::basic_outbuf<CharOut>
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
    : boost::basic_outbuf<CharOut>{nullptr, nullptr}
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
    _result.append(boost::min_size_after_recycle<CharOut>(), CharOut{'#'});
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
   using writer = input_tester<CharT>;
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
   using writer = input_tester<CharT>;
   return boost::stringify::v0::dispatcher
       < boost::stringify::v0::facets_pack<>
       , writer, const std::basic_string<CharT>&, const char*
       , int, const char*, double>
       ( expected, filename, line, function, reserve_factor);
}

#define TEST(EXPECTED)                                                  \
    make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION) \
    .reserve_calc()

#define TEST_RF(EXPECTED, RF)                                           \
    make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (RF)) \
    .reserve_calc()

#endif
