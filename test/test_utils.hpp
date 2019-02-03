#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/stringify.hpp>
#include <cctype>

inline int& global_errors_count()
{
    static int x = 0;
    return x;
}

inline int report_errors()
{
    if (global_errors_count())
    {
        std::cout << global_errors_count() << " tests failed\n";
    }
    else
    {
        std::cout << "No errors found\n";
    }

    return global_errors_count();
}


inline void print(const char* label, const std::u16string& str)
{
    std::cout << label << "\n";
    for(auto it = str.begin(); it != str.end(); ++it)
    {
        printf("%4x ", (unsigned)*it);
    }
    std::cout << "\n";
}

inline void print(const char* label, const std::u32string& str)
{
    std::cout << label << "\n";
    for(auto it = str.begin(); it != str.end(); ++it)
    {
        printf("%8x ", (unsigned)*it);
    }
    std::cout << "\n";
}

inline void print(const char* label, const std::string& str)
{
    std::cout << label << ": \"" << str << "\"\n";
}

inline void print(const char* label, const std::wstring& str)
{
    std::cout << label << ": \"";
    std::wcout << str;
    std::cout  << "\"\n";
}


template <typename CharOut>
class input_tester
    : public boost::stringify::v0::output_buffer<CharOut>
{

public:

    input_tester
        ( std::basic_string<CharOut> expected
        , std::string src_filename
        , int src_line
        , double reserve_factor )
        : boost::stringify::v0::output_buffer<CharOut>{nullptr, nullptr}
        , _expected(std::move(expected))
        , _reserved_size(0)
        , _src_filename(std::move(src_filename))
        , _src_line(src_line)
        , _reserve_factor(reserve_factor)
    {
    }

    input_tester
        ( std::basic_string<CharOut> expected
        , std::string src_filename
        , int src_line
        , std::error_code err
        , double reserve_factor )
        : boost::stringify::v0::output_buffer<CharOut>{nullptr, nullptr}
        , _expected(std::move(expected))
        , _reserved_size(0)
        , _src_filename(std::move(src_filename))
        , _src_line(src_line)
        , _reserve_factor(reserve_factor)
        , _expected_error(err)
        , _expect_error(true)
    {
    }

    ~input_tester()
    {
    }

    using char_type = CharOut;

    bool recycle() override;

    boost::stringify::v0::expected<void, std::error_code> finish();

    void reserve(std::size_t size);

private:

    void test_failed();

    bool wrongly_reserved() const;

    bool too_much_reserved() const;

    std::basic_string<CharOut> _result;
    std::basic_string<CharOut> _expected;
    std::size_t _reserved_size;
    std::string _src_filename;
    int _src_line;
    double _reserve_factor;

    std::error_code _expected_error;
    bool _expect_error = false;
    bool _recycle_called = false;
    bool _source_location_printed = false;
};


template <typename CharOut>
void input_tester<CharOut>::reserve(std::size_t size)
{
    _reserved_size = size;
    _result.resize(size, CharOut{'#'});
    this->set_pos(&*_result.begin());
    this->set_end(&*_result.begin() + size);
}

template <typename CharOut>
bool input_tester<CharOut>::recycle()
{
    test_failed();

    std::cout << " output_buffer::recycle() called "
        "( return of printer::necessary_size() too small ).\n";

    std::size_t previous_size = this->pos() - &*_result.begin();
    _result.resize(previous_size);
    _result.append(boost::stringify::v0::min_buff_size, CharOut{'#'});
    this->set_pos(&*_result.begin() + previous_size);
    this->set_end(&*_result.begin() + _result.size());
    return true;
}

template <typename CharOut>
boost::stringify::v0::expected<void, std::error_code>
input_tester<CharOut>::finish()
{
    _result.resize(this->pos() - &*_result.begin());

    if (_expected != _result)
    {
        test_failed();
        print("Expected", _expected);
        print("Obtained", _result);
    }
    if(wrongly_reserved())
    {
        test_failed();
        std::cout << "Reserved size  :" <<  _reserved_size << "\n";
        std::cout << "Necessary size :" <<  _result.length() << "\n";
    }

    if (_expect_error != this->has_error())
    {
        test_failed();
        if ( ! _expect_error)
        {
            print( "Obtained error_code: "
                 , this->get_error().message());
        }
        else
        {
            print( "Not obtained any error_code. Was expecting: "
                 , _expected_error.message());
        }

    }
    else if (_expected_error != this->get_error())
    {
        test_failed();
        print("Expected error_code: ", _expected_error.message());
        print("Obtained error_code: ", this->get_error().message());
    }

    if (this->has_error())
    {
        return {boost::stringify::v0::unexpect_t{}, std::error_code{}};
    }
    return {};
}

template <typename CharOut>
void input_tester<CharOut>::test_failed()
{
    if ( ! _source_location_printed)
    {
        std::cout << _src_filename << ":" << _src_line << ":" << " error: \n";
        _source_location_printed = true;
    }
    ++global_errors_count();
}

template <typename CharOut>
bool input_tester<CharOut>::wrongly_reserved() const
{
    return (_reserved_size < _result.length() || too_much_reserved());
}

template <typename CharOut>
bool input_tester<CharOut>::too_much_reserved() const
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
   , std::error_code err
   , double reserve_factor = 1.0 )
{
   using writer = input_tester<CharT>;
   return boost::stringify::v0::make_destination
       <writer, const CharT*, const char*, int, std::error_code, double>
       (expected, filename, line, err, reserve_factor);
}

template<typename CharT>
auto make_tester
   ( const CharT* expected
   , const char* filename
   , int line
   , double reserve_factor = 1.0 )
{
   using writer = input_tester<CharT>;
   return boost::stringify::v0::make_destination
       <writer, const CharT*, const char*, int, double>
       (expected, filename, line, reserve_factor);
}

#define TEST(EXPECTED)                                  \
    (void) make_tester((EXPECTED), __FILE__, __LINE__)  \
    .reserve_calc()

#define TEST_RF(EXPECTED, RF)                                 \
    (void) make_tester((EXPECTED), __FILE__, __LINE__, (RF))  \
    .reserve_calc()

#define TEST_ERR(EXPECTED, ERR)                                 \
    (void) make_tester((EXPECTED), __FILE__, __LINE__, (ERR)  ) \
    .reserve_calc()

#define TEST_ERR_RF(EXPECTED, ERR, RF)                              \
    (void) make_tester((EXPECTED), __FILE__, __LINE__, (ERR), (RF)) \
    .reserve_calc()

#endif
