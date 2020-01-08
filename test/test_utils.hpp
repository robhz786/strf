#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <cctype>
#include <algorithm> // for std::generate.

#if defined(_WIN32)
#include <windows.h>
#endif  // defined(_WIN32)

#include "boost/current_function.hpp"

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

int& test_err_count()
{
    static int count = 0;
    return count;
}

strf::narrow_cfile_writer<char>& test_outbuf()
{
    static strf::narrow_cfile_writer<char> ob(stdout);
    return ob;
}

class test_scope
{
public:

    test_scope(const test_scope&) = delete;

    test_scope()
        : parent_(curr_test_scope())
    {
        parent_->child_ = this;
        curr_test_scope() = this;
        description_[0] = '\0';
    }

    ~test_scope()
    {
        if (parent_) {
            parent_->child_ = child_;
        }
        if (child_) {
            child_ -> parent_ = parent_;
        }
    }

    auto description_writer()
    {
        return strf::to(description_);
    }

    static void print_stack(strf::outbuf& out)
    {
        test_scope* first = root().child_;
        if (first != nullptr) {
            strf::write(out, "\n    At ");
        }
        for(auto it = first; it != nullptr; it = it->child_) {
            strf::write(out, it->description_);
            strf::put(out, '/');
        }
        if (first != nullptr) {
            strf::put(out, '\n');
        }
    }

private:

    struct root_tag {};

    test_scope(root_tag)
    {
        description_[0] = '\0';
    }

    static test_scope& root()
    {
        static test_scope r{test_scope::root_tag{}};
        return r;
    }

    static test_scope*& curr_test_scope()
    {
        static test_scope* curr = &root();
        return curr;
    }


    test_scope* parent_ = nullptr;
    test_scope* child_ = nullptr;
    char description_[200];
};

template <typename ... Args>
auto test_failure
    ( const char* filename
    , int line
    , const char* funcname
    , const Args& ... args )
{
    ++ test_err_count();
    to(test_outbuf()) (filename, ':', line, ": ", args...);
    test_scope::print_stack(test_outbuf());
    to(test_outbuf()) ("\n    In function '", funcname, "'\n");
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

#ifdef STRF_NO_CXX17_COPY_ELISION

    input_tester(input_tester&& r);

#else

    input_tester(input_tester&& r) = delete;
    input_tester(const input_tester& r) = delete;

#endif

    ~input_tester();

    using char_type = CharOut;

    void recycle() override;

    void finish();

    void reserve(std::size_t size);

private:

    template <typename ... MsgArgs>
    void _test_failure(const MsgArgs&... msg_args)
    {
        test_utils::test_failure( _src_filename, _src_line
                                , _function, msg_args... );
    }

    bool _wrongly_reserved() const;

    bool _too_much_reserved() const;

    std::basic_string<CharOut> _result;
    std::basic_string<CharOut> _expected;
    std::size_t _reserved_size;
    const char* _src_filename;
    const char* _function;
    int _src_line;
    double _reserve_factor;

    bool _expect_error = false;
    bool _recycle_called = false;
    bool _source_location_printed = false;
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
                  "( calculated size too small ).\n");

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
        _test_failure( "\n  expected: \"", strf::cv(_expected), '\"'
                     , "\n  obtained: \"", strf::cv(_result), "\"" );

    }
    if(_wrongly_reserved())
    {
        _test_failure( "\n  reserved size  : ", _reserved_size
                     , "\n  necessary size : ", _result.length() );
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

    test_utils::input_tester<CharT> create(std::size_t size) const
    {
        return test_utils::input_tester<CharT>
            { _expected, _filename, _line, _function, _reserve_factor, size };
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
   return strf::destination_calc_size
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
   return strf::destination_calc_size
       < test_utils::input_tester_creator<CharT> >
       ( expected, filename, line, function, reserve_factor);
}

#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable: 4389)
#elif defined(__clang__) && defined(__has_warning)
# if __has_warning("-Wsign-compare")
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wsign-compare"
# endif
#elif defined(__GNUC__) && !(defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)) && (__GNUC__ * 100 + __GNUC_MINOR__) >= 406
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsign-compare"
#endif


template <typename T, typename U>
constexpr bool equal(const T&a, const U&b)
{
    return a == b;
}


#if defined(_MSC_VER)
# pragma warning(pop)
#elif defined(__clang__) && defined(__has_warning)
# if __has_warning("-Wsign-compare")
#  pragma clang diagnostic pop
# endif
#elif defined(__GNUC__) && !(defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)) && (__GNUC__ * 100 + __GNUC_MINOR__) >= 406
# pragma GCC diagnostic pop
#endif


} // namespace test_utils

#define TEST(EXPECTED)                                                  \
    test_utils::make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION)

#define TEST_RF(EXPECTED, RF)                                           \
    test_utils::make_tester((EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (RF))

#define TEST_STR_CONCAT(str1, str2) str1 ## str2

#define TEST_LABEL_IMPL(LINE)                                           \
    test_utils::test_scope TEST_STR_CONCAT(test_label_, LINE);          \
    TEST_STR_CONCAT(test_label_, LINE).description_writer()

#define BOOST_TEST_LABEL   TEST_LABEL_IMPL(__LINE__)

#define BOOST_ERROR(msg) \
    test_utils::test_failure(__FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (msg));

#define BOOST_TEST(expr)                                                \
    if (!(expr))                                                        \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , "test (" #expr ") failed. " );                            \

#define BOOST_TEST_EQ(a, b)                                             \
    if (!test_utils::equal((a), (b)))                                   \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , " test (", (a), " == ", (b), ") failed. " );

#define BOOST_TEST_CSTR_EQ(a, b)                                        \
    if (0 != std::strcmp(a, b))                                         \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , "test (s1 == s2) failed. Where:\n    s1 is \"", (a)     \
            , "\"\n    s2 is \"", (b), '\"' );

#define BOOST_TEST_THROWS( EXPR, EXCEP )                                \
  { bool caught = false;                                                \
    try { EXPR; }                                                       \
    catch(EXCEP const&) { caught = true; }                              \
    if (!caught)                                                        \
          test_utils::test_failure                                      \
              ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION              \
              , "exception " #EXCEP " not thrown as expected" );        \
  }


int test_finish()
{
    int err_count = test_utils::test_err_count();
    if (err_count == 0) {
        strf::write(test_utils::test_outbuf(), "All test passed!\n");
    }
    else {
        strf::to(test_utils::test_outbuf()) (err_count, " tests failed!\n");
        // auto digcount = strf::detail::count_digits<10>(err_count);
        // strf::detail::write_int(ob, err_count, digcount);
        // strf::write(ob, " tests failed!\n");
    }
    test_utils::test_outbuf().finish();
    return err_count;
}

namespace boost {

inline int report_errors()
{
    return test_finish();
}

}

#endif
