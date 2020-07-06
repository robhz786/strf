#ifndef STRF_TEST_TEST_UTILS_HPP_INCLUDED
#define STRF_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if ! defined(STRF_FREESTANDING)
#  include <cstdio>
#  include <string>
#  define STRF_HAS_STD_STRING_DECLARATION
#endif

#include <strf.hpp>
//#include <cctype>

#if defined(_WIN32)
#include <windows.h>
#endif  // defined(_WIN32)

#include "boost/current_function.hpp"

namespace test_utils {

#if ! defined(STRF_FREESTANDING)

std::string unique_tmp_file_name();

std::wstring read_wfile(std::FILE* file);

std::wstring read_wfile(const char* filename);

template <typename CharT>
std::basic_string<CharT> read_file(std::FILE* file)
{
    constexpr std::size_t buff_size = 500;
    CharT buff[buff_size];
    std::basic_string<CharT> result;
    std::size_t read_size = 0;
    do {
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


    if(file != nullptr) {
        result = read_file<CharT>(file);
    }
    if (file != nullptr) {
        fclose(file);
    }

    return result;
}

#endif // ! defined(STRF_FREESTANDING)

template <typename CharT>
constexpr std::size_t full_string_size
= strf::min_size_after_recycle<CharT>();

template <typename CharT>
constexpr std::size_t half_string_size = full_string_size<CharT> / 2;

template <typename CharT>
constexpr std::size_t double_string_size = full_string_size<CharT> * 2;

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_HD make_double_string()
{
    enum {arr_size = double_string_size<CharT>};
    static const CharT arr[arr_size]
      = { 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27
        , 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f
        , 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37
        , 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f
        , 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47
        , 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f
        , 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57
        , 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f
        , 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67
        , 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f
        , 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77
        , 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f

        , 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27
        , 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f
        , 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37
        , 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f };

    return {arr, arr_size};
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_HD make_full_string()
{
    return { make_double_string<CharT>().begin(), full_string_size<CharT> };
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_HD make_half_string()
{
    return { make_double_string<CharT>().begin(), half_string_size<CharT> };
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_HD make_tiny_string()
{
    static const CharT arr[5] = {'H', 'e', 'l', 'l', 'o'};
    return {arr, 5};
}

template <typename CharT>
inline void STRF_HD turn_into_bad(strf::basic_outbuff<CharT>& ob)
{
    strf::detail::outbuff_test_tool::turn_into_bad(ob);
}

int& STRF_HD test_err_count();

strf::outbuff& STRF_HD test_outbuff();

class test_scope
{
public:

    STRF_HD test_scope(const test_scope&) = delete;

    STRF_HD  test_scope()
        : parent_(curr_test_scope())
    {
        parent_->child_ = this;
        curr_test_scope() = this;
        description_[0] = '\0';
    }

    STRF_HD ~test_scope()
    {
        if (parent_) {
            parent_->child_ = child_;
            curr_test_scope() = parent_;
        }
        if (child_) {
            child_ -> parent_ = parent_;
        }
    }

    auto STRF_HD description_writer()
    {
        return strf::to(description_);
    }

    static void STRF_HD print_stack(strf::outbuff& out)
    {
        test_scope* first = root().child_;
        if (first != nullptr) {
            strf::write(out, "\n    ( ");
            strf::write(out, first->description_);
            for(auto it = first->child_; it != nullptr; it = it->child_) {
                strf::write(out, " / ");
                strf::write(out, it->description_);
            }
            strf::write(out, " )");
        }
    }

private:

    struct root_tag {};

    STRF_HD test_scope(root_tag)
    {
        description_[0] = '\0';
    }

    static test_scope& STRF_HD root()
    {
        static test_scope r{test_scope::root_tag{}};
        return r;
    }

    static test_scope*& STRF_HD curr_test_scope()
    {
        static test_scope* curr = &root();
        return curr;
    }


    test_scope* parent_ = nullptr;
    test_scope* child_ = nullptr;
    char description_[200];
};

template <typename ... Args>
void STRF_HD test_message
    ( const char* filename
    , int line
    , const char* funcname
    , const Args& ... args )
{
    to(test_outbuff()) (filename, ':', line, ": ", args...);
    test_scope::print_stack(test_outbuff());
    to(test_outbuff()) ("\n    In function '", funcname, "'\n");
}


template <typename ... Args>
void STRF_HD test_failure
    ( const char* filename
    , int line
    , const char* funcname
    , const Args& ... args )
{
    ++ test_err_count();
    test_message(filename, line, funcname, args...);
}

template <typename CharOut>
class STRF_HD input_tester
    : public strf::basic_outbuff<CharOut>
{

public:

    STRF_HD input_tester
        ( strf::detail::simple_string_view<CharOut> expected
        , const char* src_filename
        , int src_line
        , const char* function
        , double reserve_factor
        , std::size_t size = 0 );

#ifdef STRF_NO_CXX17_COPY_ELISION

    STRF_HD input_tester(input_tester&& r);

#else

    STRF_HD input_tester(input_tester&& r) = delete;
    STRF_HD input_tester(const input_tester& r) = delete;

#endif

    STRF_HD ~input_tester();

    using char_type = CharOut;

    void STRF_HD recycle() override;

    void STRF_HD finish();

private:

    template <typename ... MsgArgs>
    void STRF_HD test_failure_(const MsgArgs&... msg_args)
    {
        test_utils::test_failure( src_filename_, src_line_
                                , function_, msg_args... );
    }

    bool STRF_HD wrongly_reserved_(std::size_t result_size) const;

    strf::detail::simple_string_view<CharOut> expected_;
    std::size_t reserved_size_;
    const char* src_filename_;
    const char* function_;
    int src_line_;
    double reserve_factor_;

    bool expect_error_ = false;
    bool recycle_called_ = false;
    bool source_location_printed_ = false;

    CharOut* pointer_before_overflow_ = nullptr;
    //constexpr static std::size_t buffer_size_ = 500;
    enum {buffer_size_ = 500};
    CharOut buffer_[buffer_size_];
};


template <typename CharOut>
STRF_HD input_tester<CharOut>::input_tester
    ( strf::detail::simple_string_view<CharOut> expected
    , const char* src_filename
    , int src_line
    , const char* function
    , double reserve_factor
    , std::size_t size )
    : strf::basic_outbuff<CharOut>{buffer_, size}
    , expected_(expected)
    , reserved_size_(size)
    , src_filename_(src_filename)
    , function_(function)
    , src_line_(src_line)
    , reserve_factor_(reserve_factor)
{
    if (size > buffer_size_) {
        test_utils::test_message
            ( src_filename_, src_line_, function_
            , "Warning: reserved more characters (", size
            , ") then the tester buffer size (", buffer_size_, ")." );
        this->set_end(buffer_ + buffer_size_);
    }
}

template <typename CharOut>
STRF_HD input_tester<CharOut>::~input_tester()
{
}

template <typename CharOut>
void STRF_HD input_tester<CharOut>::recycle()
{
    test_failure_(" basic_outbuff::recycle() called "
                  "( calculated size too small ).\n");

    if ( this->pointer() + strf::min_size_after_recycle<CharOut>()
       > buffer_ + buffer_size_ )
    {
        pointer_before_overflow_ = this->pointer();
        this->set_pointer(strf::outbuff_garbage_buf<CharOut>());
        this->set_end(strf::outbuff_garbage_buf_end<CharOut>());
    } else {
        this->set_end(buffer_ + buffer_size_);
    }
}

template <typename CharOut>
void STRF_HD input_tester<CharOut>::finish()
{
    auto pointer = pointer_before_overflow_ ? pointer_before_overflow_ : this->pointer();
    strf::detail::simple_string_view<CharOut> result{buffer_, pointer};

    if ( result.size() != expected_.size()
      || ! strf::detail::str_equal<CharOut>( expected_.begin()
                                           , result.begin()
                                           , expected_.size() ))
    {
        test_failure_( "\n  expected: \"", strf::conv(expected_), '\"'
                     , "\n  obtained: \"", strf::conv(result), "\"" );

    }
    if(wrongly_reserved_(result.size())) {
        test_failure_( "\n  reserved size  : ", reserved_size_
                     , "\n  necessary size : ", result.size() );
    }
}

template <typename CharOut>
bool STRF_HD input_tester<CharOut>::wrongly_reserved_(std::size_t result_size) const
{
    return ( reserved_size_ < result_size
          || ( ( static_cast<double>(reserved_size_)
               / static_cast<double>(result_size) )
             > reserve_factor_ ) );
}

template <typename CharT>
class input_tester_creator
{
public:

    using char_type = CharT;

    STRF_HD input_tester_creator
        ( strf::detail::simple_string_view<CharT> expected
        , const char* filename
        , int line
        , const char* function
        , double reserve_factor )
        : expected_(expected)
        , filename_(filename)
        , function_(function)
        , line_(line)
        , reserve_factor_(reserve_factor)
    {
    }

    input_tester_creator(const input_tester_creator& ) = default;
    input_tester_creator(input_tester_creator&& ) = default;

    test_utils::input_tester<CharT> STRF_HD create(std::size_t size) const
    {
        return test_utils::input_tester<CharT>
            { expected_, filename_, line_, function_, reserve_factor_, size };
    }

private:

    strf::detail::simple_string_view<CharT> expected_;
    const char* filename_;
    const char* function_;
    int line_;
    double reserve_factor_ = 1.0;
};

template<typename CharT>
auto STRF_HD make_tester
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
auto STRF_HD make_tester
   ( strf::detail::simple_string_view<CharT> expected
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
constexpr bool STRF_HD equal(const T&a, const U&b)
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
    test_utils::make_tester( (EXPECTED), __FILE__, __LINE__             \
                           , BOOST_CURRENT_FUNCTION)

#define TEST_RF(EXPECTED, RF)                                           \
    test_utils::make_tester( (EXPECTED), __FILE__, __LINE__             \
                           , BOOST_CURRENT_FUNCTION, (RF))

#define TEST_STR_CONCAT_2(str1, str2) str1 ## str2

#define TEST_STR_CONCAT(str1, str2) TEST_STR_CONCAT_2(str1, str2)

#define TEST_SCOPE_DESCRIPTION                                          \
    test_utils::test_scope TEST_STR_CONCAT(test_label_, __LINE__);      \
    TEST_STR_CONCAT(test_label_, __LINE__).description_writer()

#define TEST_ERROR(msg) \
    test_utils::test_failure(__FILE__, __LINE__, BOOST_CURRENT_FUNCTION, (msg));

#define TEST_TRUE(expr)                                                 \
    if (!(expr))                                                        \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , "test (" #expr ") failed. " );                            \

#define TEST_FALSE(expr)                                                \
    if ((expr))                                                         \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , "test (" #expr ") failed. " );                            \

#define TEST_EQ(a, b)                                                   \
    if (!test_utils::equal((a), (b)))                                   \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , " test (", (a), " == ", (b), ") failed. " );

#define TEST_CSTR_EQ(s1, s2)                                            \
    for ( std::size_t len1 = strf::detail::str_length(s1)               \
        ; len1 == strf::detail::str_length(s2); ) {                     \
        if (! strf::detail::str_equal(s1, s2, len1))                    \
            test_utils::test_failure                                    \
                ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION            \
                , "test (s1 == s2) failed. Where:\n    s1 is \"", (s1)  \
                , "\"\n    s2 is \"", (s2), '\"' );                     \
        break;                                                          \
    }                                                                   \

#define TEST_THROWS( EXPR, EXCEP )                                      \
  { bool caught = false;                                                \
    try { EXPR; }                                                       \
    catch(EXCEP const&) { caught = true; }                              \
    if (!caught)                                                        \
          test_utils::test_failure                                      \
              ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION              \
              , "exception " #EXCEP " not thrown as expected" );        \
  }

#endif // defined(STRF_TEST_TEST_UTILS_HPP_INCLUDED)
