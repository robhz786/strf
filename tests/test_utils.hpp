#ifndef STRF_TEST_TEST_UTILS_HPP_INCLUDED
#define STRF_TEST_TEST_UTILS_HPP_INCLUDED

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if ! defined(STRF_TEST_FUNC)
#define STRF_TEST_FUNC
#endif

#if ! defined(STRF_FREESTANDING)
#  include <cstdio>
#  include <cstdlib>
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

template <typename CharT>
void STRF_HD write_random_ascii_chars(CharT* dest, std::size_t count)
{
#ifdef STRF_FREESTANDING
    int r = 10;
#else
    int r = std::rand() % 20;
#endif
    CharT ch = static_cast<CharT>(0x20 + r);
    for (std::size_t i = 0; i < count; ++i) {
        dest[i] = ch;
        if (++ch == 0x7F) {
            ch = 0x20;
        }
    }
}

#if ! defined(STRF_FREESTANDING)

inline std::string unique_tmp_file_name()
{

#if defined(_WIN32)

    char dirname[MAX_PATH];
    GetTempPathA(MAX_PATH, dirname);
    char fullname[MAX_PATH];
    sprintf_s(fullname, MAX_PATH, "%s\\test_strf_%x.txt", dirname, std::rand());
    return fullname;

#else // defined(_WIN32)

   char fullname[200];
   sprintf(fullname, "/tmp/test_strf_%x.txt", std::rand());
   return fullname;

#endif  // defined(_WIN32)
}

inline std::wstring read_wfile(std::FILE* file)
{
    std::wstring result;
    wint_t ch = fgetwc(file);
    while(ch != WEOF) {
        result += static_cast<wchar_t>(ch);
        ch = fgetwc(file);
    };

    return result;
}

inline std::wstring read_wfile(const char* filename)
{
    std::wstring result;

#if defined(_WIN32)

    std::FILE* file = NULL;
    (void) fopen_s(&file, filename, "r");

#else // defined(_WIN32)

    std::FILE* file = std::fopen(filename, "r");

#endif  // defined(_WIN32)

    if(file != nullptr) {
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

template <typename CharT>
std::basic_string<CharT> make_random_std_string(std::size_t size)
{
    std::basic_string<CharT> str(size, (CharT)0);
    write_random_ascii_chars(&str[0], size);
    return str;
}

#endif // ! defined(STRF_FREESTANDING)

template <typename CharT>
constexpr STRF_HD  std::size_t full_string_size()
{
    return strf::min_destination_buffer_size;
}
template <typename CharT>
constexpr STRF_HD  std::size_t half_string_size()
{
    return full_string_size<CharT>() / 2;
}

template <typename CharT>
constexpr STRF_HD  std::size_t double_string_size()
{
    return full_string_size<CharT>() * 2;
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_HD make_double_string()
{
    enum {arr_size = double_string_size<CharT>()};
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
    return { make_double_string<CharT>().begin(), full_string_size<CharT>() };
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_HD make_half_string()
{
    return { make_double_string<CharT>().begin(), half_string_size<CharT>() };
}

template <typename CharT>
inline strf::detail::simple_string_view<CharT> STRF_HD make_tiny_string()
{
    static const CharT arr[5] = {'H', 'e', 'l', 'l', 'o'};
    return {arr, 5};
}

template <typename CharT>
inline void STRF_HD turn_into_bad(strf::output_buffer<CharT, 0>& dest)
{
    strf::detail::output_buffer_test_tool::turn_into_bad(dest);
}

inline STRF_HD int& test_err_count()
{
    static int count = 0;
    return count;
}

#if defined(__CUDACC__)

STRF_HD strf::destination<char>&  test_messages_destination();
STRF_HD strf::destination<char>*& test_messages_destination_ptr();

#else

inline STRF_HD strf::destination<char>*& test_messages_destination_ptr()
{
    static strf::destination<char>* ptr = nullptr;
    return ptr;
}

inline STRF_HD strf::destination<char>& test_messages_destination()
{
    auto * ptr = test_messages_destination_ptr();
    if (ptr == nullptr) {
        static strf::discarder<char> discarder;
        return discarder;
    }
    return *ptr;
}

#endif

class test_messages_destination_guard
{
public:
    test_messages_destination_guard(strf::destination<char>& dst)
    {
        previous_dst_ = test_messages_destination_ptr();
        test_messages_destination_ptr() = &dst;
    }

    test_messages_destination_guard(const test_messages_destination_guard&) = delete;

    ~test_messages_destination_guard()
    {
        test_messages_destination_ptr() = previous_dst_;
    }

private:
    strf::destination<char>* previous_dst_ = nullptr;
};

class test_scope
{
public:

    STRF_HD test_scope(const test_scope&) = delete;

    STRF_HD  test_scope()
        : parent_(current_test_scope_())
        , id_(generate_new_id_())
    {
        if (parent_) {
            parent_->child_ = this;
        } else {
            first_test_scope_() = this;
        }
        current_test_scope_() = this;
        description_[0] = '\0';
    }

    STRF_HD ~test_scope()
    {
        if (current_test_scope_() == this) {
            current_test_scope_() = parent_;
        }
        if (parent_) {
            parent_->child_ = child_;
        } else {
            first_test_scope_() = child_;
        }
        if (child_) {
            child_ -> parent_ = parent_;
        }
    }

    using id_type = unsigned long;

    STRF_HD id_type id() const
    {
        return id_;
    }

    STRF_HD static void print_stack(strf::destination<char>& out)
    {
        auto current_id = (current_test_scope_() == nullptr ? 0 : current_test_scope_()->id());
        if (current_id != last_printed_scope_id_()) {
            last_printed_scope_id_() = current_id;
            test_scope* first = first_test_scope_();
            if (first == nullptr) {
                strf::write(out, "\n( AT ROOT TEST SCOPE )\n");
            } else {
                strf::write(out, "\n( AT TEST SCOPE: ");
                strf::write(out, first->description_);
                for(auto it = first->child_; it != nullptr; it = it->child_) {
                    strf::write(out, " / ");
                    strf::write(out, it->description_);
                }
                strf::write(out, " )\n");
            }
        }
    }

    STRF_HD auto description_writer() -> decltype(strf::to((char*)0, (std::size_t)0))
    {
        return strf::to(description_, sizeof(description_));
    }

private:

    struct root_tag {};

    STRF_HD test_scope(root_tag)
    {
        description_[0] = '\0';
    }

    STRF_HD static id_type generate_new_id_()
    {
        static id_type x = 0;
        return ++x;
    }
    STRF_HD static id_type& last_printed_scope_id_()
    {
        static id_type x = strf::detail::int_max<id_type>();
        return x;
    }
    STRF_HD static test_scope*& current_test_scope_()
    {
        static test_scope* ptr = nullptr;
        return ptr;
    }
    STRF_HD static test_scope*& first_test_scope_()
    {
        static test_scope* ptr = nullptr;
        return ptr;
    }

    test_scope* parent_ = nullptr;
    test_scope* child_ = nullptr;
    id_type id_ = 0;
    char description_[200];
};

inline STRF_HD void print_test_message_header(const char* filename, int line)
{
    test_scope::print_stack(test_messages_destination());
    to(test_utils::test_messages_destination()) (filename, ':', line, ": ");
}

inline STRF_HD void print_test_message_end(const char* funcname)
{
    to(test_utils::test_messages_destination()) ("\n    In function '", funcname, "'\n");
}

template <typename ... Args>
void STRF_HD test_message
    ( const char* filename
    , int line
    , const char* funcname
    , const Args& ... args )
{
    test_utils::print_test_message_header(filename, line);
    to(test_utils::test_messages_destination()).with(strf::lettercase::mixed) (args...);
    test_utils::print_test_message_end(funcname);
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

class test_failure_notifier: public strf::destination<char> {
public:
    STRF_HD test_failure_notifier
        ( const char* funcname
        , const char* srcfile
        , int srcline
        , strf::destination<char>& dest = test_utils::test_messages_destination() )
        : strf::destination<char>{buff_, buffsize_}
        , dest_(dest)
        , funcname_(funcname)
        , srcfile_(srcfile)
        , srcline_(srcline)
    {
    }

    STRF_HD ~test_failure_notifier()
    {
#if __cpp_exceptions  && ! defined(__CUDACC__)
        try {
            finish();
        } catch(...) {
        }
#endif
    }

    STRF_HD void recycle() override {
        do_recycle();
    }

    STRF_HD void finish() {
        if (!finished_) {
            do_recycle();
            if (has_error_) {
                test_utils::print_test_message_end(funcname_);
            }
            finished_ = true;
        }
    }

private:
    STRF_HD void do_recycle() {
        if (this->buffer_ptr() != buff_) {
            ensure_notification_init_();
            dest_.write(buff_, this->buffer_ptr() - buff_);
            this->set_buffer_ptr(buff_);
        }
    }

    STRF_HD void ensure_notification_init_() {
        if (!has_error_) {
            has_error_ = true;
            ++ test_utils::test_err_count();
            test_utils::print_test_message_header(srcfile_, srcline_);
        }
    }

    bool has_error_ = false;
    bool finished_ = false;

    strf::destination<char>& dest_;
    const char* funcname_;
    const char* srcfile_;
    int srcline_;
    constexpr static std::size_t buffsize_ =
        strf::destination<char>::min_space_after_recycle;
    char buff_[buffsize_];
};


template <typename CharOut>
class input_tester : public strf::destination<CharOut>{
public:

    struct input{
        STRF_HD input
            ( strf::detail::simple_string_view<CharOut> expected_
            , const char* src_filename_
            , int src_line_
            , const char* function_
            , double reserve_factor_
            , std::size_t size_ = 0 )
            : expected(expected_)
            , src_filename(src_filename_)
            , src_line(src_line_)
            , function(function_)
            , reserve_factor(reserve_factor_)
            , size(size_)
        {
        }

        strf::detail::simple_string_view<CharOut> expected;
        const char* src_filename;
        int src_line;
        const char* function;
        double reserve_factor;
        std::size_t size = 0;
    };

    STRF_HD input_tester(input i)
        : input_tester{ i.expected, i.src_filename, i.src_line, i.function
                      , i.reserve_factor, i.size }
    {
    }

    STRF_HD input_tester
        ( strf::detail::simple_string_view<CharOut> expected
        , const char* src_filename
        , int src_line
        , const char* function
        , double reserve_factor
        , std::size_t size = 0 );

    STRF_HD input_tester(input_tester&& r) = delete;
    STRF_HD input_tester(const input_tester& r) = delete;

    STRF_HD ~input_tester();

    using char_type = CharOut;

    void STRF_HD recycle() override;

    void STRF_HD finish();

private:

    bool STRF_HD wrong_size_(std::size_t result_size) const;
    bool STRF_HD wrong_content_( strf::detail::simple_string_view<CharOut> result ) const;

    test_failure_notifier notifier_;
    strf::detail::simple_string_view<CharOut> expected_;
    std::size_t reserved_size_;
    double reserve_factor_;

    CharOut* pointer_before_overflow_ = nullptr;
    enum {buffer_size_ = 280};
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
    : strf::destination<CharOut>{buffer_, size}
    , notifier_{function, src_filename, src_line}
    , expected_(expected)
    , reserved_size_(size)
    , reserve_factor_(reserve_factor)
{
    if (size > buffer_size_) {
        strf::to(notifier_)
            ( "Warning: reserved more characters (", size
            , ") then the tester buffer size (", buffer_size_, ")." );
        this->set_buffer_end(buffer_ + buffer_size_);
    }
}

template <typename CharOut>
STRF_HD input_tester<CharOut>::~input_tester()
{
}

template <typename CharOut>
void STRF_HD input_tester<CharOut>::recycle()
{
    strf::to(notifier_)
        (" destination::recycle() called "
         "( it means the calculated size too small ).\n");

    if ( this->buffer_ptr() + strf::min_destination_buffer_size
       > buffer_ + buffer_size_ )
    {
        pointer_before_overflow_ = this->buffer_ptr();
        this->set_buffer_ptr(strf::garbage_buff<CharOut>());
        this->set_buffer_end(strf::garbage_buff_end<CharOut>());
    } else {
        this->set_buffer_end(buffer_ + buffer_size_);
    }
}

template <typename CharOut>
void STRF_HD input_tester<CharOut>::finish()
{
    auto pointer = pointer_before_overflow_ ? pointer_before_overflow_ : this->buffer_ptr();
    strf::detail::simple_string_view<CharOut> result{buffer_, pointer};
    bool failed_content = wrong_content_(result);
    bool failed_size = wrong_size_(result.size());
    if (failed_size || failed_content){
        if (failed_content) {
            strf::to(notifier_) ("\n  expected: \"", strf::transcode(expected_));
            strf::to(notifier_) ("\"\n  obtained: \"", strf::transcode(result), '\"');
        }
        if (failed_size) {
            strf::to(notifier_) ("\n  reserved size  : ", reserved_size_);
            strf::to(notifier_) ("\n  necessary size : ", result.size());
        }
    }
    notifier_.finish();
}

template <typename CharOut>
bool STRF_HD input_tester<CharOut>::wrong_content_
    ( strf::detail::simple_string_view<CharOut> result ) const
{
   return ( result.size() != expected_.size()
         || ! strf::detail::str_equal<CharOut>
            ( expected_.begin(), result.begin(), expected_.size() ));
}

template <typename CharOut>
bool STRF_HD input_tester<CharOut>::wrong_size_(std::size_t result_size) const
{
    return ( reserved_size_ < result_size
          || ( static_cast<double>(reserved_size_) * reserve_factor_
             > static_cast<double>(result_size) ) );
}

template <typename CharT>
class input_tester_creator
{
public:

    using char_type = CharT;
    using sized_destination_type = test_utils::input_tester<CharT>;

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

    typename test_utils::input_tester<CharT>::input STRF_HD create(std::size_t size) const
    {
        return { expected_, filename_, line_, function_, reserve_factor_, size };
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
    -> strf::printing_syntax
        < test_utils::input_tester_creator<CharT>
        , strf::reserve_calc >
{
    return strf::make_printing_syntax
        ( test_utils::input_tester_creator<CharT>
            {expected, filename, line, function, reserve_factor}
        , strf::reserve_calc{} );
}

template<typename CharT>
auto STRF_HD make_tester
   ( strf::detail::simple_string_view<CharT> expected
   , const char* filename
   , int line
   , const char* function
   , double reserve_factor = 1.0 )
    -> strf::printing_syntax
        < test_utils::input_tester_creator<CharT>
        , strf::reserve_calc >
{
    return strf::make_printing_syntax
        ( test_utils::input_tester_creator<CharT>
            {expected, filename, line, function, reserve_factor}
        , strf::reserve_calc{} );
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


namespace detail {

template <unsigned... X> struct reduce;

template <> struct reduce<>{
    static constexpr unsigned value = 0;
};

template <unsigned X, unsigned... Others>
struct reduce<X, Others...>{
    static constexpr unsigned value = X + reduce<Others...>::value;
};

} // namespace detail

template <typename CharT>
struct recycle_call_tester_input {
    strf::detail::simple_string_view<CharT> expected;
    const char* function;
    const char* src_filename;
    int src_line;
    std::size_t initial_space;
};

template <typename CharT>
class recycle_call_tester: public strf::destination<CharT> {
public:

    STRF_HD recycle_call_tester(recycle_call_tester_input<CharT> input)
        : strf::destination<CharT>{buffer_, input.initial_space}
        , expected_(input.expected)
        , notifier_{input.function, input.src_filename, input.src_line}
        , dest_end_{buffer_}
    {
        if (input.initial_space + strf::min_destination_buffer_ssize > buffer_size_) {
            strf::to(notifier_) ("\nUnsupported test case: Initial space too big");
            this->set_buffer_end(buffer_);
        }
    }

    STRF_HD void recycle() override;

    STRF_HD void finish();

private:

    bool recycle_not_called_yet = true;
    strf::detail::simple_string_view<CharT> expected_;
    test_failure_notifier notifier_;
    CharT* dest_end_;

    enum {buffer_size_ = 320};
    CharT buffer_[buffer_size_];
};

template <typename CharT>
STRF_HD void recycle_call_tester<CharT>::recycle()
{
    if (recycle_not_called_yet) {
        if (this->buffer_ptr() > this->buffer_end()) {
            strf::to(notifier_) ( "\nContent written after buffer_end()." );
        }
        this->set_buffer_end(buffer_ + buffer_size_);
        recycle_not_called_yet = false;
    } else {
        strf::to(notifier_) ( "\nrecycle() called more than once here.");
        this->set_good(false);
        this->set_buffer_ptr(strf::garbage_buff<CharT>());
        this->set_buffer_end(strf::garbage_buff_end<CharT>());
    }
}

template <typename CharT>
STRF_HD void recycle_call_tester<CharT>::finish()
{
    if (recycle_not_called_yet) {
        strf::to(notifier_) ( "\nrecycle() was not called.");
    }
    if (this->good()) {
        dest_end_ = this->buffer_ptr();
        this->set_good(false);
        this->set_buffer_ptr(strf::garbage_buff<CharT>());
        this->set_buffer_end(strf::garbage_buff_end<CharT>());
    }
    strf::detail::simple_string_view<CharT> result{buffer_, dest_end_};
    bool as_expected =
        ( result.size() == expected_.size()
       && strf::detail::str_equal<CharT>(expected_.begin(), buffer_, expected_.size()) );

    if ( ! as_expected ) {
        strf::to(notifier_)
            ( "\n  expected: \"", strf::transcode(expected_)
            , "\"\n  obtained: \"", strf::transcode(result), '\"');
    }
    notifier_.finish();
}

template <typename CharT>
class recycle_call_tester_creator {
public:
    using destination_type = recycle_call_tester<CharT>;
    using char_type = CharT;

    STRF_HD recycle_call_tester_creator
        ( strf::detail::simple_string_view<CharT> expected
        , const char* function
        , const char* src_filename
        , int src_line
        , std::size_t initial_space )
        : input_{expected, function, src_filename, src_line, initial_space}
    {}

    STRF_HD recycle_call_tester_input<CharT> create() const { return input_; }

public:
    recycle_call_tester_input<CharT> input_;
};

template <typename CharT>
STRF_HD auto test_recycle_call
    ( const CharT* expected
    , const char* function
    , const char* src_filename
    , int src_line
    , std::size_t initial_space )
    -> strf::printing_syntax<test_utils::recycle_call_tester_creator<CharT>>
{
    return strf::make_printing_syntax
        ( test_utils::recycle_call_tester_creator<CharT>
            {expected, function, src_filename, src_line, initial_space} );
}

template <typename CharT>
class failed_recycle_call_tester: public strf::destination<CharT> {
public:

    STRF_HD failed_recycle_call_tester(recycle_call_tester_input<CharT> input)
        : strf::destination<CharT>{buffer_, input.initial_space}
        , expected_(input.expected)
        , function_(input.function)
        , src_filename_(input.src_filename)
        , src_line_(input.src_line)
        , dest_end_{buffer_}
    {
        if (input.initial_space + strf::min_destination_buffer_ssize > buffer_size_) {
            emit_error_message_("\nUnsupported test case: Initial space too big");
            this->set_buffer_end(buffer_);
        }
    }

    STRF_HD void recycle() override;

    STRF_HD void finish();

private:

    STRF_HD void before_emitting_error_message_()
    {
        if (!error_message_emitted_) {
            ++ test_err_count();
            print_test_message_header(src_filename_, src_line_);
            error_message_emitted_ = true;
        }
    }

    template <typename... Args>
    STRF_HD void emit_error_message_(const Args&... args)
    {
        before_emitting_error_message_();
        strf::to(test_utils::test_messages_destination()) (args...);
    }

    bool recycle_not_called_yet = true;
    bool error_message_emitted_ = false;
    strf::detail::simple_string_view<CharT> expected_;
    const char* function_;
    const char* src_filename_;
    int src_line_;
    CharT* dest_end_;

    enum {buffer_size_ = 320};
    CharT buffer_[buffer_size_];
};

template <typename CharT>
STRF_HD void failed_recycle_call_tester<CharT>::recycle()
{
    if (recycle_not_called_yet) {
        if (this->buffer_ptr() > this->buffer_end()) {
            emit_error_message_( "\nContent written after buffer_end()." );
        }
        dest_end_ = this->buffer_ptr();
        recycle_not_called_yet = false;
    } else {
        emit_error_message_( "\nrecycle() called more than once here.");
    }
    this->set_good(false);
    this->set_buffer_ptr(strf::garbage_buff<CharT>());
    this->set_buffer_end(strf::garbage_buff_end<CharT>());
}

template <typename CharT>
STRF_HD void failed_recycle_call_tester<CharT>::finish()
{
    if (recycle_not_called_yet) {
        dest_end_ = this->buffer_ptr();
    }
    this->set_good(false);
    this->set_buffer_ptr(strf::garbage_buff<CharT>());
    this->set_buffer_end(strf::garbage_buff_end<CharT>());

    strf::detail::simple_string_view<CharT> result{buffer_, dest_end_};
    bool as_expected =
        ( result.size() == expected_.size()
       && strf::detail::str_equal<CharT>(expected_.begin(), buffer_, expected_.size()) );

    if ( ! as_expected ) {
        emit_error_message_
            ( "\n  expected: \"", strf::transcode(expected_)
            , "\"\n  obtained: \"", strf::transcode(result), '\"');
    }

    if (error_message_emitted_) {
        print_test_message_end(function_);
        error_message_emitted_ = false;
    }
}

template <typename CharT>
class failed_recycle_call_tester_creator {
public:
    using destination_type = failed_recycle_call_tester<CharT>;
    using char_type = CharT;

    STRF_HD failed_recycle_call_tester_creator
        ( strf::detail::simple_string_view<CharT> expected
        , const char* function
        , const char* src_filename
        , int src_line
        , std::size_t initial_space )
        : input_{expected, function, src_filename, src_line, initial_space}
    {}

    STRF_HD recycle_call_tester_input<CharT> create() const { return input_; }

public:
    recycle_call_tester_input<CharT> input_;
};

template <typename CharT>
STRF_HD auto test_failed_recycle_call
    ( const CharT* expected
    , const char* function
    , const char* src_filename
    , int src_line
    , std::size_t initial_space )
    -> strf::printing_syntax<test_utils::failed_recycle_call_tester_creator<CharT>>
{
    return strf::make_printing_syntax
        ( test_utils::failed_recycle_call_tester_creator<CharT>
          {expected, function, src_filename, src_line, initial_space} );
}

} // namespace test_utils

#define TEST_CALLING_RECYCLE_AT(SPACE, EXPECTED)                       \
    test_utils::test_recycle_call                                      \
        (EXPECTED, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, SPACE)

#define TEST_TRUNCATING_AT(SPACE, EXPECTED)                             \
    test_utils::test_failed_recycle_call                                \
        (EXPECTED, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, SPACE)

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
            , "TEST_TRUE (" #expr ") failed. " );                            \

#define TEST_FALSE(expr)                                                \
    if ((expr))                                                         \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , "TEST_FALSE (" #expr ") failed. " );                            \

#define TEST_EQ(a, b)                                                   \
    if (!test_utils::equal((a), (b)))                                   \
        test_utils::test_failure                                        \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                \
            , "TEST_EQ (", (a), ", ", (b), ") failed. " );

#define TEST_CSTR_EQ(s1, s2)                                            \
    for ( const std::size_t len1 = strf::detail::str_length(s1); 1;){   \
        if ( len1 != strf::detail::str_length(s2)                       \
          || ! strf::detail::str_equal(s1, s2, len1))                   \
            test_utils::test_failure                                    \
                ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION            \
                , "TEST_CSTR_EQ(s1, s2) failed. Where:\n    s1 is \"", (s1)  \
                , "\"\n    s2 is \"", (s2), '\"' );                     \
        break;                                                          \
    }                                                                   \

#define TEST_STRVIEW_EQ(s1, s2, len)                                       \
    if (! strf::detail::str_equal(s1, s2, len))                            \
        test_utils::test_failure                                           \
            ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                   \
            , "TEST_STRVIEW_EQ(s1, s2, len) failed. Where:\n    s1 is \""  \
            , strf::detail::make_simple_string_view((s1), len)             \
            , "\"\n    s2 is \""                                           \
            , strf::detail::make_simple_string_view((s2), len), '\"' );

#define TEST_THROWS( EXPR, EXCEP )                                      \
  { bool caught = false;                                                \
    try { EXPR; }                                                       \
    catch(EXCEP const&) { caught = true; }                              \
    if (!caught)                                                        \
          test_utils::test_failure                                      \
              ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION              \
              , "exception " #EXCEP " not thrown as expected" );        \
  }


namespace test_utils {

template <typename T, std::size_t N>
struct simple_array {
    T elements[N];
};

template <typename T, typename... Args>
STRF_HD simple_array<T, sizeof... (Args)> make_simple_array(const Args&... args)
{
    return simple_array<T, sizeof... (Args)>{{ static_cast<T>(args)... }};
}

template <typename T>
class span {
public:

    using const_iterator = T*;
    using iterator = T*;

    span() = default;

    template <typename U, std::size_t N>
    STRF_HD span(simple_array<U, N>& arr)
        : begin_(&arr.elements[0])
        , size_(N)
    {
    }
    STRF_HD span(T* ptr, std::size_t s)
        : begin_(ptr)
        , size_(s)
    {
    }
    STRF_HD span(T* b, T* e)
        : begin_(b)
        , size_(strf::detail::safe_cast_size_t(e - b))
    {
    }

    STRF_HD T* begin() const { return begin_; }
    STRF_HD T* end()   const { return begin_ + size_; }

    STRF_HD std::size_t size() const { return size_; }
    STRF_HD std::ptrdiff_t ssize() const { return static_cast<std::ptrdiff_t>(size_); }
    STRF_HD T& operator[](std::size_t i) const { return begin_[i]; }

private:
    T* begin_ = nullptr;
    std::size_t size_ = 0;
};

template <typename T, typename U>
STRF_HD bool operator==(const span<T>& l, const span<U>& r) noexcept
{
    if (l.size() != r.size())
        return false;

    for (std::size_t i = 0; i < l.size(); ++i) {
        if (l.begin()[i] != r.begin()[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, typename U>
STRF_HD bool operator!=(const span<T>& l, const span<U>& r) noexcept
{
    return ! (l == r);
}

namespace detail {

struct testfunc_node {
    typedef void (*func_ptr_t)(void);

    func_ptr_t testfunc = nullptr;
    testfunc_node* next = nullptr;
};

struct testfunc_list_t {
    testfunc_node* first = nullptr;
    testfunc_node* last = nullptr;
};

inline testfunc_list_t& testfunc_list() noexcept {
    static testfunc_list_t lst;
    return lst;
}

inline void add_testfunc(testfunc_node* node) noexcept {
    node->next = nullptr;
    auto& list = testfunc_list();
    if (list.last) {
        list.last->next = node;
        list.last = node;
    } else {
        list.first = node;
        list.last = node;
    }
}

class testfunc_node_registration {
public:
    testfunc_node_registration(testfunc_node::func_ptr_t func) noexcept {
        node_.testfunc = func;
        add_testfunc(&node_);
    }

private:
    testfunc_node node_;
};


} // namespace detail

inline void run_all_tests() {
    for ( auto* node = detail::testfunc_list().first
        ; node != nullptr
        ; node = node->next)
    {
        if (node->testfunc)
            node->testfunc();
    }
}

} // namespace test_utils


#if defined(__CUDACC__)
#  define REGISTER_STRF_TEST(FUNC) STRF_TEST_FUNC void run_all_tests() { FUNC(); }
#else
#  define REGISTER_STRF_TEST(FUNC)                                    \
       namespace {                                                    \
           const test_utils::detail::testfunc_node_registration       \
               TEST_STR_CONCAT(testfunc_reg_, __LINE__) {FUNC};       \
       }

#endif // defined(__CUDACC__)

#endif // defined(STRF_TEST_TEST_UTILS_HPP_INCLUDED)
