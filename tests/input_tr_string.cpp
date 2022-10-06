//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

namespace {

class err_handler {

public:
    using category = strf::tr_error_notifier_c;

    STRF_HD explicit err_handler(strf::destination<char>& log)
        : log_(log)
    {
    }

    template <typename Charset>
    void STRF_HD handle
        ( const typename Charset::code_unit* str
        , std::size_t str_len
        , Charset charset
        , std::size_t err_pos ) noexcept
    {
        strf::detail::simple_string_view<typename Charset::code_unit> s(str, str_len);
        strf::to(log_) ("\n[", strf::dec(err_pos) > 2, "] ", strf::transcode(s, charset));
    }

private:
    strf::destination<char>& log_;
};

STRF_TEST_FUNC void test_printing()
{
    // basic test
    TEST("aaa__..bbb__ 0xa")
        (strf::tr( "{}__{}__{}"
                 , "aaa"
                 , strf::right("bbb", 5, '.')
                 , *strf::hex(10)>4) );

    TEST(u8"_0__1__2")   (strf::tr(u8"_{}__{}__{}",    0, 1, 2));
    TEST(u8"{0_{_{1_{2") (strf::tr(u8"{{{}_{{_{{{}_{{{}",  0, 1, 2));
    TEST(u8"0__1__2")    (strf::tr(u8"{}__{}__{}",     0, 1, 2));
    TEST(u8"0__1__2")    (strf::tr(u8"{}__{}__{",      0, 1, 2));
    TEST(u8"0__1__2")    (strf::tr(u8"{}__{}__{aaa}",  0, 1, 2));
    TEST(u8"0__1__2")    (strf::tr(u8"{}__{}__{aaa",   0, 1, 2));
    TEST(u8"0__1__2_")   (strf::tr(u8"{}__{}__{}_",    0, 1, 2));
    TEST(u8"0__1__2_")   (strf::tr(u8"{}__{}__{aaa}_", 0, 1, 2));

    TEST(u8"0__1__\uFFFD")    (strf::tr(u8"{}__{}__{}",     0, 1));
    TEST(u8"0__1__\uFFFD")    (strf::tr(u8"{}__{}__{",      0, 1));
    TEST(u8"0__1__\uFFFD")    (strf::tr(u8"{}__{}__{aaa}",  0, 1));
    TEST(u8"0__1__\uFFFD")    (strf::tr(u8"{}__{}__{aaa",   0, 1));
    TEST(u8"0__1__\uFFFD_")   (strf::tr(u8"{}__{}__{}_",    0, 1));
    TEST(u8"0__1__\uFFFD_")   (strf::tr(u8"{}__{}__{aaa}_", 0, 1));

    TEST(u8"0__3__1")  (strf::tr(u8"{}__{3}__{}",       0, 1, 2, 3));
    TEST(u8"0__3__1")  (strf::tr(u8"{}__{3}__{",        0, 1, 2, 3));
    TEST(u8"0__3__1")  (strf::tr(u8"{}__{3aa}__{aaa}",  0, 1, 2, 3));
    TEST(u8"0__3__1")  (strf::tr(u8"{}__{3}__{aaa",     0, 1, 2, 3));
    TEST(u8"0__3__1_") (strf::tr(u8"{}__{3}__{}_",      0, 1, 2, 3));
    TEST(u8"0__3__1_") (strf::tr(u8"{}__{3aa}__{aaa}_", 0, 1, 2, 3));

    TEST(u8"0__1__3")  (strf::tr(u8"{}__{1}__{3}",     0, 1, 2, 3));
    TEST(u8"0__1__3")  (strf::tr(u8"{}__{1a}__{3",     0, 1, 2, 3));
    TEST(u8"0__1__3")  (strf::tr(u8"{}__{1}__{3aaa}",  0, 1, 2, 3));
    TEST(u8"0__1__3")  (strf::tr(u8"{}__{1}__{3aaa",   0, 1, 2, 3));
    TEST(u8"0__1__3_") (strf::tr(u8"{}__{1}__{3}_",    0, 1, 2, 3));
    TEST(u8"0__1__3_") (strf::tr(u8"{}__{1}__{3aaa}_", 0, 1, 2, 3));

    TEST(u8"_0__10__")   (strf::tr(u8"_{}__{10}__"  , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10));
    TEST(u8"_0__10__")   (strf::tr(u8"_{}__{10aa}__", 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10));
    TEST(u8"_0__10__")   (strf::tr(u8"_{}__{10}__"  , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10));
    TEST(u8"_0__10")     (strf::tr(u8"_{}__{10"     , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10));
    TEST(u8"_0__\uFFFD") (strf::tr(u8"_{}__{11"     , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10));
    TEST(u8"_0__\uFFFD") (strf::tr(u8"_{}__{100"    , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10));

    // comment at the end
    TEST(u8"0__")  (strf::tr(u8"{}__{-}", 0));
    TEST(u8"0__")  (strf::tr(u8"{}__{-", 0));
    TEST(u8"0__")  (strf::tr(u8"{}__{-aaa", 0));
    TEST(u8"0__")  (strf::tr(u8"{}__{-aaa}", 0));
    // comment in the middle
    TEST(u8"0__1")   (strf::tr(u8"{}__{-}{}", 0, 1));
    TEST(u8"0__1")   (strf::tr(u8"{}__{-aaa}{}", 0, 1));
    TEST(u8"0__~1")  (strf::tr(u8"{}__{-}~{}", 0, 1));
    TEST(u8"0__~1")  (strf::tr(u8"{}__{-aaa}~{}", 0, 1));
    // comment at the begining
    TEST(u8"__0")  (strf::tr(u8"{-}__{}", 0));
    TEST(u8"-__0") (strf::tr(u8"-{-}__{}", 0));
    TEST(u8"{__0") (strf::tr(u8"{{{-}__{}", 0));


    // escaped '{'
    TEST(u8"{}_{0}__{")  (strf::tr(u8"{{}_{{{}}__{{", 0));
    TEST( u"{}_{0}__{")  (strf::tr( u"{{}_{{{}}__{{", 0));

    // invalid arguments
    TEST(u8"0 3 \uFFFD \uFFFD 1") (strf::tr(u8"{} {3} {4} {5} {1}", 0, 1, 2, 3));
    TEST(u8"0 3 \uFFFD \uFFFD")   (strf::tr(u8"{} {3} {4aa} {5}", 0, 1, 2, 3));
    TEST(u8"0 3 \uFFFD \uFFFD")   (strf::tr(u8"{} {3} {11} {5", 0, 1, 2, 3));
    TEST(u8"0 3 \uFFFD \uFFFD")   (strf::tr(u8"{} {3} {111} {5 ", 0, 1, 2, 3));
    TEST(u8"0 3 \uFFFD")          (strf::tr(u8"{} {3} {4}", 0, 1, 2, 3));
    TEST(u8"0 3 \uFFFD")          (strf::tr(u8"{} {3} {4", 0, 1, 2, 3));
    TEST(u8"0 1 2 \uFFFD_")       (strf::tr(u8"{} {} {} {}_", 0, 1, 2));
    TEST(u8"0 1 2 \uFFFD")        (strf::tr(u8"{} {} {} {}", 0, 1, 2));
    TEST(u8"0 1 2 \uFFFD")        (strf::tr(u8"{} {} {} {", 0, 1, 2));
    TEST(u8"0 1 2 \uFFFD_")       (strf::tr(u8"{} {} {} {aa}_", 0, 1, 2));
    TEST(u8"0 1 2 \uFFFD")        (strf::tr(u8"{} {} {} {aa}", 0, 1, 2));
    TEST(u8"0 1 2 \uFFFD")        (strf::tr(u8"{} {} {} {aa", 0, 1, 2));
    TEST(u8"_\uFFFD__")           (strf::tr(u8"_{}__"));
    TEST(u8"\uFFFD__")            (strf::tr(u8"{}__"));
    TEST(u8"__\uFFFD")            (strf::tr(u8"__{}"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{aa}"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{}"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{0}"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{1}"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{0 aa}"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{1 aa}"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{"));
    TEST(u8"\uFFFD")              (strf::tr(u8"{aaa"));

    // invalid arguments - now in UTF-16
    TEST(u"0 3 \uFFFD \uFFFD 1") (strf::tr(u"{} {3} {4} {5} {1}", 0, 1, 2, 3));
    TEST(u"0 3 \uFFFD \uFFFD")   (strf::tr(u"{} {3} {4aa} {5}", 0, 1, 2, 3));
    TEST(u"0 3 \uFFFD \uFFFD")   (strf::tr(u"{} {3} {11} {5", 0, 1, 2, 3));
    TEST(u"0 3 \uFFFD \uFFFD")   (strf::tr(u"{} {3} {111} {5 ", 0, 1, 2, 3));
    TEST(u"0 3 \uFFFD")          (strf::tr(u"{} {3} {4}", 0, 1, 2, 3));
    TEST(u"0 3 \uFFFD")          (strf::tr(u"{} {3} {4", 0, 1, 2, 3));
    TEST(u"0 1 2 \uFFFD_")       (strf::tr(u"{} {} {} {}_", 0, 1, 2));
    TEST(u"0 1 2 \uFFFD")        (strf::tr(u"{} {} {} {}", 0, 1, 2));
    TEST(u"0 1 2 \uFFFD")        (strf::tr(u"{} {} {} {", 0, 1, 2));
    TEST(u"0 1 2 \uFFFD_")       (strf::tr(u"{} {} {} {aa}_", 0, 1, 2));
    TEST(u"0 1 2 \uFFFD")        (strf::tr(u"{} {} {} {aa}", 0, 1, 2));
    TEST(u"0 1 2 \uFFFD")        (strf::tr(u"{} {} {} {aa", 0, 1, 2));
    TEST(u"_\uFFFD__")           (strf::tr(u"_{}__"));
    TEST(u"\uFFFD__")            (strf::tr(u"{}__"));
    TEST(u"__\uFFFD")            (strf::tr(u"__{}"));
    TEST(u"\uFFFD")              (strf::tr(u"{aa}"));
    TEST(u"\uFFFD")              (strf::tr(u"{}"));
    TEST(u"\uFFFD")              (strf::tr(u"{0}"));
    TEST(u"\uFFFD")              (strf::tr(u"{1}"));
    TEST(u"\uFFFD")              (strf::tr(u"{0 aa}"));
    TEST(u"\uFFFD")              (strf::tr(u"{1 aa}"));
    TEST(u"\uFFFD")              (strf::tr(u"{"));
    TEST(u"\uFFFD")              (strf::tr(u"{aaa"));

    // Customizing error handling
    {
        char buff[200];
        strf::cstr_destination log(buff);
        TEST(u8"0__\uFFFD--1==2..3::\uFFFD~~")
            .with(err_handler{log})
            (strf::tr(u8"{ }__{10}--{}=={}..{}::{}~~", 0, 1, 2, 3));

        log.finish();
        TEST_CSTR_EQ( buff, "\n[ 5] { }__{10}--{}=={}..{}::{}~~"
                            "\n[23] { }__{10}--{}=={}..{}::{}~~" );
    }
    {
        char buff[200];
        strf::cstr_destination log(buff);
        TEST(u8"0__\uFFFD--1==2..3::\uFFFD~~")
            .with(err_handler{log})
            (strf::tr(u8"{ }__{10}--{}=={}..{}::{blah}~~", 0, 1, 2, 3));

        log.finish();
        TEST_CSTR_EQ(buff, "\n[ 5] { }__{10}--{}=={}..{}::{blah}~~"
                           "\n[23] { }__{10}--{}=={}..{}::{blah}~~" );
    }
    {
        char buff[200];
        strf::cstr_destination log(buff);
        TEST(u"0__2--1==2..3::\uFFFD")
            .with(err_handler{log})
            (strf::tr(u"{ }__{2}--{}=={}..{}::{", 0, 1, 2, 3));

        log.finish();
        TEST_CSTR_EQ(buff, "\n[22] { }__{2}--{}=={}..{}::{");
    }
}


STRF_TEST_FUNC void test_without_preprinting()
{
    char buff[200];

    {
        auto res = strf::to(buff) (strf::tr("_{}__{}__{}", 0, 1, 2));
        TEST_FALSE(res.truncated);
        TEST_CSTR_EQ(buff, "_0__1__2");
    }

    {
        auto res = strf::to(buff) (strf::tr("blah blah"));
        TEST_FALSE(res.truncated);
        TEST_CSTR_EQ(buff, "blah blah");
    }

    {
        auto res = strf::to(buff) (strf::tr("blah {} blah"));
        TEST_FALSE(res.truncated);
        TEST_CSTR_EQ(buff, "blah \xEF\xBF\xBD blah");
    }
}

template <typename String, typename... Args>
auto first_char_of_tr_string(String str, Args&&...)
    -> strf::detail::remove_cvref_t<decltype(str[0])>
{
    return str[0];
}


#define TEST_SIZE_PRECALC(EXPECTED_SIZE, ...) \
    {                                                                                 \
        using char_t = decltype(first_char_of_tr_string(__VA_ARGS__));                \
        using pre_t = strf::preprinting                                               \
            <strf::precalc_size::yes, strf::precalc_width::no>;                       \
        pre_t pre;                                                                    \
        strf::precalculate<char_t>(pre, strf::pack(), strf::tr(__VA_ARGS__));         \
        std::size_t obtained = pre.accumulated_size();                                \
        std::size_t expected = EXPECTED_SIZE;                                         \
        if (obtained != expected) {                                                   \
            test_utils::test_failure                                                  \
                 ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                         \
                 , "    Calculated size is : ", obtained                              \
                 , " (expected ", expected, ")\n" );                                  \
        }                                                                             \
    }

STRF_TEST_FUNC void test_size_precalculation()
{
    TEST_SIZE_PRECALC(5, "ooOoo");
    TEST_SIZE_PRECALC(5,"_{}_{}_", 0, 1);
    TEST_SIZE_PRECALC(10, "_{1zz}{0zz}_", 1000, 1001);

    // when tr-string ends before the closing '}'
    TEST_SIZE_PRECALC(7, ".:.:.{", 12);
    TEST_SIZE_PRECALC(7, ".:.:.{.....", 12);

    // handle a simple "{}"
    TEST_SIZE_PRECALC(10, "____{}__", 1234);

    // handle indexed parameter
    TEST_SIZE_PRECALC(10, "____{1 blah blah }__", 0, 1234);
    TEST_SIZE_PRECALC( 8, "____{1 blah blah "   , 0, 1234);
    TEST_SIZE_PRECALC( 9, "____{10 }__", 1234);
    TEST_SIZE_PRECALC( 7, "____{10 ",    1234);
    TEST_SIZE_PRECALC( 7, "____{10 ",    1234);

    // handle '{{'
    TEST_SIZE_PRECALC(7,"...{{...", 0, 1001);

    // handle commented paramenter
    TEST_SIZE_PRECALC(9, "...{ blah blah }...", 123);
    TEST_SIZE_PRECALC(6, "...{ blah blah"     , 123);
    TEST_SIZE_PRECALC(6, "...{ blah blah }"   , 123);

    // handle comment
    TEST_SIZE_PRECALC(6, "...{- a comment here }...", 123);
    TEST_SIZE_PRECALC(3, "...{- a comment here }"   , 123);
    TEST_SIZE_PRECALC(3, "...{- a comment here"     , 123);
}

#define TEST_WIDTH_PRECALC(INITIAL_WIDTH, EXPECTED_REMAINING_WIDTH, ...)              \
    {                                                                                 \
        using char_t = decltype(first_char_of_tr_string(__VA_ARGS__));                \
        using pre_t = strf::preprinting                                               \
            <strf::precalc_size::no, strf::precalc_width::yes>;                       \
        pre_t pre(strf::width_t(INITIAL_WIDTH));                                      \
        strf::precalculate<char_t>(pre, strf::pack(), strf::tr(__VA_ARGS__));         \
        auto obtained = pre.remaining_width();                                        \
        auto expected = strf::width_t(EXPECTED_REMAINING_WIDTH);                      \
        if (obtained != expected) {                                                   \
            test_utils::test_failure                                                  \
                 ( __FILE__, __LINE__, BOOST_CURRENT_FUNCTION                         \
                 , "    Remaining width is : ", obtained.round()                      \
                 , " (expected ", expected.round(), ")\n" );                          \
        }                                                                             \
    }

STRF_TEST_FUNC void test_width_precalculation()
{
    using namespace strf::width_literal;

    // some basic general tests
    TEST_WIDTH_PRECALC(0, 0, "ooOoo");
    TEST_WIDTH_PRECALC(4, 0, "ooOoo");
    TEST_WIDTH_PRECALC(5, 0, "ooOoo");
    TEST_WIDTH_PRECALC(5.015625_w, 0.015625_w, "ooOoo");
    TEST_WIDTH_PRECALC(6, 1, "ooOoo");

    TEST_WIDTH_PRECALC(6, 1, "_{}_{}_", 0, 1);
    TEST_WIDTH_PRECALC(5, 0, "_{}_{}_", 0, 1);
    TEST_WIDTH_PRECALC(4, 0, "_{}_{}_", 0, 1);
    TEST_WIDTH_PRECALC(3, 0, "_{}_{}_", 0, 1);
    TEST_WIDTH_PRECALC(2, 0, "_{}{}_", 0, 1);

    TEST_WIDTH_PRECALC(10.5_w, 0.5_w, "_{1zz}{0zz}_", 1000, 1001);
    TEST_WIDTH_PRECALC(2, 0, "_{1zz}{0zz}_", 1000, 1001);

    // when counting stops before '{'
    TEST_WIDTH_PRECALC( 5, 0, ".:.:.{}_{}_", 0, 1);
    TEST_WIDTH_PRECALC( 5, 0, ".:.:.", 0, 1);
    TEST_WIDTH_PRECALC(10, 5, ".:.:.", 0, 1);

    // when tr-string ends before the closing '}'
    TEST_WIDTH_PRECALC(7, 0, ".:.:.{", 12);
    TEST_WIDTH_PRECALC(8, 1, ".:.:.{", 12);
    TEST_WIDTH_PRECALC(8, 1, ".:.:.{.....", 12);

    // handle a simple "{}"
    TEST_WIDTH_PRECALC( 8, 0, "____{}__", 1234);
    TEST_WIDTH_PRECALC( 5, 0, "____{}__", 1234);
    TEST_WIDTH_PRECALC(11, 1, "____{}__", 1234); // when not stopping there


    // handle indexed parameter
    TEST_WIDTH_PRECALC( 8, 0, "____{1 blah blah }__", 0, 1234);
    TEST_WIDTH_PRECALC( 8, 0, "____{1 blah blah "   , 0, 1234);
    TEST_WIDTH_PRECALC( 7, 0, "____{1 blah blah "   , 0, 1234);
    TEST_WIDTH_PRECALC(10, 2, "____{1 blah blah "   , 0, 1234);
    TEST_WIDTH_PRECALC( 5, 0, "____{10 }__", 1234);
    TEST_WIDTH_PRECALC( 5, 0, "____{10 ",    1234);
    TEST_WIDTH_PRECALC( 6, 1, "____{10 ",    1234);

    // handle '{{'
    TEST_WIDTH_PRECALC( 8, 1, "...{{...", 0, 1001);
    TEST_WIDTH_PRECALC( 7, 0, "...{{...", 0, 1001);
    TEST_WIDTH_PRECALC( 6, 0, "...{{...", 0, 1001);

    // handle commented paramenter
    TEST_WIDTH_PRECALC( 6, 0, "...{ blah blah }...", 123);
    TEST_WIDTH_PRECALC( 6, 0, "...{ blah blah"     , 123);
    TEST_WIDTH_PRECALC( 7, 1, "...{ blah blah"     , 123);
    TEST_WIDTH_PRECALC( 7, 1, "...{ blah blah }"   , 123);
    TEST_WIDTH_PRECALC(10, 1, "...{ blah blah }...", 123);

    // handle comment
    TEST_WIDTH_PRECALC( 10, 4, "...{- a comment here }...", 123);
    TEST_WIDTH_PRECALC( 10, 7, "...{- a comment here }"   , 123);
    TEST_WIDTH_PRECALC( 10, 7, "...{- a comment here"     , 123);
}


#define TEST_FULL_PRECALC(EXPECTED_SIZE, INITIAL_WIDTH, EXPECTED_REMAINING_WIDTH, ...) \
    {                                                                                 \
        using char_t = decltype(first_char_of_tr_string(__VA_ARGS__));                \
        using pre_t = strf::full_preprinting;                                         \
        pre_t pre(strf::width_t(INITIAL_WIDTH));                                      \
        strf::precalculate<char_t>(pre, strf::pack(), strf::tr(__VA_ARGS__));         \
        bool failed_size = pre.accumulated_size() != EXPECTED_SIZE;                   \
        bool failed_width =                                                           \
            ( pre.remaining_width() != strf::width_t(EXPECTED_REMAINING_WIDTH) );     \
        if (failed_size || failed_width) {                                            \
            ++ test_utils::test_err_count();                                          \
            test_utils::print_test_message_header(__FILE__, __LINE__);                \
            auto& dst = test_utils::test_messages_destination();                      \
            if (failed_size) {                                                        \
                strf::to(dst) ( "    Calculated size is : ", pre.accumulated_size()   \
                              , " (expected ", EXPECTED_SIZE, ")\n");                 \
            }                                                                         \
            if (failed_width) {                                                       \
                auto obtained = pre.remaining_width().round();                        \
                auto expected = strf::width_t(EXPECTED_REMAINING_WIDTH).round();      \
                strf::to(dst) ( "    Remaining width is : ", obtained                 \
                              , " (expected ", expected, ")\n" );                     \
            }                                                                         \
            test_utils::print_test_message_end(BOOST_CURRENT_FUNCTION);               \
        }                                                                             \
    }



STRF_TEST_FUNC void test_full_precalculation() // size and width
{
    using namespace strf::width_literal;

    // some basic general tests
    TEST_FULL_PRECALC(5, 0, 0, "ooOoo");
    TEST_FULL_PRECALC(5, 4, 0, "ooOoo");
    TEST_FULL_PRECALC(5, 5, 0, "ooOoo");
    TEST_FULL_PRECALC(5, 6, 1, "ooOoo");

    TEST_FULL_PRECALC(5, 6, 1, "_{}_{}_", 0, 1);
    TEST_FULL_PRECALC(5, 5, 0, "_{}_{}_", 0, 1);
    TEST_FULL_PRECALC(5, 4, 0, "_{}_{}_", 0, 1);
    TEST_FULL_PRECALC(5, 3, 0, "_{}_{}_", 0, 1);
    TEST_FULL_PRECALC(4, 2, 0, "_{}{}_", 0, 1);

    TEST_FULL_PRECALC(10, 10.5_w, 0.5_w, "_{1zz}{0zz}_", 1000, 1001);
    TEST_FULL_PRECALC(10,      2,     0, "_{1zz}{0zz}_", 1000, 1001);

    // when tr-string ends before the closing '}'
    TEST_FULL_PRECALC(7, 7, 0, ".:.:.{", 12);
    TEST_FULL_PRECALC(7, 8, 1, ".:.:.{", 12);
    TEST_FULL_PRECALC(7, 8, 1, ".:.:.{.....", 12);

    // handle a simple "{}"
    TEST_FULL_PRECALC(10,  8, 0, "____{}__", 1234);
    TEST_FULL_PRECALC(10,  5, 0, "____{}__", 1234);
    TEST_FULL_PRECALC(10, 11, 1, "____{}__", 1234); // when not stopping there

    // handle indexed parameter
    TEST_FULL_PRECALC(10,  8, 0, "____{1 blah blah }__", 0, 1234);
    TEST_FULL_PRECALC( 8,  8, 0, "____{1 blah blah "   , 0, 1234);
    TEST_FULL_PRECALC( 8,  7, 0, "____{1 blah blah "   , 0, 1234);
    TEST_FULL_PRECALC( 8, 10, 2, "____{1 blah blah "   , 0, 1234);
    TEST_FULL_PRECALC( 9,  5, 0, "____{10 }__", 1234);
    TEST_FULL_PRECALC( 7,  5, 0, "____{10 ",    1234);
    TEST_FULL_PRECALC( 7,  6, 1, "____{10 ",    1234);

    // handle '{{'
    TEST_FULL_PRECALC(7, 8, 1, "...{{...", 0, 1001);
    TEST_FULL_PRECALC(7, 7, 0, "...{{...", 0, 1001);
    TEST_FULL_PRECALC(7, 6, 0, "...{{...", 0, 1001);

    // handle commented paramenter
    TEST_FULL_PRECALC(9,  6, 0, "...{ blah blah }...", 123);
    TEST_FULL_PRECALC(6,  6, 0, "...{ blah blah"     , 123);
    TEST_FULL_PRECALC(6,  7, 1, "...{ blah blah"     , 123);
    TEST_FULL_PRECALC(6,  7, 1, "...{ blah blah }"   , 123);
    TEST_FULL_PRECALC(9, 10, 1, "...{ blah blah }...", 123);

    // handle comment
    TEST_FULL_PRECALC(6, 10, 4, "...{- a comment here }...", 123);
    TEST_FULL_PRECALC(3, 10, 7, "...{- a comment here }"   , 123);
    TEST_FULL_PRECALC(3, 10, 7, "...{- a comment here"     , 123);
}

} // unammed namespace

STRF_TEST_FUNC void test_input_tr_string()
{
    test_printing();
    test_without_preprinting();
    test_size_precalculation();
    test_width_precalculation();
    test_full_precalculation();
}

REGISTER_STRF_TEST(test_input_tr_string);
