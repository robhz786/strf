//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

class err_handler {

public:
    using category = strf::tr_error_notifier_c;

    err_handler(const err_handler&) = default;

    STRF_HD err_handler(strf::outbuff& log)
        : log_(log)
    {
    }

    template <typename Enc>
    void STRF_HD handle
        ( const typename Enc::char_type* str
        , std::size_t str_len
        , std::size_t err_pos
        , Enc enc ) noexcept
    {
        strf::detail::simple_string_view<typename Enc::char_type> s(str, str_len);
        strf::to(log_) ("\n[", strf::dec(err_pos) > 2, "] ", strf::conv(s, enc));
    }

private:
    strf::outbuff& log_;
};

void STRF_TEST_FUNC test_tr_string()
{
    // basic test
    TEST("aaa__..bbb__ 0xa")
        .tr("{}__{}__{}", "aaa", strf::right("bbb", 5, '.'), *strf::hex(10)>4);

    TEST(u8"_0__1__2")   .tr(u8"_{}__{}__{}",    0, 1, 2);
    TEST(u8"{0_{_{1_{2") .tr(u8"{{{}_{{_{{{}_{{{}",  0, 1, 2);
    TEST(u8"0__1__2")    .tr(u8"{}__{}__{}",     0, 1, 2);
    TEST(u8"0__1__2")    .tr(u8"{}__{}__{",      0, 1, 2);
    TEST(u8"0__1__2")    .tr(u8"{}__{}__{aaa}",  0, 1, 2);
    TEST(u8"0__1__2")    .tr(u8"{}__{}__{aaa",   0, 1, 2);
    TEST(u8"0__1__2_")   .tr(u8"{}__{}__{}_",    0, 1, 2);
    TEST(u8"0__1__2_")   .tr(u8"{}__{}__{aaa}_", 0, 1, 2);

    TEST(u8"0__1__\uFFFD")    .tr(u8"{}__{}__{}",     0, 1);
    TEST(u8"0__1__\uFFFD")    .tr(u8"{}__{}__{",      0, 1);
    TEST(u8"0__1__\uFFFD")    .tr(u8"{}__{}__{aaa}",  0, 1);
    TEST(u8"0__1__\uFFFD")    .tr(u8"{}__{}__{aaa",   0, 1);
    TEST(u8"0__1__\uFFFD_")   .tr(u8"{}__{}__{}_",    0, 1);
    TEST(u8"0__1__\uFFFD_")   .tr(u8"{}__{}__{aaa}_", 0, 1);

    TEST(u8"0__3__1")  .tr(u8"{}__{3}__{}",       0, 1, 2, 3);
    TEST(u8"0__3__1")  .tr(u8"{}__{3}__{",        0, 1, 2, 3);
    TEST(u8"0__3__1")  .tr(u8"{}__{3aa}__{aaa}",  0, 1, 2, 3);
    TEST(u8"0__3__1")  .tr(u8"{}__{3}__{aaa",     0, 1, 2, 3);
    TEST(u8"0__3__1_") .tr(u8"{}__{3}__{}_",      0, 1, 2, 3);
    TEST(u8"0__3__1_") .tr(u8"{}__{3aa}__{aaa}_", 0, 1, 2, 3);

    TEST(u8"0__1__3")  .tr(u8"{}__{1}__{3}",     0, 1, 2, 3);
    TEST(u8"0__1__3")  .tr(u8"{}__{1a}__{3",     0, 1, 2, 3);
    TEST(u8"0__1__3")  .tr(u8"{}__{1}__{3aaa}",  0, 1, 2, 3);
    TEST(u8"0__1__3")  .tr(u8"{}__{1}__{3aaa",   0, 1, 2, 3);
    TEST(u8"0__1__3_") .tr(u8"{}__{1}__{3}_",    0, 1, 2, 3);
    TEST(u8"0__1__3_") .tr(u8"{}__{1}__{3aaa}_", 0, 1, 2, 3);

    TEST(u8"_0__10__")   .tr(u8"_{}__{10}__"  , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10);
    TEST(u8"_0__10__")   .tr(u8"_{}__{10aa}__", 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10);
    TEST(u8"_0__10__")   .tr(u8"_{}__{10}__"  , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10);
    TEST(u8"_0__10")     .tr(u8"_{}__{10"     , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10);
    TEST(u8"_0__\uFFFD") .tr(u8"_{}__{11"     , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10);
    TEST(u8"_0__\uFFFD") .tr(u8"_{}__{100"    , 0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9, 10);

    // comment at the end
    TEST(u8"0__")  .tr(u8"{}__{-}", 0);
    TEST(u8"0__")  .tr(u8"{}__{-", 0);
    TEST(u8"0__")  .tr(u8"{}__{-aaa", 0);
    TEST(u8"0__")  .tr(u8"{}__{-aaa}", 0);
    // comment in the middle
    TEST(u8"0__1")   .tr(u8"{}__{-}{}", 0, 1);
    TEST(u8"0__1")   .tr(u8"{}__{-aaa}{}", 0, 1);
    TEST(u8"0__~1")  .tr(u8"{}__{-}~{}", 0, 1);
    TEST(u8"0__~1")  .tr(u8"{}__{-aaa}~{}", 0, 1);
    // comment at the begining
    TEST(u8"__0")  .tr(u8"{-}__{}", 0);
    TEST(u8"-__0") .tr(u8"-{-}__{}", 0);
    TEST(u8"{__0") .tr(u8"{{{-}__{}", 0);


    // escaped '{'
    TEST(u8"{}_{0}__{")  .tr(u8"{{}_{{{}}__{{", 0);
    TEST( u"{}_{0}__{")  .tr( u"{{}_{{{}}__{{", 0);

    // invalid arguments
    TEST(u8"0 3 \uFFFD \uFFFD 1") .tr(u8"{} {3} {4} {5} {1}", 0, 1, 2, 3);
    TEST(u8"0 3 \uFFFD \uFFFD")   .tr(u8"{} {3} {4aa} {5}", 0, 1, 2, 3);
    TEST(u8"0 3 \uFFFD \uFFFD")   .tr(u8"{} {3} {11} {5", 0, 1, 2, 3);
    TEST(u8"0 3 \uFFFD \uFFFD")   .tr(u8"{} {3} {111} {5 ", 0, 1, 2, 3);
    TEST(u8"0 3 \uFFFD")          .tr(u8"{} {3} {4}", 0, 1, 2, 3);
    TEST(u8"0 3 \uFFFD")          .tr(u8"{} {3} {4", 0, 1, 2, 3);
    TEST(u8"0 1 2 \uFFFD_")       .tr(u8"{} {} {} {}_", 0, 1, 2);
    TEST(u8"0 1 2 \uFFFD")        .tr(u8"{} {} {} {}", 0, 1, 2);
    TEST(u8"0 1 2 \uFFFD")        .tr(u8"{} {} {} {", 0, 1, 2);
    TEST(u8"0 1 2 \uFFFD_")       .tr(u8"{} {} {} {aa}_", 0, 1, 2);
    TEST(u8"0 1 2 \uFFFD")        .tr(u8"{} {} {} {aa}", 0, 1, 2);
    TEST(u8"0 1 2 \uFFFD")        .tr(u8"{} {} {} {aa", 0, 1, 2);
    TEST(u8"_\uFFFD__")           .tr(u8"_{}__");
    TEST(u8"\uFFFD__")            .tr(u8"{}__");
    TEST(u8"__\uFFFD")            .tr(u8"__{}");
    TEST(u8"\uFFFD")              .tr(u8"{aa}");
    TEST(u8"\uFFFD")              .tr(u8"{}");
    TEST(u8"\uFFFD")              .tr(u8"{0}");
    TEST(u8"\uFFFD")              .tr(u8"{1}");
    TEST(u8"\uFFFD")              .tr(u8"{0 aa}");
    TEST(u8"\uFFFD")              .tr(u8"{1 aa}");
    TEST(u8"\uFFFD")              .tr(u8"{");
    TEST(u8"\uFFFD")              .tr(u8"{aaa");

    // invalid arguments - now in UTF-16
    TEST(u"0 3 \uFFFD \uFFFD 1") .tr(u"{} {3} {4} {5} {1}", 0, 1, 2, 3);
    TEST(u"0 3 \uFFFD \uFFFD")   .tr(u"{} {3} {4aa} {5}", 0, 1, 2, 3);
    TEST(u"0 3 \uFFFD \uFFFD")   .tr(u"{} {3} {11} {5", 0, 1, 2, 3);
    TEST(u"0 3 \uFFFD \uFFFD")   .tr(u"{} {3} {111} {5 ", 0, 1, 2, 3);
    TEST(u"0 3 \uFFFD")          .tr(u"{} {3} {4}", 0, 1, 2, 3);
    TEST(u"0 3 \uFFFD")          .tr(u"{} {3} {4", 0, 1, 2, 3);
    TEST(u"0 1 2 \uFFFD_")       .tr(u"{} {} {} {}_", 0, 1, 2);
    TEST(u"0 1 2 \uFFFD")        .tr(u"{} {} {} {}", 0, 1, 2);
    TEST(u"0 1 2 \uFFFD")        .tr(u"{} {} {} {", 0, 1, 2);
    TEST(u"0 1 2 \uFFFD_")       .tr(u"{} {} {} {aa}_", 0, 1, 2);
    TEST(u"0 1 2 \uFFFD")        .tr(u"{} {} {} {aa}", 0, 1, 2);
    TEST(u"0 1 2 \uFFFD")        .tr(u"{} {} {} {aa", 0, 1, 2);
    TEST(u"_\uFFFD__")           .tr(u"_{}__");
    TEST(u"\uFFFD__")            .tr(u"{}__");
    TEST(u"__\uFFFD")            .tr(u"__{}");
    TEST(u"\uFFFD")              .tr(u"{aa}");
    TEST(u"\uFFFD")              .tr(u"{}");
    TEST(u"\uFFFD")              .tr(u"{0}");
    TEST(u"\uFFFD")              .tr(u"{1}");
    TEST(u"\uFFFD")              .tr(u"{0 aa}");
    TEST(u"\uFFFD")              .tr(u"{1 aa}");
    TEST(u"\uFFFD")              .tr(u"{");
    TEST(u"\uFFFD")              .tr(u"{aaa");

    // Customizing error handling
    {
        char buff[200];
        strf::cstr_writer log(buff);
        TEST(u8"0__\uFFFD--1==2..3::\uFFFD~~")
            .with(err_handler{log})
            .tr(u8"{ }__{10}--{}=={}..{}::{}~~", 0, 1, 2, 3);

        log.finish();
        TEST_CSTR_EQ( buff, "\n[ 5] { }__{10}--{}=={}..{}::{}~~"
                            "\n[23] { }__{10}--{}=={}..{}::{}~~" );
    }
    {
        char buff[200];
        strf::cstr_writer log(buff);
        TEST(u8"0__\uFFFD--1==2..3::\uFFFD~~")
            .with(err_handler{log})
            .tr(u8"{ }__{10}--{}=={}..{}::{blah}~~", 0, 1, 2, 3);

        log.finish();
        TEST_CSTR_EQ(buff, "\n[ 5] { }__{10}--{}=={}..{}::{blah}~~"
                           "\n[23] { }__{10}--{}=={}..{}::{blah}~~" );
    }
    {
        char buff[200];
        strf::cstr_writer log(buff);
        TEST(u"0__2--1==2..3::\uFFFD")
            .with(err_handler{log})
            .tr(u"{ }__{2}--{}=={}..{}::{", 0, 1, 2, 3);

        log.finish();
        TEST_CSTR_EQ(buff, "\n[22] { }__{2}--{}=={}..{}::{");
    }
}

