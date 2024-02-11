//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <functional> // reference_wrapper

#define TEST_NO_RESERVE(EXPECTED)                                           \
    strf::make_printing_syntax                                              \
        ( test_utils::input_tester_creator<char>                            \
            { (EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, 1.0}  \
        , strf::no_reserve() )

#define TEST_RESERVE_GIVEN_SPACE(EXPECTED)                                  \
    strf::make_printing_syntax                                              \
        ( test_utils::input_tester_creator<char>                            \
            { (EXPECTED), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION, 1.0}  \
        , strf::reserve_given_space(strf::detail::str_length((EXPECTED))) ) \


namespace {

enum class mytype{aa, bb, cc};

STRF_HD const char16_t* stringify(mytype e)
{
    switch(e) {
        case mytype::aa: return u"aa";
        case mytype::bb: return u"bb";
        case mytype::cc: return u"cc";
        default:         return u"\uFFFD";
    }
}

} // namespace

namespace strf {

template<>
struct printable_def<mytype>
{
    using representative_type = mytype;
    using forwarded_type = mytype;
    using format_specifiers = strf::tag<strf::alignment_format_specifier>;
    using is_overridable = std::true_type;

    template <typename CharT, typename FPack>
    STRF_HD static void print
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , mytype e )
    {
        to(dst) (fp, strf::unsafe_transcode(stringify(e)));
    }

    template <typename CharT, typename FPack, typename... T>
    STRF_HD static void print
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , const strf::printable_with_fmt<T...>& x )
    {
        const auto e = x.value();
        const auto afmt = x.get_alignment_format();

        to(dst) (fp, strf::unsafe_transcode(stringify(e)).set_alignment_format(afmt));
    }
};

} // namespace strf

namespace {

STRF_TEST_FUNC void test_printable_def_without_make_printer()
{
    {   // test using strf::no_reserve policy
        TEST_NO_RESERVE("bb") (mytype::bb);
        TEST_NO_RESERVE("bb") (strf::fmt(mytype::bb));
        TEST_NO_RESERVE("...aa...") (strf::center(mytype::aa, 8, U'.'));

        // in joins
        TEST_NO_RESERVE("--bb--") (strf::join("--", mytype::bb, "--"));
        TEST_NO_RESERVE("==...cc...==")
            (strf::join("==", strf::center(mytype::cc, 8, U'.'), "=="));

        // in ranges
        const mytype arr[] = {mytype::aa, mytype::bb, mytype::cc};

        TEST_NO_RESERVE("aabbcc") (strf::range(arr));
        TEST_NO_RESERVE("  aa  bb  cc") (strf::fmt_range(arr)>4);
        TEST_NO_RESERVE("aa/bb/cc") (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_NO_RESERVE("::aa::") .tr("::{}::", mytype::aa);
        TEST_NO_RESERVE("::aa:aa:") .tr("::{0}:{0}:", mytype::aa);
        TEST_NO_RESERVE("::__aa__:__aa__:")
            .tr("::{0}:{0}:", strf::center(mytype::aa, 6, '_'));
        TEST_NO_RESERVE("::aa:aa:") .tr("::{0}:{0}:", strf::join(mytype::aa));
        TEST_NO_RESERVE("::-__aa__-:-__aa__-:")
            .tr( "::{0}:{0}:", strf::join('-', strf::center(mytype::aa, 6, '_'), '-' ));
    }
    {
        // test using strf::reserve_given_space policy
        TEST_RESERVE_GIVEN_SPACE("bb") (mytype::bb);
        TEST_RESERVE_GIVEN_SPACE("bb") (strf::fmt(mytype::bb));
        TEST_RESERVE_GIVEN_SPACE("...aa...") (strf::center(mytype::aa, 8, U'.'));

        // in joins
        TEST_RESERVE_GIVEN_SPACE("--bb--") (strf::join("--", mytype::bb, "--"));
        TEST_RESERVE_GIVEN_SPACE("==...cc...==")
            (strf::join("==", strf::center(mytype::cc, 8, U'.'), "=="));

        // in ranges
        const mytype arr[] = {mytype::aa, mytype::bb, mytype::cc};

        TEST_RESERVE_GIVEN_SPACE("aabbcc") (strf::range(arr));
        TEST_RESERVE_GIVEN_SPACE("  aa  bb  cc") (strf::fmt_range(arr)>4);
        TEST_RESERVE_GIVEN_SPACE("aa/bb/cc") (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_RESERVE_GIVEN_SPACE("::aa::") .tr("::{}::", mytype::aa);
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:") .tr("::{0}:{0}:", mytype::aa);
        TEST_RESERVE_GIVEN_SPACE("::__aa__:__aa__:")
            .tr("::{0}:{0}:", strf::center(mytype::aa, 6, '_'));
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:") .tr("::{0}:{0}:", strf::join(mytype::aa));
        TEST_RESERVE_GIVEN_SPACE("::-__aa__-:-__aa__-:")
            .tr("::{0}:{0}:", strf::join('-', strf::center(mytype::aa, 6, '_'), '-' ));
    }
}

struct mytype_overrider_impl
{
    using category = strf::printable_overrider_c;

    template <typename CharT, typename FPack>
    STRF_HD static void print
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , mytype e )
    {
        to(dst) .with(fp)
            ( static_cast<CharT>('[')
            , strf::unsafe_transcode(stringify(e))
            , static_cast<CharT>(']') );
    }

    template <typename CharT, typename FPack, typename... T>
    STRF_HD static void print
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , const strf::printable_with_fmt<T...>& x )
    {
        const auto e = x.value();
        const auto afmt = x.get_alignment_format();

        to(dst) .with(fp)
            ( strf::join
                  ( static_cast<CharT>('[')
                  , strf::unsafe_transcode(stringify(e))
                  , static_cast<CharT>(']') ) .set_alignment_format(afmt) );
    }
};

template <typename T>
struct is_mytype : std::false_type {};

template <>
struct is_mytype<mytype> : std::true_type {};


STRF_TEST_FUNC void test_overrider_without_make_printer()
{
    constexpr auto mytype_overrider = strf::constrain<is_mytype>(mytype_overrider_impl());

    {   // test using strf::no_reserve policy
        TEST_NO_RESERVE("[bb]").with(mytype_overrider) (mytype::bb);
        TEST_NO_RESERVE("[bb]").with(mytype_overrider) (strf::fmt(mytype::bb));
        TEST_NO_RESERVE("..[aa]..").with(mytype_overrider) (strf::center(mytype::aa, 8, U'.'));

        // in joins
        TEST_NO_RESERVE("--[bb]--").with(mytype_overrider) (strf::join("--", mytype::bb, "--"));
        TEST_NO_RESERVE("==..[cc]..==").with(mytype_overrider)
            (strf::join("==", strf::center(mytype::cc, 8, U'.'), "=="));

        // in ranges
        const mytype arr[] = {mytype::aa, mytype::bb, mytype::cc};

        TEST_NO_RESERVE("[aa][bb][cc]").with(mytype_overrider) (strf::range(arr));
        TEST_NO_RESERVE("  [aa]  [bb]  [cc]").with(mytype_overrider) (strf::fmt_range(arr)>6);
        TEST_NO_RESERVE("[aa]/[bb]/[cc]").with(mytype_overrider) (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_NO_RESERVE("::[aa]::").with(mytype_overrider) .tr("::{}::", mytype::aa);
        TEST_NO_RESERVE("::[aa]:[aa]:").with(mytype_overrider) .tr("::{0}:{0}:", mytype::aa);
        TEST_NO_RESERVE("::__[aa]__:__[aa]__:").with(mytype_overrider)
            .tr("::{0}:{0}:", strf::center(mytype::aa, 8, '_'));
        TEST_NO_RESERVE("::[aa]:[aa]:").with(mytype_overrider) .tr("::{0}:{0}:", strf::join(mytype::aa));
        TEST_NO_RESERVE("::-__[aa]__-:-__[aa]__-:").with(mytype_overrider)
            .tr( "::{0}:{0}:", strf::join('-', strf::center(mytype::aa, 8, '_'), '-' ));
    }
    {
        // test using strf::reserve_given_space policy
        TEST_RESERVE_GIVEN_SPACE("[bb]").with(mytype_overrider) (mytype::bb);
        TEST_RESERVE_GIVEN_SPACE("[bb]").with(mytype_overrider) (strf::fmt(mytype::bb));
        TEST_RESERVE_GIVEN_SPACE("...[aa]...").with(mytype_overrider) (strf::center(mytype::aa, 10, U'.'));

        // in joins
        TEST_RESERVE_GIVEN_SPACE("--[bb]--").with(mytype_overrider) (strf::join("--", mytype::bb, "--"));
        TEST_RESERVE_GIVEN_SPACE("==...[cc]...==").with(mytype_overrider)
            (strf::join("==", strf::center(mytype::cc, 10, U'.'), "=="));

        // in ranges
        const mytype arr[] = {mytype::aa, mytype::bb, mytype::cc};

        TEST_RESERVE_GIVEN_SPACE("[aa][bb][cc]").with(mytype_overrider) (strf::range(arr));
        TEST_RESERVE_GIVEN_SPACE("  [aa]  [bb]  [cc]").with(mytype_overrider) (strf::fmt_range(arr)>6);
        TEST_RESERVE_GIVEN_SPACE("[aa]/[bb]/[cc]").with(mytype_overrider) (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_RESERVE_GIVEN_SPACE("::[aa]::").with(mytype_overrider) .tr("::{}::", mytype::aa);
        TEST_RESERVE_GIVEN_SPACE("::[aa]:[aa]:").with(mytype_overrider) .tr("::{0}:{0}:", mytype::aa);
        TEST_RESERVE_GIVEN_SPACE("::__[aa]__:__[aa]__:").with(mytype_overrider)
            .tr("::{0}:{0}:", strf::center(mytype::aa, 8, '_'));
        TEST_RESERVE_GIVEN_SPACE("::[aa]:[aa]:").with(mytype_overrider)
            .tr("::{0}:{0}:", strf::join(mytype::aa));
        TEST_RESERVE_GIVEN_SPACE("::-__[aa]__-:-__[aa]__-:").with(mytype_overrider)
            .tr("::{0}:{0}:", strf::join('-', strf::center(mytype::aa, 8, '_'), '-' ));
    }
}

class my_abstract_type
{
public:
    my_abstract_type() = default;
    STRF_HD virtual ~my_abstract_type() {}
    STRF_HD virtual const char* msg() const = 0;
};

class my_derived_type_a: public my_abstract_type
{
public:
    my_derived_type_a() = default;

    STRF_HD virtual const char* msg() const override
    {
        return "aa";
    };
};

class my_derived_type_b: public my_abstract_type
{
public:
    my_derived_type_b() = default;

    STRF_HD virtual const char* msg() const override
    {
        return "bb";
    };
};

class my_derived_type_c: public my_abstract_type
{
public:
    my_derived_type_c() = default;

    STRF_HD virtual const char* msg() const override
    {
        return "cc";
    };
};



} // namespace

namespace strf {
template <>
struct printable_def<my_abstract_type>
{
    using representative_type = strf::reference_wrapper<const my_abstract_type>;
    using forwarded_type = strf::reference_wrapper<const my_abstract_type>;
    using format_specifiers = strf::tag<strf::alignment_format_specifier>;

    template <typename CharT, typename FPack>
    STRF_HD static void print
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , const my_abstract_type& x )
    {
        to(dst) (fp, strf::transcode(x.msg()));
    }

    template <typename CharT, typename FPack, typename... T>
    STRF_HD static void print
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , const strf::printable_with_fmt<T...>& x )
    {
        to(dst) .with(fp)
            ( strf::transcode(x.value().get().msg())
              .set_alignment_format(x.get_alignment_format()) );
    }
};

STRF_HD printable_def<my_abstract_type>
tag_invoke(strf::printable_tag, const my_abstract_type&)
{
    return {};
}

} // namespace strf

namespace {

STRF_TEST_FUNC void test_abstract_printable_without_make_printer()
{
    const my_derived_type_a a;
    const my_derived_type_b b;
    const my_derived_type_c c;

    // to silent a warning from clang 6 that tag_invoke is not needed
    (void) tag_invoke(strf::printable_tag(), a);

    const strf::reference_wrapper<const my_abstract_type> arr[] = {a, b, c};

    {   // test using strf::no_reserve policy
        TEST_NO_RESERVE("bb") (b);
        TEST_NO_RESERVE("bb") (strf::fmt(b));
        TEST_NO_RESERVE("...aa...") (strf::center(a, 8, U'.'));

        // in joins
        TEST_NO_RESERVE("--bb--") (strf::join("--", b, "--"));
        TEST_NO_RESERVE("==...cc...==")
            (strf::join("==", strf::center(c, 8, U'.'), "=="));

        // in ranges
        TEST_NO_RESERVE("aabbcc") (strf::range(arr));
        TEST_NO_RESERVE("  aa  bb  cc") (strf::fmt_range(arr)>4);
        TEST_NO_RESERVE("aa/bb/cc") (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_NO_RESERVE("::aa::") .tr("::{}::", a);
        TEST_NO_RESERVE("::aa:aa:") .tr("::{0}:{0}:", a);
        TEST_NO_RESERVE("::__aa__:__aa__:")
            .tr("::{0}:{0}:", strf::center(a, 6, '_'));
        TEST_NO_RESERVE("::aa:aa:") .tr("::{0}:{0}:", strf::join(a));
        TEST_NO_RESERVE("::-__aa__-:-__aa__-:")
            .tr( "::{0}:{0}:", strf::join('-', strf::center(a, 6, '_'), '-' ));
    }
    {
        // test using strf::reserve_given_space policy
        TEST_RESERVE_GIVEN_SPACE("bb") (b);
        TEST_RESERVE_GIVEN_SPACE("bb") (strf::fmt(b));
        TEST_RESERVE_GIVEN_SPACE("...aa...") (strf::center(a, 8, U'.'));

        // in joins
        TEST_RESERVE_GIVEN_SPACE("--bb--") (strf::join("--", b, "--"));
        TEST_RESERVE_GIVEN_SPACE("==...cc...==")
            (strf::join("==", strf::center(c, 8, U'.'), "=="));

        // in ranges
        TEST_RESERVE_GIVEN_SPACE("aabbcc") (strf::range(arr));
        TEST_RESERVE_GIVEN_SPACE("  aa  bb  cc") (strf::fmt_range(arr)>4);
        TEST_RESERVE_GIVEN_SPACE("aa/bb/cc") (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_RESERVE_GIVEN_SPACE("::aa::") .tr("::{}::", a);
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:") .tr("::{0}:{0}:", a);
        TEST_RESERVE_GIVEN_SPACE("::__aa__:__aa__:")
            .tr("::{0}:{0}:", strf::center(a, 6, '_'));
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:") .tr("::{0}:{0}:", strf::join(a));
        TEST_RESERVE_GIVEN_SPACE("::-__aa__-:-__aa__-:")
            .tr("::{0}:{0}:", strf::join('-', strf::center(a, 6, '_'), '-' ));
    }
}

} // namespace

namespace {

enum class mytype2{aa, bb, cc};

STRF_HD const char16_t* stringify(mytype2 e)
{
    switch(e) {
        case mytype2::aa: return u"aa";
        case mytype2::bb: return u"bb";
        case mytype2::cc: return u"cc";
        default:          return u"\uFFFD";
    }
}

} // namespace

namespace strf {

template <>
struct printable_def<mytype2>
{
    using representative_type = mytype2;
    using forwarded_type = mytype2;
    using is_overridable = std::true_type;

    template <typename CharT, typename Pre, typename FPack>
    STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , Pre* pre
        , const FPack& fp
        , mytype2 e )
    {
        pre->add_width(static_cast<strf::width_t>(2));
        pre->add_size(2);

        auto charset = strf::use_facet<strf::charset_c<CharT>, void> (fp);
        return [e, charset] (strf::destination<CharT>& dst)
               {
                   to(dst) (charset, strf::unsafe_transcode(stringify(e)));
               };
    }
};

} // namespace strf

namespace {

STRF_TEST_FUNC void test_make_printer_that_returns_lambda()
{
    {   // test using strf::no_reserve policy
        TEST_NO_RESERVE("bb") (mytype2::bb);
        TEST_NO_RESERVE("bb") (strf::fmt(mytype2::bb));

        // in joins
        TEST_NO_RESERVE("--bb--") (strf::join("--", mytype2::bb, "--"));

        // in ranges
        const mytype2 arr[] = {mytype2::aa, mytype2::bb, mytype2::cc};

        TEST_NO_RESERVE("aabbcc") (strf::range(arr));
        TEST_NO_RESERVE("aa/bb/cc") (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_NO_RESERVE("::aa::") .tr("::{}::", mytype2::aa);
        TEST_NO_RESERVE("::aa:aa:") .tr("::{0}:{0}:", mytype2::aa);
        TEST_NO_RESERVE("::aa:aa:") .tr("::{0}:{0}:", strf::join(mytype2::aa));
    }
    {
        // test using strf::reserve_given_space policy
        TEST_RESERVE_GIVEN_SPACE("bb") (mytype2::bb);
        TEST_RESERVE_GIVEN_SPACE("bb") (strf::fmt(mytype2::bb));

        // in joins
        TEST_RESERVE_GIVEN_SPACE("--bb--") (strf::join("--", mytype2::bb, "--"));

        // in ranges
        const mytype2 arr[] = {mytype2::aa, mytype2::bb, mytype2::cc};

        TEST_RESERVE_GIVEN_SPACE("aabbcc") (strf::range(arr));
        TEST_RESERVE_GIVEN_SPACE("aa/bb/cc") (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_RESERVE_GIVEN_SPACE("::aa::") .tr("::{}::", mytype2::aa);
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:") .tr("::{0}:{0}:", mytype2::aa);
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:")
            .tr("::{0}:{0}:", strf::join(mytype2::aa));
    }
    {
        // test using strf::reserve_calc policy
        TEST("bb").with() (mytype2::bb);
        TEST("bb") (strf::fmt(mytype2::bb));

        // in joins
        TEST("--bb--") (strf::join("--", mytype2::bb, "--"));

        // in ranges
        const mytype2 arr[] = {mytype2::aa, mytype2::bb, mytype2::cc};

        TEST("aabbcc") (strf::range(arr));
        TEST("aa/bb/cc") (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST("::aa::") .tr("::{}::", mytype2::aa);
        TEST("::aa:aa:") .tr("::{0}:{0}:", mytype2::aa);
        TEST("::aa:aa:") .tr("::{0}:{0}:", strf::join(mytype2::aa));
        TEST(":: aa : aa :")
            .tr("::{0}:{0}:", strf::join_center(4)(mytype2::aa));
    }
}


struct mytype2_overrider_impl
{
    using category = strf::printable_overrider_c;

    template <typename CharT, typename Pre, typename FPack>
    STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , Pre* pre
        , const FPack& fp
        , mytype2 e )
    {
        pre->add_width(static_cast<strf::width_t>(4));
        pre->add_size(4);

        auto charset = strf::use_facet<strf::charset_c<CharT>, void> (fp);
        return [e, charset] (strf::destination<CharT>& dst)
               {
                   to(dst).with(charset)
                       ( static_cast<CharT>('[')
                       , strf::unsafe_transcode(stringify(e))
                       , static_cast<CharT>(']') );
               };
    }
};

template <typename T>
struct is_mytype2 : std::false_type {};

template <>
struct is_mytype2<mytype2> : std::true_type {};



STRF_TEST_FUNC void test_overrider_with_make_printer()
{
    constexpr auto mytype2_overrider = strf::constrain<is_mytype2>(mytype2_overrider_impl());

    {   // test using strf::no_reserve policy
        TEST_NO_RESERVE("[bb]").with(mytype2_overrider) (mytype2::bb);
        TEST_NO_RESERVE("[bb]").with(mytype2_overrider) (strf::fmt(mytype2::bb));

        // in joins
        TEST_NO_RESERVE("..--[bb]--..").with(mytype2_overrider)
            (strf::join_center(12, '.')("--", mytype2::bb, "--"));

        // in ranges
        const mytype2 arr[] = {mytype2::aa, mytype2::bb, mytype2::cc};

        TEST_NO_RESERVE("[aa][bb][cc]").with(mytype2_overrider)
            (strf::range(arr));
        TEST_NO_RESERVE("[aa]/[bb]/[cc]").with(mytype2_overrider)
            (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_NO_RESERVE("::[aa]::").with(mytype2_overrider)
            .tr("::{}::", mytype2::aa);
        TEST_NO_RESERVE("::[aa]:[aa]:").with(mytype2_overrider)
            .tr("::{0}:{0}:", mytype2::aa);
        TEST_NO_RESERVE("::[aa]:[aa]:").with(mytype2_overrider)
            .tr("::{0}:{0}:", strf::join(mytype2::aa));
    }
    {   // test using strf::reserve_given_space policy
        TEST_RESERVE_GIVEN_SPACE("[bb]").with(mytype2_overrider) (mytype2::bb);
        TEST_RESERVE_GIVEN_SPACE("[bb]").with(mytype2_overrider) (strf::fmt(mytype2::bb));

        // in joins
        TEST_RESERVE_GIVEN_SPACE("..--[bb]--..").with(mytype2_overrider)
            (strf::join_center(12, '.')("--", mytype2::bb, "--"));

        // in ranges
        const mytype2 arr[] = {mytype2::aa, mytype2::bb, mytype2::cc};

        TEST_RESERVE_GIVEN_SPACE("[aa][bb][cc]").with(mytype2_overrider)
            (strf::range(arr));
        TEST_RESERVE_GIVEN_SPACE("[aa]/[bb]/[cc]").with(mytype2_overrider)
            (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_RESERVE_GIVEN_SPACE("::[aa]::").with(mytype2_overrider)
            .tr("::{}::", mytype2::aa);
        TEST_RESERVE_GIVEN_SPACE("::[aa]:[aa]:").with(mytype2_overrider)
            .tr("::{0}:{0}:", mytype2::aa);
        TEST_RESERVE_GIVEN_SPACE("::[aa]:[aa]:").with(mytype2_overrider)
            .tr("::{0}:{0}:", strf::join(mytype2::aa));
    }
    {   // test using strf::reserve_calc policy
        TEST("[bb]").with(mytype2_overrider) (mytype2::bb);
        TEST("[bb]").with(mytype2_overrider) (strf::fmt(mytype2::bb));

        // in joins
        TEST("..--[bb]--..").with(mytype2_overrider)
            (strf::join_center(12, '.')("--", mytype2::bb, "--"));

        // in ranges
        const mytype2 arr[] = {mytype2::aa, mytype2::bb, mytype2::cc};

        TEST("[aa][bb][cc]").with(mytype2_overrider) (strf::range(arr));
        TEST("[aa]/[bb]/[cc]").with(mytype2_overrider) (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST("::[aa]::").with(mytype2_overrider) .tr("::{}::", mytype2::aa);
        TEST("::[aa]:[aa]:").with(mytype2_overrider) .tr("::{0}:{0}:", mytype2::aa);
        TEST("::[aa]:[aa]:").with(mytype2_overrider) .tr("::{0}:{0}:", strf::join(mytype2::aa));
    }
}

STRF_TEST_FUNC void test_all()
{
    test_printable_def_without_make_printer();
    test_overrider_without_make_printer();
    test_abstract_printable_without_make_printer();
    test_make_printer_that_returns_lambda();
    test_overrider_with_make_printer();
}

} // namespace

REGISTER_STRF_TEST(test_all)
