//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

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
        default:            return u"\uFFFD";
    }
}

} // namespace

namespace strf {

template<>
struct printable_traits<mytype>
{
    using representative_type = mytype;
    using forwarded_type = mytype;
    using formatters = strf::tag<strf::alignment_formatter>;

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

STRF_TEST_FUNC void test_printable_traits_without_make_printer()
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

constexpr auto mytype_overrider = strf::constrain<is_mytype>(mytype_overrider_impl());

STRF_TEST_FUNC void test_overrider_without_make_printer()
{
    {   // test using strf::no_reserve policy
        TEST_NO_RESERVE("bb").with(mytype_overrider) (mytype::bb);
        TEST_NO_RESERVE("bb").with(mytype_overrider) (strf::fmt(mytype::bb));
        TEST_NO_RESERVE("...aa...").with(mytype_overrider) (strf::center(mytype::aa, 8, U'.'));

        // in joins
        TEST_NO_RESERVE("--bb--").with(mytype_overrider) (strf::join("--", mytype::bb, "--"));
        TEST_NO_RESERVE("==...cc...==")
            (strf::join("==", strf::center(mytype::cc, 8, U'.'), "=="));

        // in ranges
        const mytype arr[] = {mytype::aa, mytype::bb, mytype::cc};

        TEST_NO_RESERVE("aabbcc").with(mytype_overrider) (strf::range(arr));
        TEST_NO_RESERVE("  aa  bb  cc").with(mytype_overrider) (strf::fmt_range(arr)>4);
        TEST_NO_RESERVE("aa/bb/cc").with(mytype_overrider) (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_NO_RESERVE("::aa::").with(mytype_overrider) .tr("::{}::", mytype::aa);
        TEST_NO_RESERVE("::aa:aa:").with(mytype_overrider) .tr("::{0}:{0}:", mytype::aa);
        TEST_NO_RESERVE("::__aa__:__aa__:")
            .tr("::{0}:{0}:", strf::center(mytype::aa, 6, '_'));
        TEST_NO_RESERVE("::aa:aa:").with(mytype_overrider) .tr("::{0}:{0}:", strf::join(mytype::aa));
        TEST_NO_RESERVE("::-__aa__-:-__aa__-:").with(mytype_overrider)
            .tr( "::{0}:{0}:", strf::join('-', strf::center(mytype::aa, 6, '_'), '-' ));
    }
    {
        // test using strf::reserve_given_space policy
        TEST_RESERVE_GIVEN_SPACE("bb").with(mytype_overrider) (mytype::bb);
        TEST_RESERVE_GIVEN_SPACE("bb").with(mytype_overrider) (strf::fmt(mytype::bb));
        TEST_RESERVE_GIVEN_SPACE("...aa...").with(mytype_overrider) (strf::center(mytype::aa, 8, U'.'));

        // in joins
        TEST_RESERVE_GIVEN_SPACE("--bb--").with(mytype_overrider) (strf::join("--", mytype::bb, "--"));
        TEST_RESERVE_GIVEN_SPACE("==...cc...==").with(mytype_overrider)
            (strf::join("==", strf::center(mytype::cc, 8, U'.'), "=="));

        // in ranges
        const mytype arr[] = {mytype::aa, mytype::bb, mytype::cc};

        TEST_RESERVE_GIVEN_SPACE("aabbcc").with(mytype_overrider) (strf::range(arr));
        TEST_RESERVE_GIVEN_SPACE("  aa  bb  cc").with(mytype_overrider) (strf::fmt_range(arr)>4);
        TEST_RESERVE_GIVEN_SPACE("aa/bb/cc").with(mytype_overrider) (strf::separated_range(arr, "/"));

        // in tr-strings
        TEST_RESERVE_GIVEN_SPACE("::aa::").with(mytype_overrider) .tr("::{}::", mytype::aa);
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:").with(mytype_overrider) .tr("::{0}:{0}:", mytype::aa);
        TEST_RESERVE_GIVEN_SPACE("::__aa__:__aa__:").with(mytype_overrider)
            .tr("::{0}:{0}:", strf::center(mytype::aa, 6, '_'));
        TEST_RESERVE_GIVEN_SPACE("::aa:aa:").with(mytype_overrider)
            .tr("::{0}:{0}:", strf::join(mytype::aa));
        TEST_RESERVE_GIVEN_SPACE("::-__aa__-:-__aa__-:").with(mytype_overrider)
            .tr("::{0}:{0}:", strf::join('-', strf::center(mytype::aa, 6, '_'), '-' ));
    }
}

class my_abstract_type
{
public:
    STRF_HD virtual ~my_abstract_type() {}
    STRF_HD virtual const char* msg() const = 0;
};

class my_derived_type_a: public my_abstract_type
{
    STRF_HD virtual const char* msg() const override
    {
        return "aa";
    };
};

class my_derived_type_b: public my_abstract_type
{
    STRF_HD virtual const char* msg() const override
    {
        return "bb";
    };
};

class my_derived_type_c: public my_abstract_type
{
    STRF_HD virtual const char* msg() const override
    {
        return "cc";
    };
};



} // namespace

namespace strf {
template <>
struct printable_traits<my_abstract_type>
{
    using representative_type = std::reference_wrapper<const my_abstract_type>;
    using forwarded_type = std::reference_wrapper<const my_abstract_type>;
    using formatters = strf::tag<strf::alignment_formatter>;

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

printable_traits<my_abstract_type> tag_invoke(strf::printable_tag, const my_abstract_type&);

} // namespace strf

namespace {

STRF_TEST_FUNC void test_abstract_printable_without_make_printer()
{
    const my_derived_type_a a;
    const my_derived_type_b b;
    const my_derived_type_c c;

    const std::reference_wrapper<const my_abstract_type> arr[] = {a, b, c};

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

STRF_TEST_FUNC void test_all()
{
    test_printable_traits_without_make_printer();
    test_overrider_without_make_printer();
    test_abstract_printable_without_make_printer();
}

} // namespace

REGISTER_STRF_TEST(test_all)
