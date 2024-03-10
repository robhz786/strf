//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

template <typename CharT>
struct my_bool_printer
{
    void STRF_HD operator()(strf::destination<CharT>& dst) const
    {
        const int size = 2 + static_cast<int>(value_);
        dst.ensure(size);
        auto *p = dst.buffer_ptr();
        if (value_) {
            p[0] = static_cast<CharT>('y');
            p[1] = static_cast<CharT>('e');
            p[2] = static_cast<CharT>('s');
        } else {
            p[0] = static_cast<CharT>('n');
            p[1] = static_cast<CharT>('o');
        }
        dst.advance(size);
    }

    bool value_;
};

template <typename T>
struct is_bool: std::is_same<T, bool> {};

struct my_bool_printing_overrider
{
    using category = strf::printable_overrider_c_of<bool>;

    template <typename CharT, typename PreMeasurements, typename FPack>
    constexpr static STRF_HD auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack&
        , bool value ) noexcept
    {
        pre->add_width(static_cast<strf::width_t>(2 + static_cast<int>(value)));
        pre->add_size(2 + static_cast<int>(value));
        return my_bool_printer<CharT>{value};
    }

    template <typename CharT, typename PreMeasurements, typename FPack, typename... T>
    static STRF_HD auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::value_and_format<T...> x ) noexcept
    {
        return strf::make_printer<CharT>
            ( pre
            , strf::pack(fp, strf::constrain<is_bool>(my_bool_printing_overrider{}))
            , strf::join(x.value()) .set_alignment_format(x.get_alignment_format()) );
    }
};

struct bool_wrapper {
    bool x;
};

namespace strf {

template <>
struct printable_def<bool_wrapper> {
    using representative = bool_wrapper;
    using forwarded_type = bool_wrapper;
    using override_tag = bool_wrapper;
    using is_overridable = std::true_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , bool_wrapper f )
    {
        return strf::make_printer<CharT>
            ( pre
            , fp
            , strf::join(static_cast<CharT>('['), f.x, static_cast<CharT>(']')) );
    }
};

}// namespace strf

STRF_TEST_FUNC void test_printable_overriding()
{
    auto alt_bool = my_bool_printing_overrider{};

    TEST("yes/no").with(alt_bool) (true, '/', false);
    TEST("..yes../..no..").with(alt_bool) (strf::center(true, 7, '.'), '/', strf::center(false, 6, '.'));
    TEST("no").with(my_bool_printing_overrider{}) (false);
    TEST("123").with(alt_bool) (123);
    TEST("[yes]/[no]").with(alt_bool) (bool_wrapper{true}, '/', bool_wrapper{false});

}

REGISTER_STRF_TEST(test_printable_overriding)

