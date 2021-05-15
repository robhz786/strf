//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

template <typename CharT>
class my_bool_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    constexpr STRF_HD my_bool_printer
        ( const strf::usual_printer_input<T...>& input )
        : value_(input.arg)
    {
        input.preview.subtract_width(2 + (int)input.arg);
        input.preview.add_size(2 + (int)input.arg);
    }

    void STRF_HD print_to(strf::basic_outbuff<CharT>& ob) const override
    {
        int size = 2 + (int)value_;
        ob.ensure(size);
        auto p = ob.pointer();
        if (value_) {
            p[0] = static_cast<CharT>('y');
            p[1] = static_cast<CharT>('e');
            p[2] = static_cast<CharT>('s');
        } else {
            p[0] = static_cast<CharT>('n');
            p[1] = static_cast<CharT>('o');
        }
        ob.advance(size);
    }

private:

    bool value_;
};

struct my_bool_printing_override
{
    using category = strf::print_override_c;

    template <typename CharT, typename Preview, typename FPack>
    constexpr static STRF_HD auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , bool x ) noexcept
        -> strf::usual_printer_input<CharT, Preview, FPack, bool, my_bool_printer<CharT>>
    {
        return {preview, fp, x};
    }

    template <typename CharT, typename Preview, typename FPack, typename... T>
    constexpr static STRF_HD auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , strf::value_with_formatters<T...> x ) noexcept
    {
        return strf::make_printer_input<CharT>
            ( preview
            , fp
            , strf::join(x.value())
                .set_alignment_format(x.get_alignment_format())  );
    }
};

struct my_int_printing_override
{
    using category = strf::print_override_c;

    template <typename CharT, typename Preview, typename FPack, typename... T>
    constexpr static STRF_HD auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack&
        , strf::value_with_formatters<T...> x ) noexcept
    {
        return strf::make_printer_input<CharT>
            ( preview
            , strf::pack()
            , strf::join
                ( static_cast<CharT>('(')
                , strf::fmt(x.value()).set_int_format(x.get_int_format())
                , static_cast<CharT>(')') )
                .set_alignment_format(x.get_alignment_format()) );
    }
};


template <typename T>
struct is_bool: std::is_same<T, bool> {};


struct bool_wrapper {
    bool x;
};


namespace strf {

template <>
struct print_traits<bool_wrapper> {
    using forwarded_type = bool_wrapper;
    using override_tag = bool_wrapper;

    template <typename CharT, typename Preview, typename FPack>
    STRF_HD static auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , bool_wrapper f )
    {
        auto arg2 = strf::join(static_cast<CharT>('['), f.x, static_cast<CharT>(']'));
        auto fp2 = strf::pack(fp, strf::constrain<strf::is_char>(strf::no_print_override{}));
        return strf::make_printer_input<CharT>(preview, fp2, arg2);
    }
};

}// namepsace strf

void STRF_TEST_FUNC test_printable_overriding()
{
    auto alt_bool = strf::constrain<is_bool>(my_bool_printing_override{});
    auto alt_int = strf::constrain<strf::is_int_number>(my_int_printing_override{});

    TEST("yes/no").with(alt_bool) (true, '/', false);
    TEST("..yes../..no..").with(alt_bool) (strf::center(true, 7, '.'), '/', strf::center(false, 6, '.'));
    TEST("no").with(my_bool_printing_override{}) (false);
    TEST("123").with(alt_bool) (123);
    TEST("[yes]/[no]").with(alt_bool) (bool_wrapper{true}, '/', bool_wrapper{false});

    TEST("(5)").with(alt_int) (5);
    TEST("__(abcd)").with(alt_int) (strf::right(0xabcd, 8, '_').hex());
    TEST("[yes](0)[no]").with(alt_bool, alt_int)
        (bool_wrapper{true}, 0, bool_wrapper{false});

}

REGISTER_STRF_TEST(test_printable_overriding);

