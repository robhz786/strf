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
        (Preview& preview, const FPack& fp, bool x) noexcept
        -> strf::usual_printer_input<CharT, Preview, FPack, bool, my_bool_printer<CharT>>
    {
        return {preview, fp, x};
    }

    template <typename CharT, typename Preview, typename FPack, typename... T>
    constexpr static STRF_HD auto make_printer_input
        (Preview& preview, const FPack& fp, strf::value_with_format<T...> x) noexcept
    {
        return strf::make_printer_input<CharT>
            ( preview, fp, strf::join(x.value()).set(x.get_alignment_format_data())  );
    }
};

template <typename T>
struct is_bool: std::is_same<T, bool> {};

void STRF_TEST_FUNC test_printable_overriding()
{
    auto f = strf::constrain<is_bool>(my_bool_printing_override{});

    TEST("yes/no").with(f) (true, '/', false);
    TEST("..yes../..no..").with(f) (strf::center(true, 7, '.'), '/', strf::center(false, 6, '.'));
    TEST("no").with(my_bool_printing_override{}) (false);
    TEST("123").with(f) (123);
}

