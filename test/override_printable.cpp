//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <limits>
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
    using category = strf::printing_c;

    template <typename CharT, typename FPack, typename Preview>
    constexpr static STRF_HD auto make_printer_input
        (bool x, const FPack& fp, Preview& preview) noexcept
        -> strf::usual_printer_input<CharT, FPack, Preview, my_bool_printer<CharT>, bool>
    {
        return {fp, preview, x};
    }
};

template <typename T>
struct is_bool: std::is_same<T, bool> {};

int main()
{
    auto f = strf::constrain<is_bool>(my_bool_printing_override{});

    TEST("yes").with(f) (true);
    TEST("no").with(f)  (false);
    TEST("no").with(my_bool_printing_override{}) (false);
    TEST("123").with(f) (123);

    return test_finish();
}

