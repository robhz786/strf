//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>

namespace xxx {

template <typename T>
struct base {
    base(const T& t) : value(t) {}
    T value;
};

template <typename CharT, typename FPack, typename Preview, typename T>
struct base_printable_traits {
    constexpr static auto make_input
        ( const FPack&, Preview& preview, const xxx::base<T>& x ) {
        return strf::make_printer_input<CharT>(strf::pack(), preview, x.value);
    }
};

template <typename CharT, typename FPack, typename Preview, typename T>
base_printable_traits<CharT, FPack, Preview, T> get_printable_traits
( Preview&, const xxx::base<T>&);

} // namespace xxx

namespace yyy {

template <typename T>
struct derived: xxx::base<T> {
    derived(const T& t): xxx::base<T>{t} {}
};

} // namespace yyy

int main()
{
    yyy::derived<int> b{55};
    auto s = strf::to_string(b);
    assert(s == "55");

    return 0;
}
