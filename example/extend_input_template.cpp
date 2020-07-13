//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>

namespace xxx {

template <typename T>
struct base {
    base(const T& t) : value(t) {}
    T value;
};
} // namespace xxx

namespace strf {

template <typename CharT, typename T, typename FPack, typename Preview>
inline auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , const xxx::base<T>& x
    , const FPack& fp
    , Preview& preview ) noexcept
{
    return strf::make_default_printer_input<CharT>(x.value, fp, preview);
}

} // namespace strf

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
