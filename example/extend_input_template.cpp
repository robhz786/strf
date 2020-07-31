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

template <typename T>
struct base_printing {
    using facet_tag = const xxx::base<T>&;
    using forwarded_type = const xxx::base<T>&;

    template <typename CharT, typename Preview, typename FPack>
    static auto make_input(forwarded_type x, Preview& preview, const FPack& fp ) noexcept
    {
        return strf::make_default_printer_input<CharT>(x.value, preview, fp);
    }
};

template <typename T>
inline base_printing<T> tag_invoke(strf::print_traits_tag, const xxx::base<T>& x)
    { return {}; }

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
