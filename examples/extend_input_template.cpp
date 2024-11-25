//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>

namespace xxx {

template <typename T>
struct base {
    explicit base(const T& t) : value(t) {}
    T value;
};
} // namespace xxx

namespace strf {

template <typename T>
struct base_printing {
    using representative = xxx::base<T>;
    using forwarded_type = strf::reference_wrapper<const xxx::base<T>>;

    template <typename CharT, typename PreMeasurements, typename FPack>
    static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , const xxx::base<T>& x ) noexcept
    {
        return strf::make_printer<CharT>(pre, fp, x.value);
    }
};

template <typename T>
inline base_printing<T> get_printable_def(strf::printable_tag, const xxx::base<T>&)
    { return {}; }

} // namespace strf

namespace yyy {

template <typename T>
struct derived: xxx::base<T> {
    explicit derived(const T& t): xxx::base<T>{t} {}
};

} // namespace yyy

int main()
{
    yyy::derived<int> b{55};
    auto s = strf::to_string(b);
    assert(s == "55");

    return 0;
}
