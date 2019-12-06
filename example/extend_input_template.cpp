//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>

namespace xxx
{

template <typename T>
struct base
{
    base(const T& t) : value(t) {}
    T value;
};


template <typename CharOut, typename FPack, typename Preview, typename T>
inline auto make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , const base<T> b )
{
    return make_printer<CharOut, FPack>(strf::rank<5>{}, fp, preview, b.value);
}


} // namespace xxx


namespace yyy
{

template <typename T>
struct derived: xxx::base<T>
{
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
