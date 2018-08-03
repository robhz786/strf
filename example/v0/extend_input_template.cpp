#include <boost/stringify.hpp>
#include <boost/assert.hpp>

namespace strf = boost::stringify::v0;

namespace xxx
{

template <typename T>
struct base
{
    base(const T& t) : value(t) {}
    T value;
};


template <typename CharOut, typename FPack, typename T>
inline auto stringify_make_printer
    ( strf::output_writer<CharOut>& out
    , const FPack& fp
    , const base<T> b )
{
    return stringify_make_printer<CharOut, FPack>(out, fp, b.value);
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
    auto s = strf::to_string(b).value();
    BOOST_ASSERT(s == "55");

    return 0;
}
