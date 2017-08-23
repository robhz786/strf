#include <boost/stringify.hpp>

namespace strf = boost::stringify;

void write_sample(std::FILE* out)
{
    strf::write_to(out) (20);
    strf::write_to(out) (20, 20u);
    strf::write_to(out) (20, 20u, 20l);
    strf::write_to(out) (20, 20u, 20l, 20ul, 20ll, 20ul);
    strf::write_to(out) ("aaa ", {123, {10, "<x"}}, "bbb");
    strf::write_to(out) ({123, "x"}, "abcdef", 456);
}
