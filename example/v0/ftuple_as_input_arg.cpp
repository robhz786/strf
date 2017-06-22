#include <boost/stringify.hpp>
#include <boost/assert.hpp>

int main()
{
    //[ftuple_as_input_arg
    namespace strf = boost::stringify::v0;

    auto outer_ftuple = strf::make_ftuple(strf::width(3), strf::left);
    auto inner_ftuple = strf::make_ftuple(strf::hex, strf::right);

    auto result = strf::make_string.with(outer_ftuple)
        (10, 11, {inner_ftuple, /*<<
        This sublist of arguments could contain just anything that can be inserted
        in the outer argument list ( including another ftuple with its own
        argument sublist ).
        >>*/{"~~~", 12, 13, {14, "d"}, 15, "~~~"}}, 16, 17);

    BOOST_ASSERT(result == "10 11 ~~~  c  d 14  f~~~16 17 "); /*< Note that `outer_ftuple`
        still has effect inside the sublist ( which is why width is 3 there ),
        but the facets of `inner_ftuple` are prefered
        ( which is why arguments are aligned to the right )>*/
    //]

    return 0;
}
