#include <boost/stringify.hpp>

namespace strf = boost::stringify;

template <typename W>
void write_samples(const W& write)
{
    write("ten = ", 10, " and twenty = ", 20, "\n");
    write(222, " in hexadecimal is ", {222, "#x"});
    write
        ( {20, {5, "<#x"}}
        , {20u, {5, "<#x"}}
        , {20l, {5, "<#x"}}
        , {20ul, {5, "<#x"}}
        , {20ll, {5, "<#x"}}
        , {20ull, {5, "<#x"}}
        , {(short)20, {5, "<#x"}}
        , {(unsigned short)20, {5, "<#x"}});

    write.with(strf::hex, strf::width(5), strf::left, strf::showbase)
        (20, 20u, 20l, 20ul, 20ll, 20ull, (short)20, (unsigned short)20);

    write("aaaa", {strf::join_right(30), {"bbb", 555, "ccc"}});
}


int main()
{
    char buff[200];
    write_samples(strf::write_to(buff));
    write_samples(strf::write_to(buff).with(strf::left, strf::showbase));
    write_samples(strf::make_string);
    write_samples(strf::make_string.with(strf::left, strf::fill(U'~')));
    std::string tmp;
    write_samples(strf::append_to(tmp));
    write_samples(strf::append_to(tmp).with(strf::showbase, strf::showpos));
    write_samples(strf::assign_to(tmp));
    write_samples(strf::assign_to(tmp).with(strf::left));

    return 0;
}
