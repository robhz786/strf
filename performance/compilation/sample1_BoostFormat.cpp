#include <boost/format.hpp>

void write_sample(std::ostream& out)
{
    out << boost::format("%1%") << 20;
    out << boost::format("%1%  %2%") << 20 << 20u;
    out << boost::format("%1%  %2%  %3%") << 20 << 20u << 20l;
    out << boost::format("%1%  %2%  %3%  %4%  %5%  %6%")
        << 20 << 20u << 20l << 20ul << 20ll << 20ul;
    out << boost::format("aaa %-10x bbb") << 123;
    out << boost::format("%x %s %d") << 123 << "abcdef" << 456;
   
}
