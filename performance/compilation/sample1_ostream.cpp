#include <ostream>
#include <iomanip>

void write_sample(std::ostream& out)
{
    out << 20 ;
    out << 20 << "  " << 20u;
    out << 20 << "  " << 20u << "  " << 20l ;
    out << 20 << "  " << 20u << "  " << 20l<< "  "
        << 20ul<< "  " << 20ll << "  " << 20ul;

    out << "aaa " << std::left << std::hex << std::setw(10) << 123 << " bbb";
    out << std::hex << 123 << "abcdef" << std::dec << 456;
};
