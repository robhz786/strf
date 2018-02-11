#include <boost/stringify.hpp>

void sample_hex()
{
    //[ trivial_hex_sample
    
    namespace strf = boost::stringify::v0;

    auto str = strf::make_string ["{}  {}"] &= {255, strf::hex(255)};

    BOOST_ASSERT(str == "255  ff");
    //]
}


void samples()
{
    //[ formatting_samples
    namespace strf = boost::stringify::v0;

    auto str = strf::make_string ["{}/{}/{}/{}/{}"] &=
    {
        strf::hex(255) > 5,
        strf::center(255, 7, '.').hex(),
        ~strf::uphex(255) % 7,
        strf::multi('a', 3) ^ 7,
        +strf::fmt(255)
    };
    
    BOOST_ASSERT(str == "   ff/..ff.../0X   FF/  aaa  /+255");
    //]
}


int main()
{
    sample_hex();
    samples();

    return 0;
}
