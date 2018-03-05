//[ custom_decoding_error_handling

#include <boost/assert.hpp>
#include <boost/stringify.hpp>
#include <system_error>

namespace strf = boost::stringify::v0;

bool cause_error_code(strf::u32output& out)
{
    out.set_error(std::make_error_code(std::errc::illegal_byte_sequence));
    return false;
}

bool replace_by_X(strf::u32output& out)
{
    out.put(U'X');
    return true;
}

int main()
{

    {   // invalid UTF-8 / error code
        auto xstr = strf::make_u32string
            .with(strf::make_u8decoder(cause_error_code))
            = { "blah blah \xFF\xFF\xFF blah" };

        BOOST_ASSERT(! xstr);
        BOOST_ASSERT(xstr.error() == std::errc::illegal_byte_sequence);
    }

    {   // invalid UTF-8 / replace
        auto xstr = strf::make_u32string
            .with(strf::make_u8decoder(replace_by_X))
            = { "blah blah \xFF blah" };

        BOOST_ASSERT(xstr.value() == U"blah blah X blah");
    }


    char16_t u16input[] = u"blah blah _ blah";
    u16input[10] = static_cast<char16_t>(0xD800);

    
    {   // invalid UTF-16 / error code
        auto xstr = strf::make_u32string
            .with(strf::make_u16decoder<char16_t>(cause_error_code)) = { u16input };

        BOOST_ASSERT( ! xstr);
        BOOST_ASSERT(xstr.error() == std::errc::illegal_byte_sequence);
    }

    {   // invalid UTF-16 / replace
        auto xstr = strf::make_u32string
            .with(strf::make_u16decoder<char16_t>(replace_by_X)) = { u16input };

        BOOST_ASSERT(xstr.value() == U"blah blah X blah");
    }
    
    
    return 0;
}

//]
