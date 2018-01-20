//[ custom_encoding_error_handling

#include <boost/assert.hpp>
#include <boost/stringify.hpp>
#include <system_error>

namespace strf = boost::stringify::v0;

template <typename CharT>
bool cause_error_code(strf::output_writer<CharT>& out, std::size_t)
{
    out.set_error(std::make_error_code(std::errc::illegal_byte_sequence));
    return false;
}

int main()
{
    // surrogates halves allowed during the convertion from UTF-8 to UTF-32
    auto f1 = strf::make_u8decoder().wtf8();

    // but not in the convertion from UTF-32 to UTF-16
    auto f2 = strf::make_u16encoder<char16_t>(cause_error_code<char16_t>);
    
    {
        const char* u8str_with_surr_D800  = "-- \xED\xA0\x80 --";
        
        auto xstr = strf::make_u16string.with(f1, f2) = {u8str_with_surr_D800};
        BOOST_ASSERT(! xstr);
        BOOST_ASSERT(xstr.error() == std::errc::illegal_byte_sequence);
    }

    
    {
        const char32_t u32str_with_surr_D800[] =
            {U'-', U'-', U' ', 0xD800, U' ', U'-', U'-'};

        auto xstr = strf::make_u16string.with(f1, f2) = {u32str_with_surr_D800};
        BOOST_ASSERT(! xstr);
        BOOST_ASSERT(xstr.error() == std::errc::illegal_byte_sequence);

    }

    return 0;
}
//]
