#ifndef BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP
#define BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>

struct exception_tag {};

constexpr exception_tag exception_thrower_arg {};

namespace detail{

template <typename CharT>
class exceptional_printer: public boost::stringify::v0::printer<CharT>
{

public:

    exceptional_printer( exception_tag) noexcept
    {
    }

    std::size_t necessary_size() const override
    {
        return 0;
    }

    bool write(boost::stringify::v0::output_buffer<CharT>& ob) const override
    {
        ob.recycle();
        throw std::invalid_argument("invalid printer");
    }

    int width(int) const override
    {
        return 0;
    }
};

} // namespace detail


//BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, typename FPack>
inline ::detail::exceptional_printer<CharT>
make_printer( const FPack&, exception_tag x )
{
    return {x};
}

//BOOST_STRINGIFY_V0_NAMESPACE_END

#endif
