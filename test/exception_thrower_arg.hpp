#ifndef STRF_TEST_EXCEPTION_THROWER_HPP
#define STRF_TEST_EXCEPTION_THROWER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>

struct exception_tag {};

constexpr exception_tag exception_thrower_arg {};

namespace detail{

template <typename CharT>
class exceptional_printer: public strf::printer<CharT>
{

public:

    exceptional_printer( exception_tag) noexcept
    {
    }

    std::size_t necessary_size() const override
    {
        return 0;
    }

    void write(strf::basic_outbuf<CharT>& ob) const override
    {
        ob.recycle();
        throw std::invalid_argument("invalid printer");
    }

    strf::width_t width(strf::width_t) const override
    {
        return 0;
    }
};

} // namespace detail


template <typename CharT, typename FPack>
inline ::detail::exceptional_printer<CharT>
make_printer( const FPack&, exception_tag x )
{
    return {x};
}

#endif
