#ifndef BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP
#define BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/formatter.hpp>

struct error_tag
{
    std::error_code ec;
};

static error_tag error_code_emitter_arg{ std::make_error_code(std::errc::invalid_argument) };


namespace detail{

template <typename CharT>
class erroneous_formatter: public boost::stringify::v0::formatter<CharT>
{

public:

    template <typename FTuple>
    erroneous_formatter(const FTuple&, error_tag t) noexcept
        : m_err(t.ec)
    {
    }

    std::size_t length() const override
    {
        return 0;
    }

    void write(boost::stringify::v0::output_writer<CharT>& out) const override
    {
        out.set_error(m_err);
    }

    int remaining_width(int w) const override
    {
        return w;
    }

private:

    std::error_code m_err;
};

} // namespace detail

template <typename CharT, typename FTuple>
inline detail::erroneous_formatter<CharT>
boost_stringify_make_formatter(const FTuple& ft, error_tag x)
{
    return {ft, x};
}

#endif
