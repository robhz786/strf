#ifndef BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP
#define BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/stringifier.hpp>

struct error_tag
{
    std::error_code ec;
};

static error_tag error_code_emitter_arg{ std::make_error_code(std::errc::invalid_argument) };


namespace detail{

template <typename CharT>
class erroneous_stringifier: public boost::stringify::v0::stringifier<CharT>
{

public:

    template <typename FTuple>
    erroneous_stringifier(const FTuple&, error_tag t) noexcept
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

struct erroneous_stringifier_traits
{
    template <typename CharT, typename>
    using stringifier = erroneous_stringifier<CharT>;    
};


}

detail::erroneous_stringifier_traits boost_stringify_input_traits_of(error_tag);


#endif
