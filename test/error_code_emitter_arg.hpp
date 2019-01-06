#ifndef BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP
#define BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/printer.hpp>

struct error_tag
{
    std::error_code ec;
};

static error_tag error_code_emitter_arg{ std::make_error_code(std::errc::invalid_argument) };


namespace detail{

template <typename CharT>
class erroneous_printer: public boost::stringify::v0::printer<CharT>
{

public:

    erroneous_printer ( error_tag t ) noexcept
        : m_err(t.ec)
    {
    }

    std::size_t necessary_size() const override
    {
        return 0;
    }

    bool write
        ( boost::stringify::v0::output_buffer<CharT>& ob
        , boost::stringify::v0::buffer_recycler<CharT>& rec ) const override
    {
        rec.set_error(m_err);
        rec.recycle(ob);
        return false;
    }

    int remaining_width(int w) const override
    {
        return w;
    }

private:

    std::error_code m_err;
};


} // namespace detail


template <typename CharT, typename FPack>
inline ::detail::erroneous_printer<CharT> make_printer(const FPack&, error_tag x)
{
    return {x};
}

#endif
