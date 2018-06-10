#ifndef BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP
#define BOOST_STRINGIFY_TEST_ERROR_CODE_EMMITER_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/basic_types.hpp>

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

    template <typename FPack>
    erroneous_printer
        ( boost::stringify::v0::output_writer<CharT>& out
        , const FPack&
        , error_tag t )
        noexcept
        : m_out(out)
        , m_err(t.ec)
    {
    }

    std::size_t length() const override
    {
        return 0;
    }

    void write() const override
    {
        m_out.set_error(m_err);
    }

    int remaining_width(int w) const override
    {
        return w;
    }

private:
    boost::stringify::v0::output_writer<CharT>& m_out;
    std::error_code m_err;
};

// struct error_tag_input_traits
// {
//     template <typename CharT, typename FPack>
//     static inline detail::erroneous_printer<CharT> make_printer
//         ( const boost::stringify::v0::output_writer<CharT>& ow
//         , const FPack& ft
//         , error_tag x )
//     {
//         return {ft, x};
//     }
// };

} // namespace detail

//detail::error_tag_input_traits stringify_get_input_traits(error_tag x);

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, typename FPack>
inline ::detail::erroneous_printer<CharT>
stringify_make_printer
    ( boost::stringify::v0::output_writer<CharT>& ow
    , const FPack& ft
    , error_tag x )
{
    return {ow, ft, x};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif
