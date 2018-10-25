#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/output_types/buffered_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT>
class narrow_file_writer final: public stringify::v0::buffered_writer<CharT>
{
public:
    constexpr static std::size_t buff_size = stringify::v0::min_buff_size;

private:
    CharT buff[buff_size];

public:

    using char_type = CharT;

    narrow_file_writer
        ( stringify::v0::output_writer_init<CharT> init
        , std::FILE* file
        , std::size_t* count
        )
        : stringify::v0::buffered_writer<CharT>{init, buff, buff_size}
        , m_file(file)
        , m_count(count)
    {
        if (m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    ~narrow_file_writer()
    {
        this->flush();
    }

protected:

    bool do_put(const CharT* str, std::size_t count) override
    {
        auto count_inc = std::fwrite(str, sizeof(char_type), count, m_file);

        if (m_count != nullptr)
        {
            *m_count += count_inc;
        }
        if (count != count_inc)
        {
            this->set_error(std::error_code{errno, std::generic_category()});
            return false;
        }
        return true;
    }

    std::FILE* m_file;
    std::size_t* m_count = nullptr;
};


class wide_file_writer final: public stringify::v0::buffered_writer<wchar_t>
{
    constexpr static std::size_t buff_size = stringify::v0::min_buff_size;
    wchar_t buff[buff_size];

public:

    using char_type = wchar_t;

    wide_file_writer
        ( stringify::v0::output_writer_init<wchar_t> init
        , std::FILE* file
        , std::size_t* count
        );

    ~wide_file_writer();

protected:

    bool do_put(const wchar_t* str, std::size_t count) override;

    std::FILE* m_file;
    std::size_t* m_count = nullptr;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<wchar_t>;

#endif

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE wide_file_writer::wide_file_writer
    ( stringify::v0::output_writer_init<wchar_t> init
    , std::FILE* file
    , std::size_t* count
    )
    : stringify::v0::buffered_writer<wchar_t>{init, buff, buff_size}
    , m_file(file)
    , m_count(count)
{
    if (m_count != nullptr)
    {
        *m_count = 0;
    }
}

BOOST_STRINGIFY_INLINE wide_file_writer::~wide_file_writer()
{
    this->flush();
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::do_put
    ( const wchar_t* str, std::size_t count )
{
    std::size_t i = 0;
    bool good = true;
    for( ; i < count && good; ++i, ++str)
    {
        auto ret = std::fputwc(*str, m_file);
        if(ret == WEOF)
        {
            this->set_error(std::error_code{errno, std::generic_category()});
            good = false;;
        }
    }
    if (m_count != nullptr)
    {
        *m_count += i;
    }
    return good;
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

template <typename CharT = char>
inline auto write(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = stringify::v0::detail::narrow_file_writer<CharT>;
    return stringify::v0::make_destination<writer>(destination, count);
}

inline auto wwrite(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return stringify::v0::make_destination<writer>(destination, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

