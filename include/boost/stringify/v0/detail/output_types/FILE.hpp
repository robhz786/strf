#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT>
class ec_narrow_file_writer final: public stringify::v0::output_buffer<CharT>
{
public:
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;

private:
    CharT _buff[_buff_size];

public:

    using char_type = CharT;

    ec_narrow_file_writer(std::FILE* file, std::size_t* count)
        : output_buffer<CharT>{ _buff, _buff + _buff_size }
        , _file(file)
        , _count_ptr(count)
    {
    }

    ~ec_narrow_file_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    bool recycle() override;

    stringify::v0::nodiscard_error_code finish()
    {
        if (! this->has_error())
        {
            recycle();
        }
        return this->get_error();
    }

protected:

    void on_error() override;

private:

    std::FILE* _file;
    std::size_t _count = 0;
    std::size_t* _count_ptr = nullptr;
};


template <typename CharT>
bool ec_narrow_file_writer<CharT>::recycle()
{
    auto it = this->pos();
    BOOST_ASSERT(_buff <= it && it <= _buff + _buff_size);
    this->set_pos(_buff);

    std::size_t count = it - _buff;
    auto count_inc = std::fwrite(_buff, sizeof(CharT), count, _file);
    _count += count_inc;
    if (count_inc != count)
    {
        this->set_error(std::error_code{errno, std::generic_category()});
        return false;
    }
    return true;
}

template <typename CharT>
void ec_narrow_file_writer<CharT>::on_error()
{
    auto it = this->pos();
    BOOST_ASSERT(_buff <= it && it <= _buff + _buff_size);
    this->set_pos(_buff);

    std::size_t count = it - _buff;
    _count += std::fwrite(_buff, sizeof(CharT), count, _file);
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_narrow_file_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_narrow_file_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_narrow_file_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_narrow_file_writer<wchar_t>;

#endif

class ec_wide_file_writer final: public stringify::v0::output_buffer<wchar_t>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    wchar_t _buff[_buff_size];

public:

    using char_type = wchar_t;

    ec_wide_file_writer(std::FILE* file, std::size_t* count)
        : output_buffer<wchar_t>{ _buff, _buff + _buff_size }
        , _file(file)
        , _count_ptr(count)
    {
    }

    ~ec_wide_file_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    bool recycle() override;

    stringify::v0::nodiscard_error_code finish()
    {
        if( ! this->has_error() )
        {
            recycle();
        }
        return this->get_error();
    }

protected:

    void on_error() override;

private:

    std::FILE* _file;
    std::size_t _count = 0;
    std::size_t* _count_ptr = nullptr;
};

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE
bool ec_wide_file_writer::recycle()
{
    auto end = this->pos();
    BOOST_ASSERT(_buff <= end && end <= _buff + _buff_size);
    this->set_pos(_buff);

    for(auto it = _buff ; it != end; ++it, ++_count)
    {
        auto ret = std::fputwc(*it, _file);
        if(ret == WEOF)
        {
            this->set_error(std::error_code{errno, std::generic_category()});
            return false;
        }
    }
    return true;
}

BOOST_STRINGIFY_INLINE
void ec_wide_file_writer::on_error()
{
    auto end = this->pos();
    BOOST_ASSERT(_buff <= end && end <= _buff + _buff_size);
    this->set_pos(_buff);

    for(auto it = _buff ; it != end; ++it, ++_count)
    {
        if (std::fputwc(*it, _file) == WEOF)
        {
            break;
        }
    }
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

template <typename CharT = char>
inline auto ec_write(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = stringify::v0::detail::ec_narrow_file_writer<CharT>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer,  FILE*, std::size_t*>
        (destination, count);
}

inline auto ec_wwrite(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = boost::stringify::v0::detail::ec_wide_file_writer;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE*, std::size_t*>
        (destination, count);
}

#if ! defined(BOOST_NO_EXCEPTION)

namespace detail {

template <typename CharT>
class narrow_file_writer final: public stringify::v0::output_buffer<CharT>
{
public:
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;

private:
    CharT _buff[_buff_size];

public:

    using char_type = CharT;

    narrow_file_writer(std::FILE* file, std::size_t* count)
        : output_buffer<CharT>{ _buff, _buff + _buff_size }
        , _file(file)
        , _count_ptr(count)
    {
    }

    ~narrow_file_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    bool recycle() override;

    std::size_t finish();

protected:

    void on_error() override;

    std::FILE* _file;
    std::size_t _count = 0;
    std::size_t* _count_ptr = nullptr;
};


template <typename CharT>
bool narrow_file_writer<CharT>::recycle()
{
    auto it = this->pos();
    BOOST_ASSERT(_buff <= it && it <= _buff + _buff_size);
    this->set_pos(_buff);

    std::size_t count = it - _buff;
    auto count_inc = std::fwrite(_buff, sizeof(CharT), count, _file);
    _count += count_inc;
    if (count_inc < count)
    {
        this->set_error(std::error_code{errno, std::generic_category()});
        return false;
    }
    return true;
}


template <typename CharT>
inline std::size_t narrow_file_writer<CharT>::finish()
{
    if (this->has_error() || (this->size() != 0 && ! recycle()))
    {
        throw stringify::v0::stringify_error{this->get_error()};
    }
    return _count;
}

template <typename CharT>
void narrow_file_writer<CharT>::on_error()
{
    auto it = this->pos();
    BOOST_ASSERT(_buff <= it && it <= _buff + _buff_size);
    this->set_pos(_buff);

    std::size_t count = it - _buff;
    _count += std::fwrite(_buff, sizeof(CharT), count, _file);
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<wchar_t>;

#endif

class wide_file_writer final: public stringify::v0::output_buffer<wchar_t>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    wchar_t _buff[_buff_size];

public:

    using char_type = wchar_t;

    wide_file_writer(std::FILE* file, std::size_t* count)
        : output_buffer<wchar_t>{ _buff, _buff + _buff_size }
        , _file(file)
        , _count_ptr(count)
    {
    }

    ~wide_file_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    bool recycle() override;

    std::size_t finish()
    {
        if (this->has_error() || (this->size() != 0 && ! recycle()))
        {
            throw stringify::v0::stringify_error{this->get_error()};
        }
        return _count;
    }

private:

    void on_error() override;

    std::FILE* _file;
    std::size_t _count = 0;
    std::size_t* _count_ptr = nullptr;
};

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE
bool wide_file_writer::recycle()
{
    auto end = this->pos();
    BOOST_ASSERT(_buff <= end && end <= _buff + _buff_size);
    this->set_pos(_buff);

    for(auto it = _buff ; it != end; ++it, ++_count)
    {
        auto ret = std::fputwc(*it, _file);
        if(ret == WEOF)
        {
            this->set_error(std::error_code{errno, std::generic_category()});
            return false;
        }
    }
    return true;
}

BOOST_STRINGIFY_INLINE
void wide_file_writer::on_error()
{
    auto end = this->pos();
    BOOST_ASSERT(_buff <= end && end <= _buff + _buff_size);
    this->set_pos(_buff);

    for(auto it = _buff ; it != end; ++it, ++_count)
    {
        if (std::fputwc(*it, _file) == WEOF)
        {
            break;
        }
    }
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

template <typename CharT = char>
inline auto write(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = stringify::v0::detail::narrow_file_writer<CharT>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE*, std::size_t* >
        (destination, count);
}

inline auto wwrite(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE*, std::size_t* >
        (destination, count);
}

#endif // ! defined(BOOST_NO_EXCEPTION)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

