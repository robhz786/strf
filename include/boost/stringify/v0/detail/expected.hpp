#ifndef BOOST_STRINGIFY_V0_EXPECTED_HPP
#define BOOST_STRINGIFY_V0_EXPECTED_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail
{

// A minimalist placeholder to std::expected

template <typename ValueType, typename ErrorType>
class expected
{
public:

    using error_type = ErrorType;
    using value_type = ValueType;

    constexpr expected()
    {
        init_error();
    }

    constexpr expected(const expected& other)
        : m_has_value(other.m_has_value)
    {
        if (m_has_value)
        {
            init_value(*other.assume_value());
        }
        else
        {
            init_error(*other.assume_error());
        }
    }

    constexpr expected(expected&& other)
        : m_has_value(other.m_has_value)
    {
        if (m_has_value)
        {
            init_value(std::move(*other.assume_value()));
        }
        else
        {
            init_error(std::move(*other.assume_error()));
        }
    }

    constexpr expected(const value_type& val)
        : m_has_value(true)
    {
        init_value(val);
    }

    constexpr expected(const error_type& err)
        : m_has_value(false)
    {
        init_error(err);
    }

    constexpr expected(value_type&& val)
        : m_has_value(true)
    {
        init_value(std::move(val));
    }

    constexpr expected(error_type&& err)
        : m_has_value(false)
    {
        init_error(std::move(err));
    }

    ~expected()
    {
        destroy_data();
    }

    expected& operator=(const expected& other)
    {
        if(other.m_has_value)
        {
            if(this->m_has_value)
            {
                *assume_value() = *other.assume_value();
            }
            else
            {
                destroy_error();
                emplace_value(*other.assume_value());
            }
        }
        else // other has error
        {
            if(this->m_has_value)
            {
                destroy_value();
                emplace_error(*other.assume_error());
            }
            else
            {
                *assume_error() = *other.assume_error();
            }
        }
        return *this;
    }

    constexpr expected& operator=(expected&& other)
    {
        if(other.m_has_value)
        {
            if(this->m_has_value)
            {
                *assume_value() = std::move(*other.assume_value());
            }
            else
            {
                destroy_error();
                emplace_value(std::move(*other.assume_value()));
            }
        }
        else // other has error
        {
            if(this->m_has_value)
            {
                destroy_value();
                emplace_error(std::move(*other.assume_error()));
            }
            else
            {
                *assume_error() = std::move(*other.assume_error());
            }
        }
        return *this;
    }

    constexpr bool operator==(const expected& other) const
    {
        if (m_has_value)
        {
            return other.m_has_value && *assume_value() == *other.assume_value();
        }
        else
        {
            return ! other.m_has_value && *assume_error() == *other.assume_error();
        }
    }

    constexpr operator bool() const
    {
        return has_value();
    }

    constexpr bool operator!() const
    {
        return ! has_value();
    }

    constexpr value_type& operator *()
    {
        return value();
    }

    constexpr const value_type& operator*() const
    {
        return value();
    }

    template <typename ... Args>
    constexpr void emplace_error(Args&& ... args)
    {
        destroy_data();
        init_error(std::forward<Args>(args)...);
    }

    template <typename ... Args>
    constexpr void emplace_value(Args&& ... args)
    {
        destroy_data();
        init_value(std::forward<Args>(args)...);
    }

    constexpr bool has_value() const noexcept
    {
        return m_has_value;
    }

    constexpr bool has_error() const noexcept
    {
        return ! m_has_value;
    }

    constexpr value_type& value()
    {
        if ( ! m_has_value)
        {
            throw std::logic_error(assume_error()->message());
        }
        return *assume_value();
    }
    
    constexpr const value_type& value() const
    {
        if ( ! m_has_value)
        {
            throw std::logic_error(assume_error()->message());
        }
        return *assume_value();
    }

    constexpr const error_type& error() const
    {
        if (m_has_value)
        {
            throw std::logic_error("expected error");
        }
        return *assume_error();
    }

private:

    constexpr static std::size_t storage_align
        = alignof(value_type) > alignof(error_type)
        ? alignof(value_type)
        : alignof(error_type);

    constexpr static std::size_t storage_size
        = sizeof(value_type) > sizeof(error_type)
        ? sizeof(value_type)
        : sizeof(error_type);

    using storage_type
    = typename std::aligned_storage<storage_size, storage_align>::type;

    storage_type m_storage;
    bool m_has_value=false;

    constexpr value_type* assume_value() noexcept
    {
        return reinterpret_cast<value_type*>(&m_storage);
    }

    constexpr error_type* assume_error() noexcept
    {
        return reinterpret_cast<error_type*>(&m_storage);
    }

    constexpr const value_type* assume_value() const noexcept
    {
        return reinterpret_cast<const value_type*>(&m_storage);
    }

    constexpr const error_type* assume_error() const noexcept
    {
        return reinterpret_cast<const error_type*>(&m_storage);
    }

    template <typename ... Args>
    constexpr void init_error(Args &&... args)
    {
        new (&m_storage) error_type(std::forward<Args>(args)...);
    }

    template <typename ... Args>
    constexpr void init_value(Args &&... args)
    {
        new (&m_storage) value_type(std::forward<Args>(args)...);
    }

    constexpr void destroy_error()
    {
        assume_error()-> ~error_type();
    }

    constexpr void destroy_value()
    {
        assume_value()-> ~value_type();
    }

    constexpr void destroy_data()
    {
        if (m_has_value)
        {
            destroy_value();
        }
        else
        {
            destroy_error();
        }
    }
};

} // namespace detail
BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_EXPECTED_HPP

