#ifndef BOOST_STRINGIFY_V0_EXPECTED_HPP
#define BOOST_STRINGIFY_V0_EXPECTED_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


// A minimalist placeholder to std::expected

#include <type_traits>
#include <boost/assert.hpp>
#include <boost/stringify/v0/config.hpp>


BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <class E>
class unexpected {
public:

    unexpected() = delete;

    constexpr explicit unexpected(const E& val)
        : _val(val)
    {
    }

    constexpr explicit unexpected(E&& val)
        : _val(std::move(val))
    {
    }

    constexpr const E& value() const &
    {
        return _val;
    }

    constexpr E& value() &
    {
        return _val;
    }

    constexpr E&& value() &&
    {
        return std::move(_val);
    }

    constexpr E const&& value() const&&
    {
        return std::move(_val);
    }

private:

    E _val;
};

template <class E>
constexpr bool operator==(const unexpected<E>& x, const unexpected<E>& y)
{
    return x.value() == y.value();
}

template <class E>
constexpr bool operator!=(const unexpected<E>& x, const unexpected<E>& y)
{
    return x.value() != y.value();
}


template <class E>
class bad_expected_access;

template <>
class bad_expected_access<void> : public std::exception {
public:

    explicit bad_expected_access() = default;
};


template <class E>
class bad_expected_access: public bad_expected_access<void>
{
public:

    explicit bad_expected_access(E val)
        : _val(val)
    {
    }

    virtual const char* what() const noexcept override
    {
        return "bad_expected_access";
    }

    E& error() &
    {
        return _val;
    }

    const E& error() const&
    {
        return _val;
    }
    E&& error() &&
    {
        return std::move(_val);
    }

    const E&&  error() const&&
    {
        return std::move(_val);
    }

private:

    E _val;
};


struct unexpect_t {};

constexpr unexpect_t unexpect{};

template <typename ValueType, typename ErrorType> class expected;


template <typename ErrorType> class expected<void, ErrorType>
{
public:

    using error_type = ErrorType;
    using value_type = void;
    using unexpected_type = boost::stringify::v0::unexpected<ErrorType>;

    constexpr expected() : _has_value(true)
    {
    }

    constexpr expected(const expected& other)
        : _has_value(other._has_value)
    {
        if (other._has_value)
        {
            init_error(assume_error());
        }
    }

    constexpr expected(expected&& other)
        : _has_value(other._has_value)
    {
        if (!other._has_value)
        {
            init_error(std::move(assume_error()));
        }
    }

    constexpr expected(const unexpected_type& unex)
        : _has_value(false)
    {
        init_error(unex.value());
    }

    constexpr expected(unexpected_type&& unex)
        : _has_value(false)
    {
        init_error(std::move(unex.value()));
    }

    template <typename ... Args>
    constexpr expected(boost::stringify::v0::unexpect_t, Args&& ... args)
        : _has_value(true)
    {
        init_error(std::forward<Args>(args)...);
    }

    ~expected()
    {
        destroy_data();
    }

    constexpr expected& operator=(const expected& other)
    {
        if(other._has_value)
        {
            destroy_data();
        }
        else // other has error
        {
            if(this->_has_value)
            {
                emplace_error(other.assume_error());
            }
            else
            {
                assume_error() = other.assume_error();
            }
        }
        return *this;
    }

    constexpr expected& operator=(expected&& other)
    {
        if(other._has_value)
        {
            destroy_data();
        }
        else // other has error
        {
            if(this->_has_value)
            {
                emplace_error(std::move(other.assume_error()));
            }
            else
            {
                assume_error() = std::move(other.assume_error());
            }
        }
        return *this;
    }

    constexpr expected& operator=(const unexpected_type& other)
    {
        if(this->_has_value)
        {
            emplace_error(other.value());
        }
        else
        {
            assume_error() = other.value();
        }
        return *this;
    }

    constexpr expected& operator=(unexpected_type&& other)
    {
        if(this->_has_value)
        {
            emplace_error(std::move(other.value()));
        }
        else
        {
            assume_error() = std::move(other.value());
        }
        return *this;
    }

    constexpr bool operator==(const expected& other) const
    {
        return _has_value == other._has_value
            && (_has_value || assume_error() == other.assume_error());
    }

    constexpr explicit operator bool() const
    {
        return _has_value;
    }
    constexpr bool operator!() const
    {
        return ! _has_value;
    }
    constexpr bool has_value() const noexcept
    {
        return _has_value;
    }
    constexpr bool has_error() const noexcept
    {
        return ! _has_value;
    }

    constexpr void value() const
    {
    }

    constexpr const error_type& error() const &
    {
        BOOST_ASSERT(!_has_value);
        return assume_error();
    }
    constexpr error_type& error() &
    {
        BOOST_ASSERT(!_has_value);
        return assume_error();
    }
    constexpr const error_type&& error() const &&
    {
        BOOST_ASSERT(!_has_value);
        return std::move(assume_error());
    }
    constexpr error_type&& error() &&
    {
        BOOST_ASSERT(!_has_value);
        return std::move(assume_error());
    }
private:

    constexpr static std::size_t storage_align = alignof(error_type);
    constexpr static std::size_t storage_size = sizeof(error_type);

    using storage_type
    = typename std::aligned_storage<storage_size, storage_align>::type;

    storage_type _storage;
    bool _has_value=false;

    template <typename ... Args>
    constexpr void emplace_error(Args&& ... args)
    {
        destroy_data();
        init_error(std::forward<Args>(args)...);
    }

     constexpr error_type& assume_error() noexcept
    {
        auto *unex = reinterpret_cast<unexpected_type*>(&_storage);
        return unex->value();
    }

    constexpr const error_type& assume_error() const noexcept
    {
        auto *unex = reinterpret_cast<const unexpected_type*>(&_storage);
        return unex->value();
    }

    template <typename ... Args>
    constexpr void init_error(Args &&... args)
    {
        new (&_storage) unexpected_type(std::forward<Args>(args)...);
        _has_value = false;
    }
    constexpr void destroy_error()
    {
        assume_error() . ~error_type();
    }
    constexpr void destroy_data()
    {
        if (! _has_value)
        {
            destroy_error();
        }
    }
};


template <typename ValueType, typename ErrorType>
class expected
{
public:

    using error_type = ErrorType;
    using value_type = ValueType;
    using unexpected_type = boost::stringify::v0::unexpected<ErrorType>;

    constexpr expected()
    {
        init_value();
    }

    constexpr expected(const expected& other)
        : _has_value(other._has_value)
    {
        if (other._has_value)
        {
            init_value(other.assume_value());
        }
        else
        {
            init_error(other.assume_error());
        }
    }

    constexpr expected(expected&& other)
        : _has_value(other._has_value)
    {
        if (other._has_value)
        {
            init_value(std::move(other.assume_value()));
        }
        else
        {
            init_error(std::move(other.assume_error()));
        }
    }

    constexpr expected(const unexpected_type& unex)
        : _has_value(false)
    {
        init_error(unex.value());
    }

    constexpr expected(unexpected_type&& unex)
        : _has_value(false)
    {
        init_error(std::move(unex.value()));
    }

    template <typename ... Args>
    constexpr expected(boost::stringify::v0::in_place_t, Args&& ... args)
        : _has_value(true)
    {
        init_value(std::forward<Args>(args)...);
    }

    template <typename ... Args>
    constexpr expected(boost::stringify::v0::unexpect_t, Args&& ... args)
        : _has_value(true)
    {
        init_error(std::forward<Args>(args)...);
    }

    ~expected()
    {
        destroy_data();
    }

    expected& operator=(const expected& other)
    {
        if(other._has_value)
        {
            if(this->_has_value)
            {
                assume_value() = other.assume_value();
            }
            else
            {
                destroy_error();
                emplace(other.assume_value());
            }
        }
        else // other has error
        {
            if(this->_has_value)
            {
                destroy_value();
                emplace_error(other.assume_error());
            }
            else
            {
                assume_error() = other.assume_error();
            }
        }
        return *this;
    }

    constexpr expected& operator=(expected&& other)
    {
        if(other._has_value)
        {
            if(this->_has_value)
            {
                assume_value() = std::move(other.assume_value());
            }
            else
            {
                destroy_error();
                emplace(std::move(other.assume_value()));
            }
        }
        else // other has error
        {
            if(this->_has_value)
            {
                destroy_value();
                emplace_error(std::move(other.assume_error()));
            }
            else
            {
                assume_error() = std::move(other.assume_error());
            }
        }
        return *this;
    }

    template <typename U>
    constexpr expected& operator=(U&& other)
    {
        if(this->_has_value)
        {
            assume_value() = std::forward<U>(other);
        }
        else
        {
            destroy_error();
            emplace_value(other);
        }
        return *this;
    }

    constexpr expected& operator=(const unexpected_type& other)
    {
        if(this->_has_value)
        {
            destroy_value();
            emplace_error(other.value());
        }
        else
        {
            assume_error() = other.value();
        }
        return *this;
    }

    constexpr expected& operator=(unexpected_type&& other)
    {
        if(this->_has_value)
        {
            destroy_value();
            emplace_error(std::move(other.value()));
        }
        else
        {
            assume_error() = std::move(other.value());
        }
        return *this;
    }

    constexpr bool operator==(const expected& other) const
    {
        if (_has_value)
        {
            return other._has_value && assume_value() == other.assume_value();
        }
        else
        {
            return ! other._has_value && assume_error() == other.assume_error();
        }
    }

    constexpr explicit operator bool() const
    {
        return has_value();
    }

    constexpr bool operator!() const
    {
        return ! has_value();
    }

    constexpr value_type& operator *() &
    {
        return assume_value();
    }

    constexpr const value_type& operator*() const &
    {
        return assume_value();
    }

    constexpr value_type&& operator *() &&
    {
        return std::move(assume_value());
    }

    constexpr const value_type&& operator *() const &&
    {
        return std::move(assume_value());
    }

    template <typename ... Args>
    constexpr void emplace(Args&& ... args)
    {
        destroy_data();
        init_value(std::forward<Args>(args)...);
    }

    constexpr bool has_value() const noexcept
    {
        return _has_value;
    }

    constexpr bool has_error() const noexcept
    {
        return ! _has_value;
    }


    constexpr value_type& value() &
    {
        if ( ! _has_value)
        {
            throw bad_expected_access<error_type>(assume_error());
        }
        return assume_value();
    }
    constexpr const value_type& value() const &
    {
        if ( ! _has_value)
        {
            throw bad_expected_access<error_type>(assume_error());
        }
        return assume_value();
    }
    constexpr value_type&& value() &&
    {
        if ( ! _has_value)
        {
            throw bad_expected_access<error_type>(assume_error());
        }
        return std::move(assume_value());
    }
    constexpr const value_type&& value() const &&
    {
        if ( ! _has_value)
        {
            throw bad_expected_access<error_type>(assume_error());
        }
        return std::move(assume_value());
    }


    constexpr const error_type& error() const &
    {
        BOOST_ASSERT(!_has_value);
        return assume_error();
    }
    constexpr error_type& error() &
    {
        BOOST_ASSERT(!_has_value);
        return assume_error();
    }
    constexpr const error_type&& error() const &&
    {
        BOOST_ASSERT(!_has_value);
        return std::move(assume_error());
    }
    constexpr error_type&& error() &&
    {
        BOOST_ASSERT(!_has_value);
        return std::move(assume_error());
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

    storage_type _storage;
    bool _has_value=false;

    template <typename ... Args>
    constexpr void emplace_error(Args&& ... args)
    {
        destroy_data();
        init_error(std::forward<Args>(args)...);
    }

    constexpr value_type& assume_value() noexcept
    {
        auto * value = reinterpret_cast<value_type*>(&_storage);
        return * value;
    }

    constexpr error_type& assume_error() noexcept
    {
        auto *unex = reinterpret_cast<unexpected_type*>(&_storage);
        return unex->value();
    }

    constexpr const value_type& assume_value() const noexcept
    {
        const auto * value = reinterpret_cast<const value_type*>(&_storage);
        return *value;
    }

    constexpr const error_type& assume_error() const noexcept
    {
        auto *unex = reinterpret_cast<const unexpected_type*>(&_storage);
        return unex->value();
    }

    template <typename ... Args>
    constexpr void init_error(Args &&... args)
    {
        new (&_storage) unexpected_type(std::forward<Args>(args)...);
        _has_value = false;
    }

    template <typename ... Args>
    constexpr void init_value(Args &&... args)
    {
        new (&_storage) value_type(std::forward<Args>(args)...);
        _has_value = true;
    }

    constexpr void destroy_error()
    {
        assume_error() . ~error_type();
    }

    constexpr void destroy_value()
    {
        assume_value() .~value_type();
    }

    constexpr void destroy_data()
    {
        if (_has_value)
        {
            destroy_value();
        }
        else
        {
            destroy_error();
        }
    }
};

#define BOOST_STRINGIFY_RETURN_ON_ERROR(x)  if ( ! x) \
    { return {boost::stringify::v0::unexpect_t{}, x.error()}; }

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_EXPECTED_HPP

