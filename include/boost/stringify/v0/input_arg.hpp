#ifndef BOOST_STRINGIFY_V0_INPUT_ARG_HPP
#define BOOST_STRINGIFY_V0_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/formatter.hpp>
#include <type_traits>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail{

struct args_pair
{
    const void* first;
    const void* second;
};

template <typename Formatter, typename FTuple, typename InputType>
void formatter_init_ref(void* mem, const FTuple& ft)
{
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    new (mem) Formatter(ft, *first);
}

template <typename Formatter, typename FTuple, typename InputType>
void formatter_init_ref_2(void* mem, const FTuple& ft)
{
    using second_arg = typename Formatter::second_arg;
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    auto* second = reinterpret_cast<const second_arg*>(args->second);
    new (mem) Formatter(ft, *first, *second);
}

template <typename Formatter, typename FTuple, typename InputType>
void formatter_init_ptr(void* mem, const FTuple& ft)
{
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    new (mem) Formatter(ft, first);
}

template <typename Formatter, typename FTuple, typename InputType>
void formatter_init_ptr_2(void* mem, const FTuple& ft)
{
    using second_arg = typename Formatter::second_arg;
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    auto* second = reinterpret_cast<const second_arg*>(args->second);
    new (mem) Formatter(ft, first, *second);
}

inline void store_args(void* mem, const void* first, const void* second = nullptr)
{
    auto* args = reinterpret_cast<args_pair*>(mem);
    args->first = first;
    args->second = second;
}

template <class S>
struct formatter_storage
{
    union
    {
        typename std::aligned_storage<sizeof(S), alignof(S)>::type data;
        args_pair args;
    };
};

} // namespace detail


template <typename CharT, typename FTuple>
class input_arg
{
    template <class T>
    using trait = decltype(boost_stringify_input_traits_of(std::declval<const T>()));

    template <class T>
    using formatter_impl = typename trait<T>::template formatter<CharT, FTuple>;

    template <class S>
    using storage = detail::formatter_storage<S>;

    typedef void (*formatter_init_func)(void* mem, const FTuple& ft);

    struct private_type {};

public:

    template
        < typename T
        , typename S = formatter_impl<T>
        , typename = typename std::enable_if
            < !std::is_array<T>::value
           && !std::is_pointer<T>::value
            > ::type
        >
    input_arg
        ( const T& arg1
        , private_type = {}
        , storage<S> && st = {}
        )
        : m_initializer(detail::formatter_init_ref<S, FTuple, T>)
        , m_formatter(reinterpret_cast<formatter<CharT>*>(&st.data))
    {
        detail::store_args(m_formatter, &arg1);
    }

    template <typename T, typename S = formatter_impl<T*>>
    input_arg
        ( const T* arg1
        , private_type = {}
        , storage<S> && st = {}
        )
        : m_initializer(detail::formatter_init_ptr<S, FTuple, T>)
        , m_formatter(reinterpret_cast<formatter<CharT>*>(&st.data))
    {
        detail::store_args(m_formatter, arg1);
    }

    template
        < typename T
        , typename S = formatter_impl<T>
        , typename = typename std::enable_if
            < ! std::is_array<T>::value
           && ! std::is_pointer<T>::value
            > ::type
        >
    input_arg
        ( const T& arg1
        , const typename S::second_arg& arg2
        , private_type = {}
        , storage<S> && st = {}
        )
        : m_initializer(detail::formatter_init_ref_2<S, FTuple, T>)
        , m_formatter(reinterpret_cast<formatter<CharT>*>(&st.data))
    {
        detail::store_args(m_formatter, &arg1, &arg2);
    }

    template <typename T, typename S = formatter_impl<T*>>
    input_arg
        ( const T* arg1
        , const typename S::second_arg& arg2
        , private_type = {}
        , storage<S> && st = {}
        )
        : m_initializer(detail::formatter_init_ptr_2<S, FTuple, T>)
        , m_formatter(reinterpret_cast<formatter<CharT>*>(&st.data))
    {
        detail::store_args(m_formatter, arg1, &arg2);
    }

    ~input_arg()
    {
        if(is_initialized())
        {
            m_formatter->~formatter<CharT>();
        }
    }

    std::size_t length(const FTuple& ft) const
    {
        init_if_necessary(ft);
        return m_formatter->length();
    }

    void write(stringify::v0::output_writer<CharT>& out, const FTuple& ft) const
    {
        init_if_necessary(ft);
        return m_formatter->write(out);
    }

    int remaining_width(int w, const FTuple& ft) const
    {
        init_if_necessary(ft);
        return m_formatter->remaining_width(w);
    }

private:

    bool is_initialized() const
    {
        return m_initializer == nullptr;
    }

    void init_if_necessary(const FTuple& ft) const
    {
        if(m_initializer != nullptr)
        {
            m_initializer(m_formatter, ft);
        }
        m_initializer = nullptr;
    }

    mutable formatter_init_func m_initializer;
    boost::stringify::v0::formatter<CharT>* m_formatter;

};

template <typename T>
constexpr auto fmt(const T& value)
{
    return decltype(boost_stringify_input_traits_of(value))::fmt(value);
}

template <typename T>
constexpr auto uphex(const T& value)
{
    return fmt(value).uphex();
}

template <typename T>
constexpr auto hex(const T& value)
{
    return fmt(value).hex();
}

template <typename T>
constexpr auto dec(const T& value)
{
    return fmt(value).dec();
}

template <typename T>
constexpr auto oct(const T& value)
{
    return fmt(value).oct();
}

template <typename T>
constexpr auto left(const T& value, int width)
{
    return fmt(value) < width;
}

template <typename T>
constexpr auto right(const T& value, int width)
{
    return fmt(value) > width;
}

template <typename T>
constexpr auto internal(const T& value, int width)
{
    return fmt(value) % width;
}

template <typename T>
constexpr auto center(const T& value, int width)
{
    return fmt(value) ^ width;
}

template <typename T>
constexpr auto left(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) < width;
}

template <typename T>
constexpr auto right(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) > width;
}

template <typename T>
constexpr auto internal(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) % width;
}

template <typename T>
constexpr auto center(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) ^ width;
}

template <typename T, typename I>
constexpr auto multi(const T& value, I count)
{
    return fmt(value).multi(count);
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_ARG_HPP */

