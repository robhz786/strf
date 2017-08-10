#ifndef BOOST_STRINGIFY_V0_INPUT_ARG_HPP
#define BOOST_STRINGIFY_V0_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/stringifier.hpp>
#include <type_traits>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail{

struct args_pair
{
    const void* first;
    const void* second;
};

template <typename Stringifier, typename FTuple, typename InputType>
void stringify_init_ref(void* mem, const FTuple& ft)
{
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    new (mem) Stringifier(ft, *first);
}

template <typename Stringifier, typename FTuple, typename InputType>
void stringify_init_ref_2(void* mem, const FTuple& ft)
{
    using second_arg = typename Stringifier::second_arg;
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    auto* second = reinterpret_cast<const second_arg*>(args->second);
    new (mem) Stringifier(ft, *first, *second);
}

template <typename Stringifier, typename FTuple, typename InputType>
void stringify_init_ptr(void* mem, const FTuple& ft)
{
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    new (mem) Stringifier(ft, first);
}

template <typename Stringifier, typename FTuple, typename InputType>
void stringify_init_ptr_2(void* mem, const FTuple& ft)
{
    using second_arg = typename Stringifier::second_arg;
    auto* args = reinterpret_cast<args_pair*>(mem);
    auto* first = reinterpret_cast<const InputType*>(args->first);
    auto* second = reinterpret_cast<const second_arg*>(args->second);
    new (mem) Stringifier(ft, first, *second);
}

inline void store_args(void* mem, const void* first, const void* second = nullptr)
{
    auto* args = reinterpret_cast<args_pair*>(mem);
    args->first = first;
    args->second = second;
}

template <class S>
struct stringify_storage
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
    using stringifier_impl = typename trait<T>::template stringifier<CharT, FTuple>;

    template <class S>
    using storage = detail::stringify_storage<S>;

    typedef void (*stringify_init_func)(void* mem, const FTuple& ft);

public:

    template
        < typename T
        , typename S = stringifier_impl<T>
        , typename = std::enable_if_t<!std::is_array<T>::value>>
    input_arg(const T& arg1, storage<S> && st = storage<S>())
        : m_initializer(detail::stringify_init_ref<S, FTuple, T>)
        , m_stringifier(reinterpret_cast<stringifier<CharT>*>(&st.data))
    {
        detail::store_args(m_stringifier, &arg1);
    }

    template <typename T, typename S = stringifier_impl<T*>>
    input_arg(const T* arg1, storage<S> && st = storage<S>())
        : m_initializer(detail::stringify_init_ptr<S, FTuple, T>)
        , m_stringifier(reinterpret_cast<stringifier<CharT>*>(&st.data))
    {
        detail::store_args(m_stringifier, arg1);
    }

    template
        < typename T
        , typename S = stringifier_impl<T>
        , typename = std::enable_if_t<!std::is_array<T>::value>>
    input_arg
        ( const T& arg1
        , const typename S::second_arg& arg2
        , storage<S> && st = storage<S>()
        )
        : m_initializer(detail::stringify_init_ref_2<S, FTuple, T>)
        , m_stringifier(reinterpret_cast<stringifier<CharT>*>(&st.data))
    {
        detail::store_args(m_stringifier, &arg1, &arg2);
    }

    template <typename T, typename S = stringifier_impl<T*>>
    input_arg
        ( const T* arg1
        , const typename S::second_arg& arg2
        , storage<S> && st = storage<S>()
        )
        : m_initializer(detail::stringify_init_ptr_2<S, FTuple, T>)
        , m_stringifier(reinterpret_cast<stringifier<CharT>*>(&st.data))
    {
        detail::store_args(m_stringifier, arg1, &arg2);
    }

    ~input_arg()
    {
        if(is_initialized())
        {
            m_stringifier->~stringifier<CharT>();
        }
    }

    std::size_t length(const FTuple& ft) const
    {
        init_if_necessary(ft);
        return m_stringifier->length();
    }

    void write(boost::stringify::v0::output_writer<CharT>& out, const FTuple& ft) const
    {
        init_if_necessary(ft);
        return m_stringifier->write(out);
    }

    int remaining_width(int w, const FTuple& ft) const
    {
        init_if_necessary(ft);
        return m_stringifier->remaining_width(w);
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
            m_initializer(m_stringifier, ft);
        }
        m_initializer = nullptr;
    }

    mutable stringify_init_func m_initializer;
    boost::stringify::v0::stringifier<CharT>* m_stringifier;

};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_ARG_HPP */

