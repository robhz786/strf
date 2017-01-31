#ifndef BOOST_STRINGIFY_INPUT_ARG_HPP
#define BOOST_STRINGIFY_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/stringifier.hpp>

namespace boost {
namespace stringify {

template <class T>
struct input_traits;

namespace detail {

template <typename InputType>
struct failed_intput_traits
{
    template <typename C, typename O, typename F>
    struct stringifier
    {
        static_assert(sizeof(C) == 0,
                      "unable to find stringifier for this input type");
    };
};

} // namespace detail

template <typename CharT, typename Output, typename FTuple>
class input_arg
{
    template <typename T>
    static decltype(boost_stringify_input_traits_of(std::declval<const T>()))
    input_traits_of(const T&);

    template <typename T>
    static boost::stringify::detail::failed_intput_traits<T>
    input_traits_of(...);
    
    template <class T>
    using input_traits = decltype(input_traits_of<T>(std::declval<const T>()));

    template <class T>
    using stringifier = typename input_traits<T>        
        :: template stringifier<CharT, Output, FTuple>;

    template <typename T>
    struct alignas(alignof(T)) memory_space
    {
        char space[sizeof(T)];
    };

    using stringifier_base
    = boost::stringify::stringifier<CharT, Output, FTuple>;

    typedef void (*construtor_function)
    (stringifier_base*, const FTuple&, const void*, const void*);
    
    template <typename Child, typename InputType>
    static void construct__arg_is_ptr
        ( stringifier_base* bptr
        , const FTuple& fmt
        , const void* input_arg_v
        , const void*
        )
    {
        new (bptr) Child (fmt, static_cast<InputType>(input_arg_v));
    }

    template <typename Child, typename InputType>
    static void construct__arg_is_ref
        ( stringifier_base* bptr
        , const FTuple& fmt
        , const void* input_arg_v
        , const void*
        )
    {
        new (bptr) Child (fmt, *static_cast<const InputType*>(input_arg_v));
    }

    template <typename Child, typename InputType>
    static void construct__arg_is_ptr__with_fmt
        ( stringifier_base* bptr
        , const FTuple& fmt
        , const void* input_arg_v
        , const void* format_arg_v
        )
    {
        typedef typename Child::arg_format_type format_type;
        new (bptr) Child
            ( fmt
            , static_cast<InputType>(input_arg_v)
            , *static_cast<const format_type*>(format_arg_v)
            );
    }

    template <typename Child, typename InputType>
    static void construct__arg_is_ref__withf_fmt
        ( stringifier_base* bptr
        , const FTuple& fmt
        , const void* input_arg_v
        , const void* format_arg_v
        )
    {
        typedef typename Child::arg_format_type format_type;
        new (bptr) Child
            ( fmt
            , *static_cast<const InputType*>(input_arg_v)
            , *static_cast<const format_type*>(format_arg_v)
            );
    }

public:

    template <typename T>
    input_arg
        ( const T* value
        , memory_space<stringifier<const T*> > && ms
          =  memory_space<stringifier<const T*> >()
        ) noexcept
        : m_value(value)
        , m_arg_format(nullptr)
        , m_stringifier(reinterpret_cast<stringifier<const T*>*>(&ms))
        , m_construct_function
             (construct__arg_is_ptr<stringifier<const T*>, const T*>)
    {
    }
   
    template <typename T>
    input_arg
        ( const T* value
        , const typename stringifier<T>::arg_format_type& arg_format
        , memory_space<stringifier<const T*> > && ms
          =  memory_space<stringifier<const T*> >()
        ) noexcept
        : m_value(value)
        , m_arg_format(&arg_format)
        , m_stringifier(reinterpret_cast<stringifier<const T*>*>(&ms))
        , m_construct_function
             (construct__arg_is_ptr__with_fmt<stringifier<const T*>, const T*>)
    {
       
    }

    template <typename T>
    input_arg
        ( const T& value
        , memory_space<stringifier<T>>&& ms = memory_space<stringifier<T>>()
        ) noexcept
        : m_value(&value)
        , m_arg_format(nullptr)
        , m_stringifier(reinterpret_cast<stringifier<T>*>(&ms))
        , m_construct_function
             (construct__arg_is_ref<stringifier<T>, T>)
    {
        
    }

    template <typename T>
    input_arg
        ( const T& value
        , const typename stringifier<T>::arg_format_type& arg_format
        , memory_space<stringifier<T>>&& ms = memory_space<stringifier<T>>()
        ) noexcept
        : m_value(&value)
        , m_arg_format(&arg_format)
        , m_stringifier(reinterpret_cast<stringifier<T>*>(&ms))
        , m_construct_function
             (construct__arg_is_ref__withf_fmt<stringifier<T>, T>)
    {
        
    }

    ~input_arg()
    {
        m_stringifier->~stringifier_base();
    }

    std::size_t length(const FTuple& fmt) const
    {
        construct_if_necessary(fmt);
        return m_stringifier->length();
    }

    void write(Output& out, const FTuple& fmt) const
    {
        construct_if_necessary(fmt);
        return m_stringifier->write(out);
    }

private:
    
    void construct_if_necessary(const FTuple& fmt) const
    {
        if (! m_constructed)
        {
            m_construct_function(m_stringifier, fmt, m_value, m_arg_format);            
        }
        m_constructed = true;
    }
    
    const void* m_value;
    const void* m_arg_format;
    boost::stringify::stringifier<CharT, Output, FTuple>* m_stringifier;
    construtor_function m_construct_function;
    mutable bool m_constructed = false;
};


} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP */

