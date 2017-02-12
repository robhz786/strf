#ifndef BOOST_STRINGIFY_DETAIL_DEFERRED_STRINGIFIER_CONSTRUCTION_HPP
#define BOOST_STRINGIFY_DETAIL_DEFERRED_STRINGIFIER_CONSTRUCTION_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/stringifier.hpp>
#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {
namespace detail {

template <typename CharT, typename Output, typename FTuple>
class deferred_stringifier_construction
{
public:
    virtual ~deferred_stringifier_construction() {};

    virtual std::size_t length(const FTuple& fmt) = 0;
        
    virtual void write(Output& out, const FTuple& fmt) = 0;
};


template <typename StringifierImpl, typename CharT, typename Output, typename FTuple>
class deferred_stringifier_construction_impl
    : public boost::stringify::detail::deferred_stringifier_construction
        <CharT, Output, FTuple>
{
    using input_type = typename StringifierImpl::input_type;
    
    using input_type_is_pointer = std::is_pointer<input_type>;

    using has_arg_format_type
    = boost::stringify::detail::has_arg_format_type<StringifierImpl>;
    
    struct no_arg_format_type{};

    template <typename S, bool HasArgFormatType>
    struct arg_format_type_trait
    {
        using type = typename S::arg_format_type;
    };
    
    template <typename S>
    struct arg_format_type_trait<S, false>
    {
        using type = no_arg_format_type;
    };

    using arg_format_type
    = typename arg_format_type_trait
        <StringifierImpl, has_arg_format_type::value>
        ::type;

    using input_type_ptr
    = boost::stringify::detail::ternary_t
        < input_type_is_pointer::value
        , input_type
        , const input_type*
        >;
    
public:

    deferred_stringifier_construction_impl()
        : state(nothing_initialized)
    {
    }

    ~deferred_stringifier_construction_impl()
    {
        if(state == stringifier_initialized)
        {
            do_get() -> ~StringifierImpl();
        }
    }
    
    void set_args(const input_type& value)
    {
        BOOST_ASSERT(state != stringifier_initialized);
        set_value(value);
        m_formatter_ptr = nullptr;
        state = args_initialized;
    }

    void set_args(const input_type& value, const arg_format_type& formatter)
    {
        BOOST_ASSERT(state != stringifier_initialized);
        set_value(value);
        m_formatter_ptr = &formatter;
        state = args_initialized;
    }

    std::size_t length(const FTuple& fmt)
    {
        construct_if_necessary(fmt);
        return do_get()->length();
    }
        
    virtual void write(Output& out, const FTuple& fmt)
    {
        construct_if_necessary(fmt);
        do_get()->write(out);
    };

private:
    
    auto get(const FTuple& fmt) -> const StringifierImpl*
    {
        if(state != stringifier_initialized)
        {
            BOOST_ASSERT(state == args_initialized);
            construct(fmt);
            state = stringifier_initialized;
        }
        return do_get();
    }

    constexpr auto do_get() const -> const StringifierImpl*
    {
        return reinterpret_cast<const StringifierImpl*>(&space[0]);
    }

    void construct_if_necessary(const FTuple& fmt)
    {
        if(state != stringifier_initialized)
        {
            BOOST_ASSERT(state == args_initialized);
            construct(fmt, get_value(), m_formatter_ptr, has_arg_format_type());
        }
    }
    
    void construct
        ( const FTuple& fmt
        , const input_type& value
        , const arg_format_type* formatter_ptr
        , std::true_type
        )
    {
        if (formatter_ptr)
        {
            new (&space[0]) StringifierImpl(fmt, value, *formatter_ptr);
        }
        else
        {
            new (&space[0]) StringifierImpl(fmt, value);
        }
        state = stringifier_initialized;
    }

    void construct
        ( const FTuple& fmt
        , const input_type& value
        , const arg_format_type*
        , std::false_type
        )
    {
        new (&space[0]) StringifierImpl(fmt, value);
        state = stringifier_initialized;
    }

    union
    {
        struct alignas(alignof(StringifierImpl)) 
        {
            char space[sizeof(StringifierImpl)];
        };
        struct
        {
            input_type_ptr m_value_ptr;
            const arg_format_type* m_formatter_ptr;
        };
    };

    
    enum
    {
        nothing_initialized,
        args_initialized,
        stringifier_initialized
    } state;

    void set_value(const input_type& v)
    {
        set_value(v, input_type_is_pointer());
    }

    void set_value(const input_type& v, std::false_type)
    {
        m_value_ptr = & v;
    }

    void set_value(input_type v, std::true_type)
    {
        m_value_ptr = v;
    }
    
    constexpr decltype(auto) get_value() const
    {
        return get_value(input_type_is_pointer());
    }

    constexpr auto get_value(std::false_type) const -> const input_type&
    {
        return * m_value_ptr;
    }
    
    constexpr auto get_value(std::true_type) const -> input_type
    {
        return m_value_ptr;
    }

};


} //namespace detail
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_DETAIL_DEFERRED_STRINGIFIER_CONSTRUCTION_HPP

