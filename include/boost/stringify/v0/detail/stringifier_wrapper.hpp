#ifndef BOOST_STRINGIFY_V0_DETAIL_STRINGIFIER_WRAPPER_HPP
#define BOOST_STRINGIFY_V0_DETAIL_STRINGIFIER_WRAPPER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/mp_if.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename CharT, typename FTuple>
class stringifier_wrapper
{
public:

    using writer_type = boost::stringify::v0::output_writer<CharT>;
    
    virtual ~stringifier_wrapper()
    {
    }

    virtual std::size_t length(const FTuple& fmt) = 0;
        
    virtual void write(writer_type& out, const FTuple& fmt) = 0;

    virtual int remaining_width(int w, const FTuple& fmt) = 0;
};

template <typename StringifierImpl, typename FTuple>
class stringifier_wrapper_impl
    : public boost::stringify::v0::detail::stringifier_wrapper
        < typename StringifierImpl::char_type
        , FTuple
        >
{
    using char_type   = typename StringifierImpl::char_type;
    using input_type  = typename StringifierImpl::input_type;
    using ftuple_type = FTuple;
    using writer_type = boost::stringify::v0::output_writer<char_type>;
    using input_type_is_pointer = std::is_pointer<input_type>;

    template <typename Stringifier, typename = typename Stringifier::second_arg>
    static std::true_type has_second_arg_helper(const Stringifier*);

    template <typename Stringifier>
    static std::false_type has_second_arg_helper(...);

    template <typename Stringifier>
    using has_second_arg
		= decltype(has_second_arg_helper<Stringifier>((Stringifier*)0));
	
    
    struct no_second_arg{};

    template <typename S, bool HasArgFormatType>
    struct second_arg_trait
    {
        using type = typename S::second_arg;
    };
    
    template <typename S>
    struct second_arg_trait<S, false>
    {
        using type = no_second_arg;
    };

    using second_arg
    = typename second_arg_trait
        < StringifierImpl
        , has_second_arg<StringifierImpl>::value
        >
        ::type;

    using input_type_ptr = boost::stringify::v0::detail::mp_if
        < input_type_is_pointer::value
        , input_type
        , const input_type*
        >;

    using constructible_from_single_arg
    = std::is_constructible<StringifierImpl, const ftuple_type&, input_type>;
    
public:

    stringifier_wrapper_impl()
        : state(nothing_initialized)
    {
    }

    ~stringifier_wrapper_impl()
    {
        if(state == stringifier_initialized)
        {
            do_get() -> ~StringifierImpl();
        }
    }
    
    void set_args(const input_type& value)
    {
        static_assert(constructible_from_single_arg::value, "second argument needed");
        BOOST_ASSERT(state != stringifier_initialized);
        set_value(value);
        m_formatter_ptr = nullptr;
        state = args_initialized;
    }

    void set_args(const input_type& value, const second_arg& formatter)
    {
        BOOST_ASSERT(state != stringifier_initialized);
        set_value(value);
        m_formatter_ptr = &formatter;
        state = args_initialized;
    }

    std::size_t length(const ftuple_type& fmt) override
    {
        construct_if_necessary(fmt);
        return do_get()->length();
    }
        
    virtual void write(writer_type& out, const ftuple_type& fmt) override
    {
        construct_if_necessary(fmt);
        do_get()->write(out);
    };

    
    int remaining_width(int w, const ftuple_type& fmt) override
    {
        construct_if_necessary(fmt);
        return do_get()->remaining_width(w);
    }
    
private:

    constexpr auto do_get() const -> const StringifierImpl*
    {
        return reinterpret_cast<const StringifierImpl*>(&space[0]);
    }

    void construct_if_necessary(const ftuple_type& fmt)
    {
        if(state != stringifier_initialized)
        {
            BOOST_ASSERT(state == args_initialized);
            construct<StringifierImpl>
                ( fmt
                , get_value()
                , m_formatter_ptr
                , has_second_arg<StringifierImpl>()
                , constructible_from_single_arg()
                );
        }
    }

    template <typename S>
    void construct
        ( const ftuple_type& fmt
        , const input_type& value
        , const second_arg* formatter_ptr
        , std::true_type
        , std::true_type  
        )
    {
        if (formatter_ptr)
        {
            new (&space[0]) S(fmt, value, *formatter_ptr);
        }
        else
        {
            new (&space[0]) S(fmt, value);
        }
        state = stringifier_initialized;
    }

    template <typename S>
    void construct
        ( const ftuple_type& fmt
        , const input_type& value
        , const second_arg* formatter_ptr
        , std::true_type
        , std::false_type  
        )
    {
        BOOST_ASSERT(formatter_ptr);
        new (&space[0]) S(fmt, value, *formatter_ptr);
        state = stringifier_initialized;
    }


    
    template <typename S>
    void construct
        ( const ftuple_type& fmt
        , const input_type& value
        , const second_arg*
        , std::false_type
        , std::true_type  
        )
    {
        new (&space[0]) S(fmt, value);
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
            const second_arg* m_formatter_ptr;
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


} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_STRINGIFIER_WRAPPER_HPP

