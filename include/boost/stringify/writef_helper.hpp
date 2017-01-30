#ifndef BOOST_STRINGIFY_WRITEF_HELPER_CPP
#define BOOST_STRINGIFY_WRITEF_HELPER_CPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/input_arg.hpp>

namespace boost {
namespace stringify {
namespace detail {

template <typename T>
struct has_reserve
{
private:

    template <class U>
    static auto test(U* u)
        -> decltype(u->reserve(std::size_t()), std::true_type());

    template <class U>
    static std::false_type test(...);

public:

    static constexpr bool value = decltype(test<T>((T*)0))::value;
    
};



template
    < typename CharT
    , typename Formatting
    , typename output_type  
    >
class final_writer
{
    using arg_type =  boost::stringify::input_arg<CharT, output_type, Formatting>;
  
    template <typename Arg1, typename ... Args>
    std::size_t length(Arg1 && arg1, Args && ... args)
    {
        return arg1.length(m_fmt) + length(args...);
    }

    std::size_t length()
    {
        return 0;
    }
 
    template <typename output_type2, typename ... Args>
    auto reserve(output_type2& output, Args && ... args)
    -> decltype(output.reserve(std::size_t()), void())
    {
        output.reserve(1 + length(args...));
    }

    
    template <typename output_type2, typename ... Args>
    std::enable_if_t<!boost::stringify::detail::has_reserve<output_type2>::value>
    reserve(output_type2&, Args && ... args)
    {
    }
    
    template <typename output_type2>
    auto write_inilist
        ( const std::initializer_list<arg_type>& lst
        , output_type2& output
        )
       -> decltype(output.reserve(std::size_t()), void())
    {
        std::size_t len = 0;
        for(auto arg : lst)
        {
            len += arg.length();
        }
        output.reserve(len + 1);
        for(auto arg : lst)
        {
            arg.write(output, m_fmt);
        }
    }

    template <typename output_type2>
    auto write_inilist(const std::initializer_list<arg_type>& lst, ...)
    {
        for(auto arg : lst)
        {
            arg.write(m_output, m_fmt);
        }
    }
    
    void write_args()
    {
    }
    
    template <typename Arg1, typename ... Args>
    void write_args(Arg1&& arg1, Args && ... args)
    {
        arg1.write(m_output, m_fmt);
        write_args(args ...);
    }
    
public:
    
    template <typename ... Formaters>
    final_writer(output_type&& output, const Formaters& ... fmtargs)
        : m_output(std::move(output))
        , m_fmt(fmtargs ...)
    {
    }

    final_writer(final_writer&& other) = default;
    
    decltype(auto) operator[](const std::initializer_list<arg_type>& lst) &&
    {
        write_inilist<output_type>(lst, m_output);
        return m_output.finish();
    }
    
    decltype(auto) operator()() &&
    {
        return m_output.finish();
    }

    decltype(auto) operator()(const arg_type& a1) &&
    {
        reserve<output_type>(m_output, a1);
        write_args(a1);
        return m_output.finish();
    }
  
    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        ) &&
    {
        reserve<output_type>(m_output, a1, a2);
        write_args(a1, a2);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3);
        write_args(a1, a2, a3);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3, a4);
        write_args(a1, a2, a3, a4);
        return m_output.finish();

    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5);
        write_args(a1, a2, a3, a4, a5);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6);
        write_args(a1, a2, a3, a4, a5, a6);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7);
        write_args(a1, a2, a3, a4, a5, a6, a7);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        , const arg_type& a8
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7, a8);
        write_args(a1, a2, a3, a4, a5, a6, a7, a8);      
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        , const arg_type& a8
        , const arg_type& a9
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7, a8, a9);
        write_args(a1, a2, a3, a4, a5, a6, a7, a8, a9);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        , const arg_type& a8
        , const arg_type& a9
        , const arg_type& a10
        ) &&
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
        write_args(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);        
        return m_output.finish();
    }

private:

    output_type m_output;
    Formatting m_fmt;
};

} //namescpace detail


template <typename output_type>
class writef_helper
{
public:
    
    typedef typename output_type::char_type char_type;
    
   
    template <typename ... Formaters>
        using final_writer_type
        = boost::stringify::detail::final_writer
            < char_type
            , boost::stringify::ftuple<Formaters...>
            , output_type 
            >;

    typedef 
        boost::stringify::input_arg
            < char_type
            , output_type
            , boost::stringify::ftuple<>
            >
        default_input_arg;
    
public:
    
    writef_helper() = delete;
    writef_helper(const writef_helper&) = delete;
    writef_helper& operator=(const writef_helper&) = delete;
    
    writef_helper(writef_helper&& x)
        : m_output(std::move(x.m_output))
    {
    }
    
    template
        < typename ... Args
        , typename = typename std::enable_if
            <std::is_constructible<output_type, Args...>::value>::type
        >
    writef_helper(Args&& ... args)
        : m_output(args ...)
    {
    }
    
    template <typename ... Formaters>
    final_writer_type<Formaters ...> operator() (const Formaters& ... formaters) &&
    {
        return final_writer_type<Formaters ...>(std::move(m_output), formaters ...);
    }

    template <typename ... Formaters>
    final_writer_type<Formaters ...> operator()
        (const boost::stringify::ftuple<Formaters...>& formaters) &&
    {
        return final_writer_type<Formaters ...>(std::move(m_output), formaters);
    }

    
    decltype(auto) operator[](std::initializer_list<default_input_arg> lst) &&
    {
        return final_writer_type<>(std::move(m_output))[lst];
    }
    
private:
    
    output_type m_output;
};


} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_WRITEF_HELPER_CPP

