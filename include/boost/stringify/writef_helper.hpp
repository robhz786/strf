#ifndef BOOST_STRINGIFY_WRITEF_HELPER_CPP
#define BOOST_STRINGIFY_WRITEF_HELPER_CPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/input_arg.hpp>

namespace boost {
namespace stringify {
namespace detail {

template
    < typename CharT
    , typename Formatting
    , typename output_type  
    >
class final_writer
{
    typedef boost::stringify::input_arg<CharT, output_type, Formatting> arg_type;

    static constexpr bool noexcept_output
    = boost::stringify::stringifier<CharT, output_type, Formatting>::noexcept_output;
    
    template <typename Arg1, typename ... Args>
    std::size_t length(Arg1 && arg1, Args && ... args) noexcept
    {
        return arg1.writer.length(m_fmt) + length(args...);
    }

    std::size_t length() noexcept
    {
        return 0;
    }
 
    template <typename output_type2, typename ... Args>
    auto reserve(output_type2& output, Args && ... args) noexcept
    -> decltype(output.reserve(std::size_t()), void())
    {
        output.reserve(1 + length(args...));
    }

    template <typename output_type2, typename ... Args>
    void reserve(output_type2&, ...) noexcept
    {
    }

public:
    
    template <typename ... Formaters>
    final_writer(output_type&& output, const Formaters& ... fmtargs)
        : m_output(std::move(output))
        , m_fmt(fmtargs ...)
    {
    }

    final_writer(final_writer&& other) = default;
    
    decltype(auto) operator[](const std::initializer_list<arg_type>& lst)
        && noexcept(noexcept_output)
    {
        for(auto arg : lst)
        {
            arg.writer.write(m_output, m_fmt);
        }
        return m_output.finish();
    }
    
    decltype(auto) operator()() && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output);
        return m_output.finish();
    }

    decltype(auto) operator()(const arg_type& a1) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1);
        a1.writer.write(m_output, m_fmt);
        return m_output.finish();
    }
  
    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3, a4);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        a4.writer.write(m_output, m_fmt);
        return m_output.finish();

    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        a4.writer.write(m_output, m_fmt);
        a5.writer.write(m_output, m_fmt);
        return m_output.finish();
    }

    decltype(auto) operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        a4.writer.write(m_output, m_fmt);
        a5.writer.write(m_output, m_fmt);
        a6.writer.write(m_output, m_fmt);
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
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        a4.writer.write(m_output, m_fmt);
        a5.writer.write(m_output, m_fmt);
        a6.writer.write(m_output, m_fmt);
        a7.writer.write(m_output, m_fmt);
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
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7, a8);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        a4.writer.write(m_output, m_fmt);
        a5.writer.write(m_output, m_fmt);
        a6.writer.write(m_output, m_fmt);
        a7.writer.write(m_output, m_fmt);
        a8.writer.write(m_output, m_fmt);
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
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7, a8, a9);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        a4.writer.write(m_output, m_fmt);
        a5.writer.write(m_output, m_fmt);
        a6.writer.write(m_output, m_fmt);
        a7.writer.write(m_output, m_fmt);
        a8.writer.write(m_output, m_fmt);
        a9.writer.write(m_output, m_fmt);
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
        ) && noexcept(noexcept_output)
    {
        reserve<output_type>(m_output, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
        a1.writer.write(m_output, m_fmt);
        a2.writer.write(m_output, m_fmt);
        a3.writer.write(m_output, m_fmt);
        a4.writer.write(m_output, m_fmt);
        a5.writer.write(m_output, m_fmt);
        a6.writer.write(m_output, m_fmt);
        a7.writer.write(m_output, m_fmt);
        a8.writer.write(m_output, m_fmt);
        a9.writer.write(m_output, m_fmt);
        a10.writer.write(m_output, m_fmt);        
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

