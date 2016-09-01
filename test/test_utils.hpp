#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

#include <boost/detail/lightweight_test.hpp>
#include <sstream>
// #include <boost/stringify/detail/utf32_to_utf8.hpp>
// #include <boost/stringify/detail/utf16_to_utf8.hpp>
#include <boost/stringify.hpp>

namespace test_utils {
/*
template <typename output_type>
struct output_traits
{
  typedef typename output_type::traits_type traits_type;
  typedef typename traits_type::char_type char_type;
};


template <typename charT, int size>
struct output_traits<charT[size] >
{
  typedef charT char_type;
  typedef std::char_traits<charT> traits_type;
};


template <typename charT, typename traits>
std::basic_string<charT, traits>
str(const std::basic_ostringstream<charT, traits>& oss)
{
  return oss.str();
}

template <typename charT, typename traits>
std::basic_string<charT, traits>
str(const std::basic_string<charT, traits>& s)
{
  return s;
}

template <typename charT>
std::basic_string<charT, std::char_traits<charT> >
str(const charT* s)
{
  return s;
}

template <typename charT, int sizeof_char> struct utfx_to_utf8_tratis{};

template <typename charT> struct utfx_to_utf8_tratis<charT, 1>
{
  typedef boost::stringify::detail::char_ptr_writer<char> writer;
};

template <typename charT> struct utfx_to_utf8_tratis<charT, 2>
{
  typedef boost::stringify::detail::utf16_to_utf8<charT> writer;
};

template <typename charT> struct utfx_to_utf8_tratis<charT, 4>
{
  typedef boost::stringify::detail::utf32_to_utf8<charT> writer;
};

template <typename output_type>
std::string str8(const output_type& x)
{
  std::string result;
  typedef typename output_traits<output_type>::char_type charT;
  typedef typename utfx_to_utf8_tratis<charT, sizeof(charT)>::writer utfx_to_utf8;
  result << utfx_to_utf8(str(x).c_str());
  return result;
}
*/

template <typename charT>
struct to_upper_char_traits : public std::char_traits<charT>
{
  static charT*
  copy(charT* to, const charT* from, std::size_t n)
  {
    charT* it = to;
    while(n--)
      *it++ = std::toupper(*from++);
    return to;
  }

  static void
  assign(charT& c1, const charT& c2)
  {
    c1 = std::toupper(c2);
  }

  static charT*
  assign(charT* dest, std::size_t n, charT a)
  {
    std::fill_n(dest, n, std::toupper(a));
    return dest;
  }
};
}; //namespace test_utils




template
    < int LINE_NUM
    , typename charT
    , typename ... GlobalFormaters
    , typename ... Args  
    >
void test_impl
    ( const std::basic_string<charT>& expected
    , const boost::stringify::formater_tuple<GlobalFormaters ...>& fmt
    , const Args& ... args
    )
{
    BOOST_TEST(expected.length() == boost::stringify::lengthf<charT>(fmt, args...));

    BOOST_TEST(expected.length() == boost::stringify::lengthf_il<charT>(fmt, {args...}));
    
    charT char_arr_output[200];
    boost::stringify::writef(char_arr_output, fmt, args...);
    BOOST_TEST(expected == char_arr_output);

    boost::stringify::writef_il(char_arr_output, fmt, {args...});
    BOOST_TEST(expected == char_arr_output);
}

/*
template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    , typename Arg  
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , Arg&& arg
    )
{
    typedef
        const boost::stringify::input_base_ref
            < decltype(getChar(expected))
            , typename std::decay<decltype(fmt)>::type
            >
        A;
    test_impl(expected, fmt, A(arg));
}

template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    , typename Args  
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , Arg1 ... arg1
    , Args&& ... arg
    )
{
 
}
*/


template <typename charT> charT getChar(const charT*);

template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(getChar(expected))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    )
{
    test_impl<LINE_NUM, charT>(expected, fmt, arg1);
}

    
template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(getChar(expected))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    )
{
    test_impl<LINE_NUM, charT>(expected, fmt, arg1, arg2);
}


template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(getChar(expected))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    )
{
    test_impl<LINE_NUM, charT>(expected, fmt, arg1, arg2, arg3);
}


template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(getChar(expected))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    )
{
    test_impl<LINE_NUM, charT>(expected, fmt, arg1, arg2, arg3, arg4);
}


template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(getChar(expected))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    )
{
    test_impl<LINE_NUM, charT>(expected, fmt, arg1, arg2, arg3, arg4, arg5);
}



#endif


