#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>


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


template <typename charT>
struct weird_char_traits : public std::char_traits<charT>
{
  static charT*
  copy(charT* to, const charT* from, std::size_t n)
  {
    charT* it = to;
    while(n--)
    {
        assign(*it++, *from++);
    }
    return to;
  }

  static void
  assign(charT& c1, const charT& c2)
  {
      if (c2 == charT())
      {
          c1 = c2;
      }
      else
      {
          c1 = c2 | ( 1 << (sizeof(charT) * 8 - 1));
      }
  }

  static charT*
  assign(charT* dest, std::size_t n, charT a)
  {
      charT b;
      assign(b, a);
      std::fill_n(dest, n, b);
      return dest;
  }
};


template <typename charT> charT getChar(const charT*);

template
    < int LINE_NUM
    , typename charT
    , typename charTraits  
    , typename ... Formaters
    , typename Arg
    >
void test_with_traits
    ( const charT* _expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const Arg& arg
    )
{
    typedef
        boost::stringify::formater_tuple<Formaters ...>
        Fmt;

    typedef
        boost::stringify::input_base_ref<charT, charTraits, Fmt>
        str_arg;
    
    charT expected[200];
    charT resulted[200];
    std::size_t expected_len = charTraits::length(_expected);
    charTraits::copy(expected, _expected, expected_len);
  
    std::size_t resulted_length = str_arg(arg).length(fmt);
    BOOST_TEST(expected_len == resulted_length);
    
    charT* end = str_arg(arg).write(resulted, fmt);
    BOOST_TEST(expected_len == static_cast<std::size_t>(end - resulted));
    
    int string_comparation = charTraits::compare(expected, resulted, expected_len);
    BOOST_TEST(0 == string_comparation);
}


using namespace boost::stringify;

template <typename charT, typename Formating>
struct input_arg_dual_chartraits
{
    template <typename T, typename traits>
    using input_writer = decltype(argf<charT, traits, Formating>(std::declval<const T>()));

    template <typename T>
    using input_writer_std_traits = input_writer<T, std::char_traits<charT> >;

    template <typename T>
    using input_writer_weird_traits = input_writer<T, weird_char_traits<charT> >;
  
    template <typename T>
    input_arg_dual_chartraits
        ( const T& value
        , input_writer_std_traits<T>&&   writer_std = input_writer_std_traits<T>()
        , input_writer_weird_traits<T>&& writer_wrd = input_writer_weird_traits<T>()
        ) noexcept
        : m_writer_std(writer_std)
        , m_writer_wrd(writer_wrd)
    {
        writer_std.set(value);
        writer_wrd.set(value);
    }

    template <typename T, typename ExtraArg>
    input_arg_dual_chartraits
        ( const T& value
        , ExtraArg && arg  
        , input_writer_std_traits<T>&&   writer_std = input_writer_std_traits<T>()
        , input_writer_weird_traits<T>&& writer_wrd = input_writer_weird_traits<T>()
        ) noexcept
        : m_writer_std(writer_std)
        , m_writer_wrd(writer_wrd)
    {
        writer_std.set(value, arg);
        writer_wrd.set(value, arg);
    }

   template <typename T, typename ExtraArg1, typename ExtraArg2>
    input_arg_dual_chartraits
        ( const T& value
        , ExtraArg1 && arg1
        , ExtraArg2 && arg2
        , input_writer_std_traits<T>&&   writer_std = input_writer_std_traits<T>()
        , input_writer_weird_traits<T>&& writer_wrd = input_writer_weird_traits<T>()
        ) noexcept
        : m_writer_std(writer_std)
        , m_writer_wrd(writer_wrd)
    {
        writer_std.set(value, arg1, arg2);
        writer_wrd.set(value, arg1, arg2);
    }
    
    const boost::stringify::input_base<charT, Formating>& m_writer_std;
    const boost::stringify::input_base<charT, Formating>& m_writer_wrd;
};




template
    < int LINE_NUM
    , typename charT
    , typename charTraits  
    , typename Formating  
    >
void do_tests
    ( const charT* _expected
    , const Formating& fmt
    , const boost::stringify::input_base<charT, Formating>& arg_writer
    )
{
    charT expected[200];
    charT resulted[200];
    std::size_t expected_len = charTraits::length(_expected);
    charTraits::copy(expected, _expected, expected_len);
  
    std::size_t resulted_length = arg_writer.length(fmt);
    BOOST_TEST(expected_len == resulted_length);
    
    charT* end = arg_writer.write(resulted, fmt);
    BOOST_TEST(expected_len == static_cast<std::size_t>(end - resulted));
    
    int string_comparation = charTraits::compare(expected, resulted, expected_len);
    BOOST_TEST(0 == string_comparation);
}



template
    < int LINE_NUM
    , typename charT
    , typename ... Formaters
    >
void test
    ( const charT* expected
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const input_arg_dual_chartraits
        < typename std::decay<decltype(*expected)>::type
        , typename std::decay<decltype(fmt)>::type
        >& arg
    )
{
    do_tests<LINE_NUM, charT, std::char_traits<charT> >
        (expected, fmt, arg.m_writer_std);
    
    do_tests<LINE_NUM, charT, weird_char_traits<charT> >
        (expected, fmt, arg.m_writer_wrd);
}

#endif


