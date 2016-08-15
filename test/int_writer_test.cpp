#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"
#include <limits>
#include <locale>
#include <sstream>

template <typename output_type>
struct tester
{
  typedef test_utils::output_traits<output_type> output_traits;
  typedef typename output_traits::char_type charT;
  typedef typename output_traits::traits_type char_traits;

  static void test(const boost::stringify::str_writer<charT>& writer, const char* expected)
  {
    output_type out;
    out << writer;
    BOOST_TEST(test_utils::str8(out) == expected);
  }

  static void test()
  {
    using namespace boost::stringify;

    test(basic_argf<charT>(0), "0") ;

    test(basic_argf<charT>(123), "123");

    test(basic_argf<charT>(-123), "-123");

    char buff[500];
    sprintf(buff, "%d", std::numeric_limits<int32_t>::max());
    test(basic_argf<charT>(std::numeric_limits<int32_t>::max()), buff);

    sprintf(buff, "%d", std::numeric_limits<int32_t>::min());
    test(basic_argf<charT>(std::numeric_limits<int32_t>::min()), buff);
  }
};


void test_many_output_types()
{
  tester<char[200]>::test();
  tester<wchar_t[200]>::test();
  tester<char16_t[200]>::test();
  tester<char32_t[200]>::test();

  tester<std::basic_string<char> >::test();
  tester<std::basic_string<wchar_t> >::test();
  tester<std::basic_string<char16_t> >::test();
  tester<std::basic_string<char32_t> >::test();

  tester<std::basic_ostringstream<char> >::test();
  tester<std::basic_ostringstream<wchar_t> >::test();
  tester<std::basic_ostringstream<char16_t> >::test();
  tester<std::basic_ostringstream<char32_t> >::test();

  typedef test_utils::to_upper_char_traits<char> traits8;
  typedef test_utils::to_upper_char_traits<wchar_t>  wtraits ;
  typedef test_utils::to_upper_char_traits<char16_t> traits16;
  typedef test_utils::to_upper_char_traits<char32_t> traits32;

  tester<std::basic_string<char, traits8 > >::test();
  tester<std::basic_string<wchar_t,  wtraits> >::test();
  tester<std::basic_string<char16_t, traits16> >::test();
  tester<std::basic_string<char32_t, traits32> >::test();

  tester<std::basic_ostringstream<char, traits8 > >::test();
  tester<std::basic_ostringstream<wchar_t,  wtraits> >::test();
  tester<std::basic_ostringstream<char16_t, traits16> >::test();
  tester<std::basic_ostringstream<char32_t, traits32> >::test();
}


int main()
{
  test_many_output_types();

  return  boost::report_errors();
}






