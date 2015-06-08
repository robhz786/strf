#include "test_utils.hpp"
#include <string.h>
#include <boost/detail/lightweight_test.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/rose.hpp>

template <typename output_type>
void test_template_output_type()
{
  typedef typename test_utils::output_traits<output_type>::char_type charT;
  typedef typename test_utils::output_traits<output_type>::traits_type traits;

  typedef boost::rose::basic_listf<charT> gen_listf;

  std::basic_string<charT, traits> input = BOOST_GENERIC_STR(charT, "ASDFasdf");
  {
    output_type output;
    output << gen_listf{input};
    BOOST_TEST(test_utils::str(output) == input);
  }
  {
    output_type output;
    output << gen_listf{input.c_str(), input};
    BOOST_TEST(test_utils::str(output) == input + input);
  }
}

void test_many_output_types()
{
  test_template_output_type<char[200]> ();
  test_template_output_type<wchar_t[200]> ();
  test_template_output_type<char16_t[200]> ();
  test_template_output_type<char32_t[200]> ();

  test_template_output_type<std::basic_string<char> >();
  test_template_output_type<std::basic_string<wchar_t> >();
  test_template_output_type<std::basic_string<char16_t> >();
  test_template_output_type<std::basic_string<char32_t> >();

  test_template_output_type<std::basic_ostringstream<char> >();
  test_template_output_type<std::basic_ostringstream<wchar_t> >();
  test_template_output_type<std::basic_ostringstream<char16_t> >();
  test_template_output_type<std::basic_ostringstream<char32_t> >();


  typedef test_utils::to_upper_char_traits<char> traits8;
  typedef test_utils::to_upper_char_traits<wchar_t> wtraits;
  typedef test_utils::to_upper_char_traits<char16_t> traits16;
  typedef test_utils::to_upper_char_traits<char32_t> traits32;

  test_template_output_type<std::basic_string<char, traits8> >();
  test_template_output_type<std::basic_string<wchar_t, wtraits> >();
  test_template_output_type<std::basic_string<char16_t, traits16> >();
  test_template_output_type<std::basic_string<char32_t, traits32> >();

  test_template_output_type<std::basic_ostringstream<char, traits8> >();
  test_template_output_type<std::basic_ostringstream<wchar_t, wtraits> >();
  test_template_output_type<std::basic_ostringstream<char16_t, traits16> >();
  test_template_output_type<std::basic_ostringstream<char32_t, traits32> >();
}

int main()
{
  test_many_output_types();

  return  boost::report_errors();
}





