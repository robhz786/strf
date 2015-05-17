#include <string>
#include <string.h>
#include <boost/detail/lightweight_test.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/rose.hpp>

template <typename charT, typename traits=std::char_traits<charT> >
void test_std_basic_string()
{
  std::basic_string<charT> output;
  std::basic_string<charT> input = BOOST_STRING_LITERAL(charT, "asdfghjl");
  typedef boost::rose::basic_listf<charT, traits> gen_listf;

  output << gen_listf{input};
  BOOST_TEST(output == input);
  output.clear();

  output << gen_listf{input, input};
  BOOST_TEST(output == input + input);
  output.clear();

  output << gen_listf{input, input.c_str(), input};
  BOOST_TEST(output == input + input + input);
  output.clear();
}


void test_char_ptr()
{

  std::string output;
  char   non_const_charT_arr[] = "hello";
  char * non_const_charT_ptr   = non_const_charT_arr;
  const char* const_charT_ptr  = "blablabla";

  output << boost::rose::listf({non_const_charT_arr});
  BOOST_TEST(output == non_const_charT_arr);

  output.clear();
  output << boost::rose::listf{non_const_charT_ptr};
  BOOST_TEST(output == non_const_charT_ptr);


  output.clear();
  output << boost::rose::listf{const_charT_ptr};
  BOOST_TEST(output == const_charT_ptr);


  output.clear();
  output << boost::rose::listf
  {  
     non_const_charT_arr,
     const_charT_ptr,
     non_const_charT_ptr
  };
  BOOST_TEST(output == (std::string(non_const_charT_arr) +
                        const_charT_ptr +
                        non_const_charT_ptr));

}

int main()
{
  test_std_basic_string<char>();
  test_std_basic_string<wchar_t>();
  test_std_basic_string<char16_t>();
  test_std_basic_string<char32_t>();
  // todo: test with some other char_traits

  test_char_ptr();

  //todo: try to simplify compile error message on invalid types:
  //std::string output;
  //output << boost::rose::listf{L"asdf"};


  return  boost::report_errors();
}
