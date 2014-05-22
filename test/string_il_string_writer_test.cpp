#include <string>
#include <string.h>
#include <boost/detail/lightweight_test.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/string_ini_list.hpp>


#define BOOST_STRING_LITERAL(charT, str)                                \
  (::boost::is_same<charT, char32_t>::value ? reinterpret_cast<const charT*>(U ## str) : \
   ::boost::is_same<charT, char16_t>::value ? reinterpret_cast<const charT*>(u ## str) : \
   ::boost::is_same<charT, wchar_t>::value  ? reinterpret_cast<const charT*>(L ## str) : \
   /*else: */                                 reinterpret_cast<const charT*>(str)) 


template <typename charT, typename traits=std::char_traits<charT> >
void test_std_basic_string()
{
  std::basic_string<charT> output;
  std::basic_string<charT> input = BOOST_STRING_LITERAL(charT, "asdfghjl");
  typedef boost::basic_string_il<charT, traits> gen_string_il;

  output += gen_string_il{input};
  BOOST_TEST(output == input);
  output.clear();

  output += gen_string_il{input, input};
  BOOST_TEST(output == input + input);
  output.clear();

  output += gen_string_il{input, input.c_str(), input};
  BOOST_TEST(output == input + input + input);
  output.clear();
}


void test_char_ptr()
{
  std::string output;
  char   non_const_charT_arr[] = "hello";
  char * non_const_charT_ptr   = non_const_charT_arr;
  const char* const_charT_ptr  = "blablabla";

  output += boost::string_il({non_const_charT_arr});
  BOOST_TEST(output == non_const_charT_arr);

  output.clear();
  output += boost::string_il{non_const_charT_ptr};
  BOOST_TEST(output == non_const_charT_ptr);


  output.clear();
  output += boost::string_il{const_charT_ptr};
  BOOST_TEST(output == const_charT_ptr);


  output.clear();
  output += boost::string_il
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

  return  boost::report_errors();
}
