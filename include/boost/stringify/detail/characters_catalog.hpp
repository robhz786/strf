#ifndef BOOST_STRINGIFY_DETAIL_CHARACTERS_CATALOG_HPP_INCLUDED
#define BOOST_STRINGIFY_DETAIL_CHARACTERS_CATALOG_HPP_INCLUDED


namespace boost
{
namespace stringify
{
namespace detail
{
  template<typename charT> constexpr charT the_digit_zero();
  template<> constexpr char     the_digit_zero<char>()    { return  '0'; }
  template<> constexpr wchar_t  the_digit_zero<wchar_t>() { return L'0'; }
  template<> constexpr char16_t the_digit_zero<char16_t>(){ return u'0'; }
  template<> constexpr char32_t the_digit_zero<char32_t>(){ return U'0'; }
 
  template<typename charT> constexpr charT the_character_x();
  template<> constexpr char     the_character_x<char>()    { return  'x'; }
  template<> constexpr wchar_t  the_character_x<wchar_t>() { return L'x'; }
  template<> constexpr char16_t the_character_x<char16_t>(){ return u'x'; }
  template<> constexpr char32_t the_character_x<char32_t>(){ return U'x'; }
 
  template<typename charT> constexpr charT the_character_X();
  template<> constexpr char     the_character_X<char>()    { return  'X'; }
  template<> constexpr wchar_t  the_character_X<wchar_t>() { return L'X'; }
  template<> constexpr char16_t the_character_X<char16_t>(){ return u'X'; }
  template<> constexpr char32_t the_character_X<char32_t>(){ return U'X'; }

  template<typename charT> constexpr charT the_character_a();
  template<> constexpr char     the_character_a<char>()    { return  'a'; }
  template<> constexpr wchar_t  the_character_a<wchar_t>() { return L'a'; }
  template<> constexpr char16_t the_character_a<char16_t>(){ return u'a'; }
  template<> constexpr char32_t the_character_a<char32_t>(){ return U'a'; }

  template<typename charT> constexpr charT the_character_A();
  template<> constexpr char     the_character_A<char>()    { return  'A'; }
  template<> constexpr wchar_t  the_character_A<wchar_t>() { return L'A'; }
  template<> constexpr char16_t the_character_A<char16_t>(){ return u'A'; }
  template<> constexpr char32_t the_character_A<char32_t>(){ return U'A'; }

  template<typename charT> constexpr charT the_space_character();
  template<> constexpr char     the_space_character<char>()    { return  ' '; }
  template<> constexpr wchar_t  the_space_character<wchar_t>() { return L' '; }
  template<> constexpr char16_t the_space_character<char16_t>(){ return u' '; }
  template<> constexpr char32_t the_space_character<char32_t>(){ return U' '; }

  template<typename charT> constexpr charT the_dot_character();
  template<> constexpr char     the_dot_character<char>()    { return  '.'; }
  template<> constexpr wchar_t  the_dot_character<wchar_t>() { return L'.'; }
  template<> constexpr char16_t the_dot_character<char16_t>(){ return u'.'; }
  template<> constexpr char32_t the_dot_character<char32_t>(){ return U'.'; }

  template<typename charT> constexpr charT the_comma_character();
  template<> constexpr char     the_comma_character<char>()    { return  ','; }
  template<> constexpr wchar_t  the_comma_character<wchar_t>() { return L','; }
  template<> constexpr char16_t the_comma_character<char16_t>(){ return u','; }
  template<> constexpr char32_t the_comma_character<char32_t>(){ return U','; }

  template<typename charT> constexpr charT the_sign_minus();
  template<> constexpr char     the_sign_minus<char>()    { return  '-'; }
  template<> constexpr wchar_t  the_sign_minus<wchar_t>() { return L'-'; }
  template<> constexpr char16_t the_sign_minus<char16_t>(){ return u'-'; }
  template<> constexpr char32_t the_sign_minus<char32_t>(){ return U'-'; }
 
  template<typename charT> constexpr charT the_sign_plus();
  template<> constexpr char     the_sign_plus<char>()    { return  '+'; }
  template<> constexpr wchar_t  the_sign_plus<wchar_t>() { return L'+'; }
  template<> constexpr char16_t the_sign_plus<char16_t>(){ return u'+'; }
  template<> constexpr char32_t the_sign_plus<char32_t>(){ return U'+'; }

}//namespace detail
}//namespace stringify
}//namespace boot



#endif
