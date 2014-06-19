#ifndef BOOST_DETAIL_BASIC_STRING_IL_ELEMENT_HPP_INCLUDED
#define BOOST_DETAIL_BASIC_STRING_IL_ELEMENT_HPP_INCLUDED

#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/static_assert.hpp>
#include <boost/string_ini_list/string_il_string_writer.hpp>
#include <boost/string_ini_list/string_il_int_writer.hpp>
#include <boost/type_traits/is_base_of.hpp>

#ifndef BOOST_MAX_SIZEOF_STRING_IL_WRITER
#define BOOST_MAX_SIZEOF_STRING_IL_WRITER (4*sizeof(void*))
#endif

namespace boost
{
namespace detail
{

  template <typename charT, typename traits=std::char_traits<charT> >
  class basic_string_il_element
  {
    typedef string_il_writer_base<charT> writer_base;

  public:
    template <class T>
    basic_string_il_element(const T& argument)
    {
      init(argument);
    }
    template <class T>
    basic_string_il_element(T* argument)
    {
      init(static_cast<const T*>(argument));
    }
    std::size_t minimal_length() const
    {
      return get_writer().minimal_length();
    }
    charT* write_without_termination_char(charT* output) const
    {
      return get_writer().write_without_termination_char(output);
    }
    ~basic_string_il_element()
    {
      get_writer().~writer_base();
    }

  private:
    typedef void* data;  // using pointer in order to force proper alignment
    data pool[BOOST_MAX_SIZEOF_STRING_IL_WRITER / sizeof(data)];
    void* get_pool()
    {
      return reinterpret_cast<void*>(&pool[0]);
    }
    const void* get_pool() const
    {
      return reinterpret_cast<const void*>(&pool[0]);
    }
    const writer_base& get_writer() const
    {
      return *reinterpret_cast<const writer_base*>(get_pool());
    }
/*
    template <class T>
    void init(const T& argument)
    {
      typedef
        typename remove_cv<typename remove_reference<T>::type>::type
        adjusted_T;

      typedef
        string_il_writer<adjusted_T, charT, traits>
        writer_t;
     
      construct_writer<T, writer_t>(argument);
    }
*/

    template <class DERIVED_WRITER>
    struct is_aligned_with_base
    {
      static constexpr DERIVED_WRITER * some_addr = ((DERIVED_WRITER*)(0) + 1);
      static constexpr writer_base    * base_addr = some_addr;
      static constexpr bool value = (reinterpret_cast<void*>(some_addr) == 
                                     reinterpret_cast<void*>(base_addr)); 
    };

/*
    template <class T, class writer_t>
    void construct_writer(const T& argument)
    {
      
      BOOST_STATIC_ASSERT(sizeof(writer_t) <= BOOST_MAX_SIZEOF_STRING_IL_WRITER);
      BOOST_STATIC_ASSERT(is_aligned_with_base<writer_t>::value);

      new (get_pool()) writer_t(argument);

    }
*/

    template <class T>
    void init(const T& arg)
    {
      construct_writer(arg, string_ini_list_argument_traits(arg));
    }

    template <class T, class arg_traits>
    void construct_writer(const T& arg, arg_traits)
    {
      typedef typename arg_traits::template writer<charT, traits> arg_writer_type;
      verify_arg_writer_type<arg_writer_type>();
      new (get_pool()) arg_writer_type(arg);
    }

    template <class arg_writer_type>
    void verify_arg_writer_type()
    {
      BOOST_STATIC_ASSERT(sizeof(arg_writer_type) <= BOOST_MAX_SIZEOF_STRING_IL_WRITER);
      BOOST_STATIC_ASSERT(is_aligned_with_base<arg_writer_type>::value);
    }
    
  };
}; //namespace detail
}; //namespace boost



#endif
