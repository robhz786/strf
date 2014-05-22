#ifndef BOOST_DETAIL_BASIC_STRING_IL_ELEMENT_HPP_INCLUDED
#define BOOST_DETAIL_BASIC_STRING_IL_ELEMENT_HPP_INCLUDED

#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/static_assert.hpp>
#include <boost/string_ini_list/string_il_string_writer.hpp>
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

    template <class T, class writer_t>
    void construct_writer(const T& argument)
    {
      BOOST_STATIC_ASSERT(sizeof(writer_t) <= BOOST_MAX_SIZEOF_STRING_IL_WRITER);
      //BOOST_STATIC_ASSERT(is_base_of<writer_base, writer_t>::value_type == true);

      new (get_pool()) writer_t(argument);

      BOOST_ASSERT(((const void*) static_cast<const writer_t*>(&get_writer())) ==
                   ((const void*) &get_writer()));
    }


/*
  // todo: new version

    template <class T>
    void init_v2(const T& argument)
    {
      construct_writer(get_string_il_element_traits(argument));
    }

    template <class T_writer_traits>
    void construct_writer(T_writer_traits dummy_arg)
    {
      typedef typename T_writer_traits::type<charT, traits> writer_t;

      new (get_pool()) writer_t(argument);
    }

*/
  };
}; //namespace detail
}; //namespace boost



#endif
