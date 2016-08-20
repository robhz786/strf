#ifndef BOOST_STRINGIFY_FORMATER_TUPLE_HPP
#define BOOST_STRINGIFY_FORMATER_TUPLE_HPP

namespace boost {
namespace stringify {

template <typename ... Formaters>
class formater_tuple
{
public:
    formater_tuple(const Formaters& ... args)
    {
    }
    //todo
};


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_FORMATER_TUPLE_HPP */

