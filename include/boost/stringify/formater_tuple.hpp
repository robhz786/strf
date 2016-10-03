#ifndef BOOST_STRINGIFY_FORMATER_TUPLE_HPP
#define BOOST_STRINGIFY_FORMATER_TUPLE_HPP

#include <type_traits>

namespace boost {
namespace stringify {


template <typename ... Formaters>
class formater_tuple;


template <>
class formater_tuple<>
{
private:
    
    template <typename FmtImpl, typename InputArg>
    using input_permitted = typename FmtImpl::template accept_input_type<InputArg>;
    
protected:

    struct matching_preference {};
    
    template <typename InputArg, typename FmtType>
    typename FmtType::default_impl
    do_get_fmt(const matching_preference&, FmtType) const
    {
        return typename FmtType::default_impl();
    }

public:
   
    template <typename InputArg, typename FmtType>
    typename FmtType::default_impl
    get_fmt(FmtType fmt_type) const
    {
        return typename FmtType::default_impl();
    }
    
};


template <typename FmtImpl, typename ... OtherFmtImpls>
class formater_tuple<FmtImpl, OtherFmtImpls ...>
    : public formater_tuple<OtherFmtImpls ...>
{
private:
    
    typedef formater_tuple<OtherFmtImpls ...> parent;

    template <typename InputArg>
    using accept_input = typename FmtImpl::template accept_input_type<InputArg>;
    
protected:
    
    struct matching_preference: public parent::matching_preference
    {
    };

    using parent::do_get_fmt;

    template <typename InputArg>
    typename std::enable_if
         < accept_input<InputArg>::value
         , const FmtImpl&
         > :: type
    do_get_fmt
        ( const matching_preference&
        , typename FmtImpl::fmt_type
        ) const
    {
        return m_fmt_impl;
    }

public:
 
    formater_tuple
        ( const FmtImpl& fmt_impl
        , const OtherFmtImpls& ... otherfmtimpls
        )
        : parent(otherfmtimpls ...)
        , m_fmt_impl(fmt_impl)
    {
    }

    template <typename InputArg, typename FmtType>
    auto get_fmt(FmtType fmt_type) const
    -> decltype(this->template do_get_fmt<InputArg>(matching_preference(), fmt_type))
    {
        return this->do_get_fmt<InputArg>(matching_preference(), fmt_type);
    }

private:

    // todo: use private inheritance instead of composition in order to reduce sizeof
    FmtImpl m_fmt_impl;
};


template <typename ... Fmts>
boost::stringify::formater_tuple<Fmts ...> make_formating(const Fmts& ... fmts)
{
    return boost::stringify::formater_tuple<Fmts ...>(fmts ...);
}


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_FORMATER_TUPLE_HPP */

