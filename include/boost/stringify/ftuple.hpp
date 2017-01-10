#ifndef BOOST_STRINGIFY_FTUPLE_HPP
#define BOOST_STRINGIFY_FTUPLE_HPP

#include <type_traits>

namespace boost {
namespace stringify {
namespace detail {

template <typename, typename ...>
class ftuple_impl;

template <typename, typename, typename ... >
class ftuple_simple;

template <typename, typename, typename ... >
class ftuple_merge;


template <typename T>
struct is_ftuple
{
private:

    template <typename U>
    static typename U::boost_stringify_ftuple_flag test(U*);

    template <typename U>
    static std::false_type test(...);

public:

    static constexpr bool value = decltype(test<T>((T*)(0)))::value;
};


template
    < typename PreviousTag
    , typename FmtImpl
    , typename ... OtherFmts
    >
class ftuple_simple
    : private ftuple_impl<PreviousTag, OtherFmts ...>
{
    typedef ftuple_impl<PreviousTag, OtherFmts ...> parent;

    template <typename, typename, typename ... >
    friend class boost::stringify::detail::ftuple_simple;

    FmtImpl m_fmt;
    
public:

    struct tag: public parent::tag
    {
    };

    ftuple_simple(const FmtImpl& fmtImpl, const OtherFmts& ...fmts)
        : parent(fmts ...)
        , m_fmt(fmtImpl)
    {
    }

    ftuple_simple(const ftuple_simple&) = default;

    template <typename P>
    ftuple_simple(const ftuple_simple<P, FmtImpl, OtherFmts...>& r)
        : parent(r)
        , m_fmt(r.m_fmt)
    {
    }

    using parent::do_get;

    template <typename InputArg>
    typename std::enable_if
        < FmtImpl::template accept_input_type<InputArg>::value
        , const FmtImpl&
        > :: type
    do_get(const tag&, typename FmtImpl::category) const
    {
        return m_fmt;
    }

    template <typename P>
    using rebind_tag = ftuple_simple<P, FmtImpl, OtherFmts...>;

    typedef std::true_type boost_stringify_ftuple_flag;
};


template
    < typename PreviousTag
    , typename FTuple
    , typename ... OtherFmts
    >
class ftuple_merge
    : private FTuple::template rebind_tag
        < typename boost::stringify::detail::ftuple_impl
              < PreviousTag
              , OtherFmts ...
              >::tag
        >
    , private boost::stringify::detail::ftuple_impl<PreviousTag, OtherFmts ...>
{
    typedef
        boost::stringify::detail::ftuple_impl<PreviousTag, OtherFmts ...>
        parent2;

    typedef
        typename FTuple::template rebind_tag<typename parent2::tag>
        parent1;

    template <typename, typename, typename ... >
    friend class boost::stringify::detail::ftuple_merge;

public:

    ftuple_merge(const FTuple& ft, const OtherFmts& ... others)
        : parent1(ft)
        , parent2(others ...)
    {
    }

    template <typename P>
    ftuple_merge(const ftuple_merge<P, FTuple, OtherFmts...>& r)
        : parent1(r)
        , parent2(r)
    {
    }

    using parent1::do_get;
    using parent2::do_get;

    struct tag : public parent1::tag
    {
    };

    template <typename P>
    using rebind_tag
    = boost::stringify::detail::ftuple_merge<P, FTuple, OtherFmts...>;

    typedef std::true_type boost_stringify_ftuple_flag;
};



template
    < typename PreviousTag
    , typename FTuple
    >
class ftuple_merge<PreviousTag, FTuple>
    : private FTuple::template rebind_tag<PreviousTag>
{
    typedef typename FTuple::template rebind_tag<PreviousTag> parent;

    template <typename, typename, typename ... >
    friend class boost::stringify::detail::ftuple_merge;

public:

    ftuple_merge(const FTuple& ft)
        : parent(ft)
    {
    }

    template <typename P>
    ftuple_merge(const ftuple_merge<P, FTuple>& r)
        : parent(r)
    {
    }

    using parent::do_get;
    
    struct tag : public parent::tag
    {
    };

    template <typename P>
    using rebind_tag
    = boost::stringify::detail::ftuple_merge<P, FTuple>;

    typedef std::true_type boost_stringify_ftuple_flag;
};



struct ftuple_tag_zero
{
};


template <>
class ftuple_impl<ftuple_tag_zero>
{
public:
    ftuple_impl()
    {
    }

    ftuple_impl(const ftuple_impl&)
    {
    }

    template <typename P> ftuple_impl(const ftuple_impl<P>&)
    {
    }
    
    template <typename P>
    using rebind_tag = ftuple_impl<P>;

    typedef std::true_type boost_stringify_ftuple_flag;

    typedef ftuple_tag_zero tag;
    
    template <typename InputArg, typename FmtType>
    typename FmtType::default_impl do_get(const tag&, FmtType) const
    {
        return typename FmtType::default_impl();
    }

};

template <typename PreviousTag>
class ftuple_impl<PreviousTag>
{

public:

    typedef std::true_type boost_stringify_ftuple_flag;

    struct tag : public PreviousTag
    {
    };

    void do_get() const
    {
    }

    template <typename P> ftuple_impl(const ftuple_impl<P>&)
    {
    }

    ftuple_impl() = default;

    ftuple_impl(const ftuple_impl&) = default;

    template <typename P>
    using rebind_tag = ftuple_impl<P>;
};


template
    < typename PreviousTag
    , typename Fmt
    , typename ... OtherFmts
    >
class ftuple_impl<PreviousTag, Fmt, OtherFmts ...>
    : private boost::stringify::detail::ternary_trait
        < boost::stringify::detail::is_ftuple<Fmt>::value
        , ftuple_merge<PreviousTag, Fmt, OtherFmts ...>
        , ftuple_simple<PreviousTag, Fmt, OtherFmts ...>
        > :: type
{

private:

    typedef
    typename boost::stringify::detail::ternary_trait
        < boost::stringify::detail::is_ftuple<Fmt>::value
        , ftuple_merge<PreviousTag, Fmt, OtherFmts ...>
        , ftuple_simple<PreviousTag, Fmt, OtherFmts ...>
        > :: type
    parent;

    template <typename, typename ...>
    friend class boost::stringify::detail::ftuple_impl;

public:

    typedef std::true_type boost_stringify_ftuple_flag;

    using parent::do_get;

    struct tag : public parent::tag
    {
    };

    template <typename P>
    using rebind_tag = ftuple_impl<P, Fmt, OtherFmts ...>;

    ftuple_impl(const Fmt& f, const OtherFmts& ... otherFs)
        : parent(f, otherFs ...)
    {
    }

    template <typename P>
    ftuple_impl(const ftuple_impl<P, Fmt, OtherFmts...>& r)
        : parent(r)
    {
    }
};


} // namespace detail


template <typename ... FmtImpls>
class ftuple
    : private boost::stringify::detail::ftuple_impl
        < boost::stringify::detail::ftuple_tag_zero
        , FmtImpls ...
        >
{
    typedef
        boost::stringify::detail::ftuple_impl
            < boost::stringify::detail::ftuple_tag_zero
            , FmtImpls ...
            >
        parent;

    typedef typename parent::tag tag;

    template <typename, typename, typename...>
    friend class boost::stringify::detail::ftuple_merge;

    template <typename T>
    friend struct boost::stringify::detail::is_ftuple;

    typedef std::true_type boost_stringify_ftuple_flag;

public:

    template <typename P>
    using rebind_tag
    = typename boost::stringify::detail::ftuple_impl<P, FmtImpls ...>;

    ftuple(const FmtImpls& ... fmtimpls) : parent(fmtimpls ...)
    {
    }

    ftuple(const ftuple&) = default;

    template <typename FmtType, typename InputArg>
    decltype(auto) get() const
    {
        return parent::template do_get<InputArg>(tag(), FmtType());
    }
};


template <typename FTuple, typename FmtType, typename InputArg>
using ftuple_get_return_type
= decltype(std::declval<FTuple>().template get<FmtType, InputArg>());


template <typename ... Fmts>
boost::stringify::ftuple<Fmts ...> make_ftuple(const Fmts& ... fmts)
{
    return boost::stringify::ftuple<Fmts ...>(fmts ...);
}


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_FTUPLE_HPP */

