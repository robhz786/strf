#ifndef STRINGIFY_TEST_LIGHTWEIGHT_TEST_LABEL_HPP_INCLUDED
#define STRINGIFY_TEST_LIGHTWEIGHT_TEST_LABEL_HPP_INCLUDED

#include <iostream>
#include <sstream>

namespace test_utils {

class test_label
{
public:

    test_label();

    ~test_label();

    static void print_labels(std::ostream& out);

    std::ostream& get_label_ostream_writer()
    {
        return _ostr;
    }

private:

    static test_label*& _first_of_list();
    static test_label*& _last_of_list();
    static void _push_back_on_list(test_label* node);
    static void _remove_from_list(test_label* node);
    static test_label* _find_last_descendent(test_label* node);

    std::ostringstream _ostr;

    test_label* _parent = nullptr;
    test_label* _child = nullptr;
};

test_label*& test_label::_first_of_list()
{
    static thread_local test_label* ptr = nullptr;
    return ptr;
}

test_label*& test_label::_last_of_list()
{
    static thread_local test_label* ptr = nullptr;
    return ptr;
}

test_label::test_label()
{
    _push_back_on_list(this);
}

test_label::~test_label()
{
    _remove_from_list(this);
}

void test_label::_push_back_on_list(test_label* node)
{
    node->_parent = test_label::_last_of_list();
    if (node->_parent != nullptr)
    {
        node->_parent->_child = node;
    }
    else
    {
        test_label::_first_of_list() = node;
    }
    test_label::_last_of_list() = _find_last_descendent(node);
}

void test_label::_remove_from_list(test_label* node)
{
    if (test_label::_last_of_list() == node)
    {
        test_label::_last_of_list() = node->_parent;
    }
    if (node->_parent != nullptr)
    {
        node->_parent->_child = node->_child;
    }
    else if(test_label::_first_of_list() == node)
    {
        test_label::_first_of_list() = nullptr;
    }
}

test_label* test_label::_find_last_descendent(test_label* node)
{
    while(true)
    {
        if (node->_child == nullptr)
        {
            return node;
        }
        node = node->_child;
    }
}

void test_label::print_labels(std::ostream& out)
{
    auto* node = _first_of_list();
    if (node)
    {
        out << "\n[ " << node->_ostr.str();
        for (node = node->_child; node != nullptr; node = node->_child)
        {
            out << " ; " << node->_ostr.str();
        }
        out << " ]\n";
    }
}

std::ostream& get_test_ostream()
{
    test_label::print_labels(std::cerr);
    return std::cerr;
}

} // namespace test_utils

#define STR_CONCAT(str1, str2) str1 ## str2

#define BOOST_TEST_LABEL_IMPL(LINE)                                     \
    ::test_utils::test_label STR_CONCAT(test_label_, LINE) {};     \
    STR_CONCAT(test_label_, LINE).get_label_ostream_writer()

#define BOOST_TEST_LABEL   BOOST_TEST_LABEL_IMPL(__LINE__)

#ifdef BOOST_LIGHTWEIGHT_TEST_OSTREAM
#  error <boost/core/lightweight_test.hpp> must not be defined before "lightweight_test_label.hpp"
#endif

#define BOOST_LIGHTWEIGHT_TEST_OSTREAM test_utils::get_test_ostream()

#include "boost/lightweight_test.hpp"

#endif // STRINGIFY_TEST_LIGHTWEIGHT_TEST_LABEL_HPP_INCLUDED
