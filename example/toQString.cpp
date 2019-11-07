//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <QString>
#include <stringify.hpp>
#include <climits>

class QStringCreator: public strf::basic_outbuf<char16_t>
{
public:

    QStringCreator()
        : strf::basic_outbuf<char16_t>(_buffer, _buffer_size)
    {
    }

    explicit QStringCreator(std::size_t size)
        : strf::basic_outbuf<char16_t>(_buffer, _buffer_size)
    {
        Q_ASSERT(size < static_cast<std::size_t>(INT_MAX));
        _str.reserve(static_cast<int>(size));
    }

    void recycle() override;

    QString finish();

private:

    QString _str;
    std::exception_ptr _eptr = nullptr;
    constexpr static std::size_t _buffer_size = strf::min_size_after_recycle<char16_t>();
    char16_t _buffer[_buffer_size];
};

void QStringCreator::recycle()
{
    if (this->good())
    {
        const QChar * qchar_buffer = reinterpret_cast<QChar*>(_buffer);
        std::size_t count = this->pos() - _buffer;
        try
        {
            _str.append(qchar_buffer, count);
        }
        catch(...)
        {
            _eptr = std::current_exception();
            this->set_good(false);
        }
    }
    this->set_pos(_buffer);
}

QString QStringCreator::finish()
{
    recycle();
    this->set_good(false);
    if (_eptr != nullptr)
    {
        std::rethrow_exception(_eptr);
    }
    return std::move(_str);
}

class QStringCreatorCreator
{
public:
    using char_type = char16_t;
    using finish_type = QString;

    template <typename ... Printers>
    finish_type write(const Printers& ... printers) const
    {
        QStringCreator ob;
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }

    template <typename ... Printers>
    finish_type sized_write( std::size_t size
                           , const Printers& ... printers ) const
    {
        QStringCreator ob(size);
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }
};


constexpr strf::dispatcher_no_reserve<QStringCreatorCreator> toQString{};

int main()
{
    int x = 255;
    QString str = toQString(x, u" in hexadecimal is ", ~strf::hex(x));
    assert(str == "255 in hexadecimal is 0xff");

    return 0;
}
