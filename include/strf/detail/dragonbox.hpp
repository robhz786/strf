#ifndef STRF_DETAIL_DRAGONBOX_HPP
#define STRF_DETAIL_DRAGONBOX_HPP

// Copyright 2020-2021 Junekey Jeon
//
// The contents of this file may be used under the terms of
// the Apache License v2.0 with LLVM Exceptions.
//
//    (See accompanying file LICENSE-Apache or copy at
//     https://llvm.org/foundation/relicensing/LICENSE.txt)
//
// Alternatively, the contents of this file may be used under the terms of
// the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE-Boost or copy at
//     https://www.boost.org/LICENSE_1_0.txt)
//
// Unless required by applicable law or agreed to in writing, this software
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.


#include <strf/detail/standard_lib_functions.hpp>

// Suppress additional buffer overrun check
// I have no idea why MSVC thinks some functions here are vulnerable to the buffer overrun attacks
// No, they aren't.
#if defined(_MSC_VER) && !defined(__CUDACC__)
#define JKJ_SAFEBUFFERS __declspec(safebuffers)
#else
#define JKJ_SAFEBUFFERS
#endif

#define JKJ_FORCEINLINE STRF_FORCEINLINE

#if defined(__SIZEOF_INT128__) && !defined(__CUDACC__)
#define JKJ_WITH_INTERNAL_INT128
#endif

#if defined(__cpp_if_constexpr)
#	define JKJ_IF_CONSTEXPR if constexpr
#else
#	define JKJ_IF_CONSTEXPR if
#	if defined(_MSC_VER) && !defined(__CUDACC__)
#		pragma warning( push )
#		pragma warning( disable : 4127 )
#		define JKJ_MSC_WARNING_4127_DISABLED
#	endif
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif
namespace strf {
namespace detail {
namespace jkj {
namespace dragonbox {
	namespace detail {
		template <class T>
		constexpr STRF_HD int physical_bits() noexcept {
			return sizeof(T) * std::numeric_limits<unsigned char>::digits;
		}

		template <class T>
		constexpr STRF_HD int value_bits() noexcept {
			static_assert(std::is_unsigned<T>::value, "");
			return std::numeric_limits<T>::digits;
		}
	}  // namespace detail

	// These classes expose encoding specs of IEEE-754-like floating-point formats.
	// Currently available formats are IEEE754-binary32 & IEEE754-binary64.

	struct ieee754_binary32 {
		static constexpr int significand_bits = 23;
		static constexpr int exponent_bits = 8;
		static constexpr int min_exponent = -126;
		static constexpr int max_exponent = 127;
		static constexpr int exponent_bias = -127;
		static constexpr int decimal_digits = 9;
	};
	struct ieee754_binary64 {
		static constexpr int significand_bits = 52;
		static constexpr int exponent_bits = 11;
		static constexpr int min_exponent = -1022;
		static constexpr int max_exponent = 1023;
		static constexpr int exponent_bias = -1023;
		static constexpr int decimal_digits = 17;
	};

	// A floating-point traits class defines ways to interpret
	// a bit pattern of given size as an encoding of floating-point number.
	// This is a default implementation of such a traits class,
	// supporting ways to interpret 32-bits into a binary32-encoded floating-point number
	// and to interpret 64-bits into a binary64-encoded floating-point number.
	// Users might specialize this class to change the default behavior for certain types.
	template <class T>
	struct default_float_traits {
		// I don't know if there is a truly reliable way of detecting
		// IEEE-754 binary32/binary64 formats; I just did my best here.
		static_assert(std::numeric_limits<T>::is_iec559&&
			std::numeric_limits<T>::radix == 2 &&
			(detail::physical_bits<T>() == 32 || detail::physical_bits<T>() == 64),
			"default_ieee754_traits only works for 32-bits or 64-bits types "
			"supporting binary32 or binary64 formats!");

		// The type that is being viewed.
		using type = T;

		// Refers to the format specification class.
		using format = typename std::conditional<
			detail::physical_bits<T>() == 32,
			ieee754_binary32,
			ieee754_binary64
		>::type;

		// Defines an unsigned integer type that is large enough
		// to carry a variable of type T.
		// Most of the operations will be done on this integer type.
		using carrier_uint = typename std::conditional<
			detail::physical_bits<T>() == 32,
			std::uint32_t,
			std::uint64_t
		>::type;
		static_assert(sizeof(carrier_uint) == sizeof(T), "");

		// Defines a signed integer type for holding
		// significand bits together with the sign bit.
		using signed_significand = typename std::conditional<
			detail::physical_bits<T>() == 32,
			std::int32_t,
			std::int64_t
		>::type;

		// Number of bits in the above unsigned integer type.
		static constexpr int carrier_bits = detail::physical_bits<carrier_uint>();

		// Extract exponent bits from a bit pattern.
		// The result must be aligned to the LSB so that there is
		// no additional zero paddings on the right.
		// This function does not do bias adjustment.
		STRF_HD static constexpr unsigned int extract_exponent_bits(carrier_uint u) noexcept {
			static_assert(detail::value_bits<unsigned int>() > format::exponent_bits, "");
			//constexpr auto exponent_bits_mask = (unsigned int)(((unsigned int)(1) << format::exponent_bits) - 1);
			return (unsigned int)(u >> format::significand_bits)
				& (unsigned int)(((unsigned int)(1) << format::exponent_bits) - 1);
		}

		// Extract significand bits from a bit pattern.
		// The result must be aligned to the LSB so that there is
		// no additional zero paddings on the right.
		// The result does not contain the implicit bit.
		STRF_HD static constexpr carrier_uint extract_significand_bits(carrier_uint u) noexcept {
			return carrier_uint(u & carrier_uint((carrier_uint(1) << format::significand_bits) - 1));
		}

		// Remove the exponent bits and extract
		// significand bits together with the sign bit.
		STRF_HD static constexpr signed_significand remove_exponent_bits(carrier_uint u,
			unsigned int exponent_bits) noexcept
		{
			return signed_significand(u ^ (carrier_uint(exponent_bits) << format::significand_bits));
		}

		// Shift the obtained signed significand bits to the left by 1 to remove the sign bit.
		STRF_HD static constexpr carrier_uint remove_sign_bit_and_shift(signed_significand s) noexcept {
			return carrier_uint(carrier_uint(s) << 1);
		}

		// The actual value of exponent is obtained by adding
		// this value to the extracted exponent bits.
		static constexpr int exponent_bias = 1 - (1 << (carrier_bits - format::significand_bits - 2));

		// Obtain the actual value of the binary exponent from the
		// extracted exponent bits.
		// STRF_HD static constexpr int binary_exponent(unsigned int exponent_bits) noexcept {
		// 	if (exponent_bits == 0) {
		// 		return format::min_exponent;
		// 	}
		// 	else {
		// 		return int(exponent_bits) + format::exponent_bias;
		// 	}
		// }

		// // Obtain the actual value of the binary exponent from the
		// // extracted significand bits and exponent bits.
		// STRF_HD static constexpr carrier_uint binary_significand(
		// 	carrier_uint significand_bits, unsigned int exponent_bits) noexcept
		// {
		// 	if (exponent_bits == 0) {
		// 		return significand_bits;
		// 	}
		// 	else {
		// 		return significand_bits | (carrier_uint(1) << format::significand_bits);
		// 	}
		// }


		// /* Various boolean observer functions */

		// STRF_HD static constexpr bool is_nonzero(carrier_uint u) noexcept {
		// 	return (u << 1) != 0;
		// }
		// STRF_HD static constexpr bool is_positive(carrier_uint u) noexcept {
		// 	return signed_significand(u) >= 0;
		// }
		// STRF_HD static constexpr bool is_negative(carrier_uint u) noexcept {
		// 	return signed_significand(u) < 0;
		// }
		// STRF_HD static constexpr bool is_positive(signed_significand s) noexcept {
		// 	return s >= 0;
		// }
		// STRF_HD static constexpr bool is_negative(signed_significand s) noexcept {
		// 	return s < 0;
		// }
		STRF_HD static constexpr bool is_finite(unsigned int exponent_bits) noexcept {
			//constexpr unsigned int exponent_bits_all_set = (1u << format::exponent_bits) - 1;
			return exponent_bits != ((1U << format::exponent_bits) - 1);
		}
		// STRF_HD static constexpr bool has_all_zero_significand_bits(signed_significand s) noexcept {
		// 	return (s << 1) == 0;
		// }
		STRF_HD static constexpr bool has_even_significand_bits(carrier_uint u) noexcept {
			return u % 2 == 0;
		}
		STRF_HD static constexpr bool has_even_significand_bits(signed_significand s) noexcept {
			return s % 2 == 0;
		}
	};

	// Convenient wrappers for floating-point traits classes.
	// In order to reduce the argument passing overhead,
	// these classes should be as simple as possible
	// (e.g., no inheritance, no private non-static data member, etc.;
	// this is an unfortunate fact about x64 calling convention).

	template <class T, class Traits = default_float_traits<T>>
	struct float_bits;

	template <class T, class Traits = default_float_traits<T>>
	struct signed_significand_bits;

	template <class T, class Traits>
	struct float_bits {
		using type = T;
		using traits_type = Traits;
		using carrier_uint = typename traits_type::carrier_uint;

		carrier_uint u;

		float_bits() = default;
		STRF_HD explicit float_bits(carrier_uint bit_pattern) noexcept :
			u{ bit_pattern } {}
		STRF_HD explicit float_bits(T float_value) noexcept :
			u{ strf::detail::bit_cast<carrier_uint>(float_value) } {}

		STRF_HD constexpr T to_float() const noexcept {
			return strf::detail::bit_cast<T>(u);
		}

		// Extract exponent bits from a bit pattern.
		// The result must be aligned to the LSB so that there is
		// no additional zero paddings on the right.
		// This function does not do bias adjustment.
		STRF_HD constexpr unsigned int extract_exponent_bits() const noexcept {
			return traits_type::extract_exponent_bits(u);
		}

		// // Extract significand bits from a bit pattern.
		// // The result must be aligned to the LSB so that there is
		// // no additional zero paddings on the right.
		// // The result does not contain the implicit bit.
		// STRF_HD constexpr carrier_uint extract_significand_bits() const noexcept {
		// 	return traits_type::extract_significand_bits(u);
		// }

		// Remove the exponent bits and extract
		// significand bits together with the sign bit.
		STRF_HD constexpr signed_significand_bits<type, traits_type> remove_exponent_bits(
			unsigned int exponent_bits) const noexcept
		{
			return signed_significand_bits<type, traits_type>(
				traits_type::remove_exponent_bits(u, exponent_bits));
		}

		// // Obtain the actual value of the binary exponent from the
		// // extracted exponent bits.
		// STRF_HD static constexpr int binary_exponent(unsigned int exponent_bits) noexcept {
		// 	return traits_type::binary_exponent(exponent_bits);
		// }
		// STRF_HD constexpr int binary_exponent() const noexcept {
		// 	return binary_exponent(extract_exponent_bits());
		// }

		// // Obtain the actual value of the binary exponent from the
		// // extracted significand bits and exponent bits.
		// STRF_HD static constexpr carrier_uint binary_significand(
		// 	carrier_uint significand_bits, unsigned int exponent_bits) noexcept
		// {
		// 	return traits_type::binary_significand(significand_bits, exponent_bits);
		// }
		// STRF_HD constexpr carrier_uint binary_significand() const noexcept
		// {
		// 	return binary_significand(extract_significand_bits(), extract_exponent_bits());
		// }

		// STRF_HD constexpr bool is_nonzero() const noexcept {
		// 	return traits_type::is_nonzero(u);
		// }
		// STRF_HD constexpr bool is_positive() const noexcept {
		// 	return traits_type::is_positive(u);
		// }
		// STRF_HD constexpr bool is_negative() const noexcept {
		// 	return traits_type::is_negative(u);
		// }
		STRF_HD constexpr bool is_finite(unsigned int exponent_bits) const noexcept {
			return traits_type::is_finite(exponent_bits);
		}
		STRF_HD constexpr bool is_finite() const noexcept {
			return traits_type::is_finite(extract_exponent_bits());
		}
		// STRF_HD constexpr bool has_even_significand_bits() const noexcept {
		// 	return traits_type::has_even_significand_bits(u);
		// }
	};

	template <class T, class Traits>
	struct signed_significand_bits {
		using type = T;
		using traits_type = Traits;
		using carrier_uint = typename traits_type::carrier_uint;
		using signed_significand = typename traits_type::signed_significand;

		signed_significand s;

		signed_significand_bits() = default;
		STRF_HD constexpr explicit signed_significand_bits(signed_significand bit_pattern) noexcept :
			s{ bit_pattern } {}

		// Shift the obtained signed significand bits to the left by 1 to remove the sign bit.
		STRF_HD constexpr carrier_uint remove_sign_bit_and_shift() const noexcept {
			return traits_type::remove_sign_bit_and_shift(s);
		}

		STRF_HD constexpr bool is_positive() const noexcept {
			return traits_type::is_positive(s);
		}
		STRF_HD constexpr bool is_negative() const noexcept {
			return traits_type::is_negative(s);
		}
		STRF_HD constexpr bool has_all_zero_significand_bits() const noexcept {
			return traits_type::has_all_zero_significand_bits(s);
		}
		STRF_HD constexpr bool has_even_significand_bits() const noexcept {
			return traits_type::has_even_significand_bits(s);
		}
	};

	namespace detail {
		////////////////////////////////////////////////////////////////////////////////////////
		// Bit operation intrinsics.
		////////////////////////////////////////////////////////////////////////////////////////

		namespace bits {
			template <class UInt>
			STRF_HD inline int countr_zero(UInt n) noexcept {
				static_assert(std::is_unsigned<UInt>::value && value_bits<UInt>() <= 64, "");
				STRF_ASSERT(n != 0);
#if defined(__CUDACC__)
				JKJ_IF_CONSTEXPR (sizeof(UInt) == 4) {
					return __ffs(n) - 1;
				} else {
					return __ffsll(n) - 1;
				}
#elif defined(__GNUC__) || defined(__clang__)
#define JKJ_HAS_COUNTR_ZERO_INTRINSIC 1
				static_assert( std::is_same<UInt, unsigned long>::value
					|| std::is_same<UInt, unsigned long long>::value
					|| sizeof(UInt) <= sizeof(unsigned int), "" );
				JKJ_IF_CONSTEXPR (std::is_same<UInt, unsigned long>::value) {
					return __builtin_ctzl(n);
				}
				else JKJ_IF_CONSTEXPR (std::is_same<UInt, unsigned long long>::value) {
					return __builtin_ctzll(n);
				}
				else {
					return __builtin_ctz((unsigned int)n);
				}
#elif defined(_MSC_VER) && !defined(__CUDACC__)
#define JKJ_HAS_COUNTR_ZERO_INTRINSIC 1
				static_assert( std::is_same<UInt, unsigned __int64>::value
					|| sizeof(UInt) <= sizeof(unsigned int), "" );
				JKJ_IF_CONSTEXPR (std::is_same<UInt, unsigned __int64>::value) {
#if defined(_M_X64)
					return int(_tzcnt_u64(n));
#else
					return ((unsigned int)(n) == 0) ?
						(32 + (_tzcnt_u32((unsigned int)(n >> 32)))) :
						(_tzcnt_u32((unsigned int)n));
#endif
				}
				else {
					return int(_tzcnt_u32((unsigned int)n));
				}
#else
#define JKJ_HAS_COUNTR_ZERO_INTRINSIC 0
				int count;
				auto n32 = std::uint32_t(n);

				JKJ_IF_CONSTEXPR (value_bits<UInt>() > 32) {
					if (n32 != 0) {
						count = 31;
					}
					else {
						n32 = std::uint32_t(n >> 32);
						JKJ_IF_CONSTEXPR (value_bits<UInt>() == 64) {
							if (n32 != 0) {
								count = 63;
							}
							else {
								return 64;
							}
						}
						else {
							count = value_bits<UInt>();
						}
					}
				}
				else {
					JKJ_IF_CONSTEXPR (value_bits<UInt>() == 32) {
						if (n32 != 0) {
							count = 31;
						}
						else {
							return 32;
						}
					}
					else {
						count = value_bits<UInt>();
					}
				}

				n32 &= (0 - n32);
				JKJ_IF_CONSTEXPR (value_bits<UInt>() > 16) {
					if ((n32 & 0x0000ffff) != 0) count -= 16;
				}
				JKJ_IF_CONSTEXPR (value_bits<UInt>() > 8) {
					if ((n32 & 0x00ff00ff) != 0) count -= 8;
				}
				if ((n32 & 0x0f0f0f0f) != 0) count -= 4;
				if ((n32 & 0x33333333) != 0) count -= 2;
				if ((n32 & 0x55555555) != 0) count -= 1;

				return count;
#endif
			}
		}  // namespace bits

		////////////////////////////////////////////////////////////////////////////////////////
		// Utilities for wide unsigned integer arithmetic.
		////////////////////////////////////////////////////////////////////////////////////////

		namespace wuint {
			struct uint128 {
				uint128() = default;

#if defined(JKJ_WITH_INTERNAL_INT128)
				// To silence "error: ISO C++ does not support '__int128' for 'type name' [-Wpedantic]"
#if defined(__GNUC__)
				__extension__
#endif
				using uint128_internal = unsigned __int128;

				uint128_internal	internal_;

				STRF_HD constexpr uint128(std::uint64_t high, std::uint64_t low) noexcept :
					internal_{ ((uint128_internal)low) | (((uint128_internal)high) << 64) } {}

				STRF_HD constexpr uint128(uint128_internal u) noexcept : internal_{ u } {}

				STRF_HD constexpr std::uint64_t high() const noexcept {
					return std::uint64_t(internal_ >> 64);
				}
				STRF_HD constexpr std::uint64_t low() const noexcept {
					return std::uint64_t(internal_);
				}

				STRF_HD uint128& operator+=(std::uint64_t n) & noexcept {
					internal_ += n;
					return *this;
				}
#else
				std::uint64_t	high_;
				std::uint64_t	low_;

				STRF_HD constexpr uint128(std::uint64_t high, std::uint64_t low) noexcept :
					high_{ high }, low_{ low } {}

				STRF_HD constexpr std::uint64_t high() const noexcept {
					return high_;
				}
				STRF_HD constexpr std::uint64_t low() const noexcept {
					return low_;
				}

				STRF_HD uint128& operator+=(std::uint64_t n) & noexcept {
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__CUDACC__)
					auto carry = _addcarry_u64(0, low_, n, &low_);
					_addcarry_u64(carry, high_, 0, &high_);
					return *this;
#else
					auto sum = low_ + n;
					high_ += (sum < low_ ? 1 : 0);
					low_ = sum;
					return *this;
#endif
				}
#endif
			};

#if !defined(JKJ_WITH_INTERNAL_INT128)
			STRF_HD static inline std::uint64_t umul64(std::uint32_t x, std::uint32_t y) noexcept {
#if defined(_MSC_VER) && defined(_M_IX86) && !defined(__CUDACC__)
				return __emulu(x, y);
#else
				return x * std::uint64_t(y);
#endif
			}
#endif

			// Get 128-bit result of multiplication of two 64-bit unsigned integers.
			STRF_HD JKJ_SAFEBUFFERS inline uint128 umul128(std::uint64_t x, std::uint64_t y) noexcept {
#if defined(JKJ_WITH_INTERNAL_INT128)
				return (uint128::uint128_internal)(x) * (uint128::uint128_internal)(y);
#elif defined(_MSC_VER) && defined(_M_X64) && !defined(__CUDACC__)
				uint128 result;
				result.low_ = _umul128(x, y, &result.high_);
				return result;
#else
				auto a = std::uint32_t(x >> 32);
				auto b = std::uint32_t(x);
				auto c = std::uint32_t(y >> 32);
				auto d = std::uint32_t(y);

				auto ac = umul64(a, c);
				auto bc = umul64(b, c);
				auto ad = umul64(a, d);
				auto bd = umul64(b, d);

				auto intermediate = (bd >> 32) + std::uint32_t(ad) + std::uint32_t(bc);

				return{ ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32),
					(intermediate << 32) + std::uint32_t(bd) };
#endif
			}

			STRF_HD JKJ_SAFEBUFFERS inline std::uint64_t umul128_upper64(std::uint64_t x, std::uint64_t y) noexcept {
#if defined(JKJ_WITH_INTERNAL_INT128)
				auto p = (uint128::uint128_internal)(x) * (uint128::uint128_internal)(y);
				return std::uint64_t(p >> 64);
#elif defined(_MSC_VER) && defined(_M_X64) && !defined(__CUDACC__)
				return __umulh(x, y);
#else
				auto a = std::uint32_t(x >> 32);
				auto b = std::uint32_t(x);
				auto c = std::uint32_t(y >> 32);
				auto d = std::uint32_t(y);

				auto ac = umul64(a, c);
				auto bc = umul64(b, c);
				auto ad = umul64(a, d);
				auto bd = umul64(b, d);

				auto intermediate = (bd >> 32) + std::uint32_t(ad) + std::uint32_t(bc);

				return ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32);
#endif
			}

			// Get upper 64-bits of multiplication of a 64-bit unsigned integer and a 128-bit unsigned integer.
			STRF_HD JKJ_SAFEBUFFERS inline std::uint64_t umul192_upper64(std::uint64_t x, uint128 y) noexcept {
				auto g0 = umul128(x, y.high());
				g0 += umul128_upper64(x, y.low());
				return g0.high();
			}

			// Get upper 32-bits of multiplication of a 32-bit unsigned integer and a 64-bit unsigned integer.
			STRF_HD inline std::uint32_t umul96_upper32(std::uint32_t x, std::uint64_t y) noexcept {
#if defined(JKJ_WITH_INTERNAL_INT128) || (defined(_MSC_VER) && defined(_M_X64) && !defined(__CUDACC__))
				return std::uint32_t(umul128_upper64(x, y));
#else
				//std::uint32_t a = 0;
				auto b = x;
				auto c = std::uint32_t(y >> 32);
				auto d = std::uint32_t(y);

				//std::uint64_t ac = 0;
				auto bc = umul64(b, c);
				//std::uint64_t ad = 0;
				auto bd = umul64(b, d);

				auto intermediate = (bd >> 32) + bc;
				return std::uint32_t(intermediate >> 32);
#endif
			}

			// Get middle 64-bits of multiplication of a 64-bit unsigned integer and a 128-bit unsigned integer.
			STRF_HD JKJ_SAFEBUFFERS inline std::uint64_t umul192_middle64(std::uint64_t x, uint128 y) noexcept {
				auto g01 = x * y.high();
				auto g10 = umul128_upper64(x, y.low());
				return g01 + g10;
			}

			// Get middle 32-bits of multiplication of a 32-bit unsigned integer and a 64-bit unsigned integer.
			STRF_HD inline std::uint64_t umul96_lower64(std::uint32_t x, std::uint64_t y) noexcept {
				return x * y;
			}
		}  // namespace wuint

		////////////////////////////////////////////////////////////////////////////////////////
		// Some simple utilities for constexpr computation.
		////////////////////////////////////////////////////////////////////////////////////////

		template <typename IntT, IntT X, unsigned K>
		struct mp_compute_power
		{
			static constexpr IntT value = X * mp_compute_power<IntT, X, K - 1>::value;
		};
		template <typename IntT, IntT X>
		struct mp_compute_power<IntT, X, 0>
		{
			static constexpr IntT value = 1;
		};

		template <int k, class Int>
		STRF_HD STRF_CONSTEXPR_IN_CXX14 Int compute_power(Int a) noexcept {
			static_assert(k >= 0, "");
			Int p = 1;
			for (int i = 0; i < k; ++i) {
				p *= a;
			}
			return p;
		}

		template <int a, class UInt>
		STRF_HD constexpr int do_count_factors(UInt n, int c) noexcept {
			static_assert(a > 1, "");
			return (n % a == 0) ? do_count_factors<a>(n / a, c + 1) : c;
		}

		template <int a, class UInt>
		STRF_HD constexpr int count_factors(UInt n) noexcept {
			return do_count_factors<a>(n, 0);
		}

		////////////////////////////////////////////////////////////////////////////////////////
		// Utilities for fast/constexpr log computation.
		////////////////////////////////////////////////////////////////////////////////////////

		namespace log {

			template
				< std::uint32_t integer_part
				, std::uint64_t fractional_digits
				, std::size_t shift_amount >
			struct floor_shift {
				static_assert(shift_amount < 32, "");
				// Ensure no overflow
				static_assert(shift_amount == 0 || integer_part < (std::uint32_t(1) << (32 - shift_amount)), "");
				static constexpr std::int32_t value =
					( shift_amount == 0
					? std::int32_t(integer_part)
					: ( (integer_part << shift_amount) |
							(fractional_digits >> (64 - shift_amount)) ));
			};

			// Compute floor(e * c - s).
			template <
				std::uint32_t c_integer_part,
				std::uint64_t c_fractional_digits,
				std::size_t shift_amount,
				std::int32_t max_exponent,
				std::uint32_t s_integer_part = 0,
				std::uint64_t s_fractional_digits = 0
			>
			STRF_HD constexpr int compute(int e) noexcept {
				STRF_ASSERT_IN_CONSTEXPR(e <= max_exponent && e >= -max_exponent);
				using c = floor_shift<c_integer_part, c_fractional_digits, shift_amount>;
				using s = floor_shift<s_integer_part, s_fractional_digits, shift_amount>;
				return int((std::int32_t(e) * c::value - s::value) >> shift_amount);
			}

			constexpr std::uint64_t log10_2_fractional_digits{ 0x4d104d427de7fbcc };
			constexpr std::uint64_t log10_4_over_3_fractional_digits{ 0x1ffbfc2bbc780375 };
			constexpr std::size_t floor_log10_pow2_shift_amount = 22;
			constexpr int floor_log10_pow2_input_limit = 1700;
			constexpr int floor_log10_pow2_minus_log10_4_over_3_input_limit = 1700;

			constexpr std::uint64_t log2_10_fractional_digits{ 0x5269e12f346e2bf9 };
			constexpr std::size_t floor_log2_pow10_shift_amount = 19;
			constexpr int floor_log2_pow10_input_limit = 1233;

			constexpr std::uint64_t log5_2_fractional_digits{ 0x6e40d1a4143dcb94 };
			constexpr std::uint64_t log5_3_fractional_digits{ 0xaebf47915d443b24 };
			constexpr std::size_t floor_log5_pow2_shift_amount = 20;
			constexpr int floor_log5_pow2_input_limit = 1492;
			constexpr int floor_log5_pow2_minus_log5_3_input_limit = 2427;

			// For constexpr computation.
			// Returns -1 when n = 0.

			template <class UInt>
			STRF_HD constexpr int do_floor_log2(UInt n, int count) noexcept {
				return n != 0 ? do_floor_log2(n >> 1, count + 1) : count;
			}

			template <class UInt>
			STRF_HD constexpr int floor_log2(UInt n) noexcept {
				return do_floor_log2(n, -1);
			}

			STRF_HD constexpr int floor_log10_pow2(int e) noexcept {
				using namespace log;
				return compute<
					0, log10_2_fractional_digits,
					floor_log10_pow2_shift_amount,
					floor_log10_pow2_input_limit>(e);
			}

			STRF_HD constexpr int floor_log2_pow10(int e) noexcept {
				using namespace log;
				return compute<
					3, log2_10_fractional_digits,
					floor_log2_pow10_shift_amount,
					floor_log2_pow10_input_limit>(e);
			}

			STRF_HD constexpr int floor_log5_pow2(int e) noexcept {
				using namespace log;
				return compute<
					0, log5_2_fractional_digits,
					floor_log5_pow2_shift_amount,
					floor_log5_pow2_input_limit>(e);
			}

			STRF_HD constexpr int floor_log5_pow2_minus_log5_3(int e) noexcept {
				using namespace log;
				return compute<
					0, log5_2_fractional_digits,
					floor_log5_pow2_shift_amount,
					floor_log5_pow2_minus_log5_3_input_limit,
					0, log5_3_fractional_digits>(e);
			}

			STRF_HD constexpr int floor_log10_pow2_minus_log10_4_over_3(int e) noexcept {
				using namespace log;
				return compute<
					0, log10_2_fractional_digits,
					floor_log10_pow2_shift_amount,
					floor_log10_pow2_minus_log10_4_over_3_input_limit,
					0, log10_4_over_3_fractional_digits>(e);
			}
		}  // namespace log

		////////////////////////////////////////////////////////////////////////////////////////
		// Utilities for fast divisibility tests.
		////////////////////////////////////////////////////////////////////////////////////////

		namespace div {
			template <class UInt> struct division_by_5;

			template <>
			struct division_by_5<std::uint32_t> {
				static constexpr std::size_t table_size = 11;

				STRF_HD static inline std::uint32_t mod_inv(std::size_t x) noexcept {
					STRF_ASSERT(x < table_size);
					static constexpr std::uint32_t table[table_size] = {
						0x1UL, 0xcccccccdUL, 0xc28f5c29UL, 0x26e978d5UL, 0x3afb7e91UL,
						0xbcbe61dUL, 0x68c26139UL, 0xae8d46a5UL,
						0x22e90e21UL, 0x3a2e9c6dUL, 0x3ed61f49UL
					};
					return table[x];
				}

				STRF_HD static inline std::uint32_t max_quotients(std::size_t x) noexcept {
					STRF_ASSERT(x < table_size);
					static constexpr std::uint32_t  table[table_size] = {
						0xffffffffUL, 0x33333333UL, 0xa3d70a3UL, 0x20c49baUL, 0x68db8bUL,
						0x14f8b5UL, 0x431bdUL, 0xd6bfUL, 0x2af3UL, 0x897UL, 0x1b7UL
					};
					return table[x];
				}
			};

			template <>
			struct division_by_5<std::uint64_t> {
				static constexpr std::size_t table_size = 24;

				STRF_HD static inline std::uint64_t mod_inv(std::size_t x) noexcept {
					STRF_ASSERT(x < table_size);
					static constexpr std::uint64_t table[table_size] = {
						0x1ULL, 0xcccccccccccccccdULL, 0x8f5c28f5c28f5c29ULL,
						0x1cac083126e978d5ULL, 0xd288ce703afb7e91ULL, 0x5d4e8fb00bcbe61dULL,
						0x790fb65668c26139ULL, 0xe5032477ae8d46a5ULL, 0xc767074b22e90e21ULL,
						0x8e47ce423a2e9c6dULL, 0x4fa7f60d3ed61f49ULL, 0xfee64690c913975ULL,
						0x3662e0e1cf503eb1ULL, 0xa47a2cf9f6433fbdULL, 0x54186f653140a659ULL,
						0x7738164770402145ULL, 0xe4a4d1417cd9a041ULL, 0xc75429d9e5c5200dULL,
						0xc1773b91fac10669ULL, 0x26b172506559ce15ULL, 0xd489e3a9addec2d1ULL,
						0x90e860bb892c8d5dULL, 0x502e79bf1b6f4f79ULL, 0xdcd618596be30fe5ULL
					};
					return table[x];
				}

				STRF_HD static inline std::uint64_t max_quotients(std::size_t x) noexcept {
					STRF_ASSERT(x < table_size);
					static constexpr std::uint64_t table[table_size] = {
						0xffffffffffffffffULL, 0x3333333333333333ULL, 0xa3d70a3d70a3d70ULL,
						0x20c49ba5e353f7cULL, 0x68db8bac710cb2ULL, 0x14f8b588e368f0ULL,
						0x431bde82d7b63ULL, 0xd6bf94d5e57aULL, 0x2af31dc46118ULL,
						0x89705f4136bULL, 0x1b7cdfd9d7bULL, 0x57f5ff85e5ULL, 0x119799812dULL,
						0x384b84d09ULL, 0xb424dc35ULL, 0x24075f3dULL, 0x734aca5ULL, 0x170ef54ULL,
						0x49c977ULL, 0xec1e4ULL, 0x2f394ULL, 0x971dULL, 0x1e39ULL, 0x60bULL
					};
					return table[x];
				}
			};

			template <std::size_t asserted_table_size, class UInt>
			STRF_HD constexpr bool divisible_by_power_of_5(UInt x, unsigned int exp) noexcept {
				using div5 = division_by_5<UInt>;
				static_assert(asserted_table_size <= div5::table_size,
					"division_by_5 tables are too small");
				STRF_ASSERT_IN_CONSTEXPR(exp < div5::table_size);
				return (x * div5::mod_inv(exp)) <= div5::max_quotients(exp);
			}

			template <class UInt>
			STRF_HD STRF_CONSTEXPR_IN_CXX14 bool divisible_by_power_of_2(UInt x, unsigned int exp) noexcept {
				STRF_ASSERT(exp >= 1);
				STRF_ASSERT(x != 0);
#if JKJ_HAS_COUNTR_ZERO_INTRINSIC
				return bits::countr_zero(x) >= int(exp);
#else
				if ((int)exp >= value_bits<UInt>()) {
					return false;
				}
				auto mask = UInt((UInt(1) << exp) - 1);
				return (x & mask) == 0;
#endif
			}

			// Replace n by floor(n / 10^N).
			// Returns true if and only if n is divisible by 10^N.
			// Precondition: n <= 10^(N+1)
			// !!It takes an in-out parameter!!
			template <int N>
			struct check_divisibility_and_divide_by_pow10_info;

			template <>
			struct check_divisibility_and_divide_by_pow10_info<1> {
				static constexpr std::uint32_t magic_number = 0xcccd;
				static constexpr int bits_for_comparison = 16;
				static constexpr std::uint32_t threshold = 0x3333;
				static constexpr int shift_amount = 19;
			};

			template <>
			struct check_divisibility_and_divide_by_pow10_info<2> {
				static constexpr std::uint32_t magic_number = 0x147c29;
				static constexpr int bits_for_comparison = 12;
				static constexpr std::uint32_t threshold = 0xa3;
				static constexpr int shift_amount = 27;
			};

			template <int N>
			STRF_HD STRF_CONSTEXPR_IN_CXX14 bool check_divisibility_and_divide_by_pow10(std::uint32_t& n) noexcept
			{
				// Make sure the computation for max_n does not overflow.
				static_assert(N + 1 <= log::floor_log10_pow2(31), "");
				STRF_ASSERT(n <= (mp_compute_power<std::uint32_t, 10, N + 1>::value));

				using info = check_divisibility_and_divide_by_pow10_info<N>;
				n *= info::magic_number;

				constexpr std::uint32_t comparison_mask =
					( info::bits_for_comparison >= 32
					? ((std::uint32_t)-1)
					: std::uint32_t((std::uint32_t(1) << info::bits_for_comparison) - 1) );

				// The lowest N bits of (n & comparison_mask) must be zero, and
				// (n >> N) & comparison_mask must be at most threshold.
				auto c = ((n >> N) | (n << (info::bits_for_comparison - N))) & comparison_mask;

				n >>= info::shift_amount;
				return c <= info::threshold;
			}

			// Compute floor(n / 10^N) for small n and N.
			// Precondition: n <= 10^(N+1)
			template <int N>
			struct small_division_by_pow10_info;

			template <>
			struct small_division_by_pow10_info<1> {
				static constexpr std::uint32_t magic_number = 0xcccd;
				static constexpr int shift_amount = 19;
			};

			template <>
			struct small_division_by_pow10_info<2> {
				static constexpr std::uint32_t magic_number = 0xa3d8;
				static constexpr int shift_amount = 22;
			};

			template <int N>
			STRF_HD STRF_CONSTEXPR_IN_CXX14 std::uint32_t small_division_by_pow10(std::uint32_t n) noexcept
			{
				STRF_ASSERT(n <= compute_power<N + 1>(std::uint32_t(10)));
				return (n * small_division_by_pow10_info<N>::magic_number)
					>> small_division_by_pow10_info<N>::shift_amount;
			}

			template <int, typename UInt>
			STRF_HD constexpr UInt divide_by_pow10_(std::true_type, UInt n) noexcept {
				return wuint::umul128_upper64(n, 0x83126e978d4fdf3c) >> 9;
			}
			template <int N, typename UInt>
			STRF_HD constexpr UInt divide_by_pow10_(std::false_type, UInt n) noexcept {
				return n / mp_compute_power<UInt, 10, N>::value;
			}
			// Compute floor(n / 10^N) for small N.
			// Precondition: n <= 2^a * 5^b (a = max_pow2, b = max_pow5)
			template <int N, int max_pow2, int max_pow5, class UInt>
			STRF_HD constexpr UInt divide_by_pow10(UInt n) noexcept
			{
				static_assert(N >= 0, "");

				// Ensure no overflow.
				static_assert(max_pow2 + (log::floor_log2_pow10(max_pow5) - max_pow5) < value_bits<UInt>(), "");

				// Specialize for 64-bit division by 1000.
				// Ensure that the correctness condition is met.
				using c = std::integral_constant< bool
					, ( std::is_same<UInt, std::uint64_t>::value && N == 3 &&
					max_pow2 + (log::floor_log2_pow10(N + max_pow5) - (N + max_pow5)) < 70) >;
				return divide_by_pow10_<N>(c{}, n);
			}
		}  // namespace div
	}  // namespace detail

	////////////////////////////////////////////////////////////////////////////////////////
	// Return types for the main interface function.
	////////////////////////////////////////////////////////////////////////////////////////

	template <class UInt, bool is_signed, bool trailing_zero_flag>
	struct decimal_fp;

	template <class UInt>
	struct decimal_fp<UInt, false, false> {
		using carrier_uint = UInt;

		carrier_uint	significand;
		int				exponent;
	};

	template <class UInt>
	struct decimal_fp<UInt, true, false> {
		using carrier_uint = UInt;

		carrier_uint	significand;
		int				exponent;
		bool			is_negative;
	};

	template <class UInt>
	struct decimal_fp<UInt, false, true> {
		using carrier_uint = UInt;

		carrier_uint	significand;
		int				exponent;
		bool			may_have_trailing_zeros;
	};

	template <class UInt>
	struct decimal_fp<UInt, true, true> {
		using carrier_uint = UInt;

		carrier_uint	significand;
		int				exponent;
		bool			is_negative;
		bool			may_have_trailing_zeros;
	};

	template <class UInt>
	using unsigned_decimal_fp = decimal_fp<UInt, false, false>;

	template <class UInt>
	using signed_decimal_fp = decimal_fp<UInt, true, false>;


	////////////////////////////////////////////////////////////////////////////////////////
	// Computed cache entries.
	////////////////////////////////////////////////////////////////////////////////////////

	namespace detail {
		template <class FloatFormat>
		struct cache_holder;

		template <>
		struct cache_holder<ieee754_binary32> {
			using cache_entry_type = std::uint64_t;
			static constexpr int cache_bits = 64;
			static constexpr int min_k = -31;
			static constexpr int max_k = 46;
			STRF_HD static inline cache_entry_type cache(std::size_t idx) noexcept {
				static const cache_entry_type table[] = {
					0x81ceb32c4b43fcf5,
					0xa2425ff75e14fc32,
					0xcad2f7f5359a3b3f,
					0xfd87b5f28300ca0e,
					0x9e74d1b791e07e49,
					0xc612062576589ddb,
					0xf79687aed3eec552,
					0x9abe14cd44753b53,
					0xc16d9a0095928a28,
					0xf1c90080baf72cb2,
					0x971da05074da7bef,
					0xbce5086492111aeb,
					0xec1e4a7db69561a6,
					0x9392ee8e921d5d08,
					0xb877aa3236a4b44a,
					0xe69594bec44de15c,
					0x901d7cf73ab0acda,
					0xb424dc35095cd810,
					0xe12e13424bb40e14,
					0x8cbccc096f5088cc,
					0xafebff0bcb24aaff,
					0xdbe6fecebdedd5bf,
					0x89705f4136b4a598,
					0xabcc77118461cefd,
					0xd6bf94d5e57a42bd,
					0x8637bd05af6c69b6,
					0xa7c5ac471b478424,
					0xd1b71758e219652c,
					0x83126e978d4fdf3c,
					0xa3d70a3d70a3d70b,
					0xcccccccccccccccd,
					0x8000000000000000,
					0xa000000000000000,
					0xc800000000000000,
					0xfa00000000000000,
					0x9c40000000000000,
					0xc350000000000000,
					0xf424000000000000,
					0x9896800000000000,
					0xbebc200000000000,
					0xee6b280000000000,
					0x9502f90000000000,
					0xba43b74000000000,
					0xe8d4a51000000000,
					0x9184e72a00000000,
					0xb5e620f480000000,
					0xe35fa931a0000000,
					0x8e1bc9bf04000000,
					0xb1a2bc2ec5000000,
					0xde0b6b3a76400000,
					0x8ac7230489e80000,
					0xad78ebc5ac620000,
					0xd8d726b7177a8000,
					0x878678326eac9000,
					0xa968163f0a57b400,
					0xd3c21bcecceda100,
					0x84595161401484a0,
					0xa56fa5b99019a5c8,
					0xcecb8f27f4200f3a,
					0x813f3978f8940984,
					0xa18f07d736b90be5,
					0xc9f2c9cd04674ede,
					0xfc6f7c4045812296,
					0x9dc5ada82b70b59d,
					0xc5371912364ce305,
					0xf684df56c3e01bc6,
					0x9a130b963a6c115c,
					0xc097ce7bc90715b3,
					0xf0bdc21abb48db20,
					0x96769950b50d88f4,
					0xbc143fa4e250eb31,
					0xeb194f8e1ae525fd,
					0x92efd1b8d0cf37be,
					0xb7abc627050305ad,
					0xe596b7b0c643c719,
					0x8f7e32ce7bea5c6f,
					0xb35dbf821ae4f38b,
					0xe0352f62a19e306e
				};
				return table[idx];
			}
		};

		template <>
		struct cache_holder<ieee754_binary64> {
			using cache_entry_type = wuint::uint128;
			static constexpr int cache_bits = 128;
			static constexpr int min_k = -292;
			static constexpr int max_k = 326;
			STRF_HD static inline cache_entry_type cache(std::size_t idx) noexcept {
				static const cache_entry_type table[] = {
					{ 0xff77b1fcbebcdc4f, 0x25e8e89c13bb0f7b },
					{ 0x9faacf3df73609b1, 0x77b191618c54e9ad },
					{ 0xc795830d75038c1d, 0xd59df5b9ef6a2418 },
					{ 0xf97ae3d0d2446f25, 0x4b0573286b44ad1e },
					{ 0x9becce62836ac577, 0x4ee367f9430aec33 },
					{ 0xc2e801fb244576d5, 0x229c41f793cda740 },
					{ 0xf3a20279ed56d48a, 0x6b43527578c11110 },
					{ 0x9845418c345644d6, 0x830a13896b78aaaa },
					{ 0xbe5691ef416bd60c, 0x23cc986bc656d554 },
					{ 0xedec366b11c6cb8f, 0x2cbfbe86b7ec8aa9 },
					{ 0x94b3a202eb1c3f39, 0x7bf7d71432f3d6aa },
					{ 0xb9e08a83a5e34f07, 0xdaf5ccd93fb0cc54 },
					{ 0xe858ad248f5c22c9, 0xd1b3400f8f9cff69 },
					{ 0x91376c36d99995be, 0x23100809b9c21fa2 },
					{ 0xb58547448ffffb2d, 0xabd40a0c2832a78b },
					{ 0xe2e69915b3fff9f9, 0x16c90c8f323f516d },
					{ 0x8dd01fad907ffc3b, 0xae3da7d97f6792e4 },
					{ 0xb1442798f49ffb4a, 0x99cd11cfdf41779d },
					{ 0xdd95317f31c7fa1d, 0x40405643d711d584 },
					{ 0x8a7d3eef7f1cfc52, 0x482835ea666b2573 },
					{ 0xad1c8eab5ee43b66, 0xda3243650005eed0 },
					{ 0xd863b256369d4a40, 0x90bed43e40076a83 },
					{ 0x873e4f75e2224e68, 0x5a7744a6e804a292 },
					{ 0xa90de3535aaae202, 0x711515d0a205cb37 },
					{ 0xd3515c2831559a83, 0x0d5a5b44ca873e04 },
					{ 0x8412d9991ed58091, 0xe858790afe9486c3 },
					{ 0xa5178fff668ae0b6, 0x626e974dbe39a873 },
					{ 0xce5d73ff402d98e3, 0xfb0a3d212dc81290 },
					{ 0x80fa687f881c7f8e, 0x7ce66634bc9d0b9a },
					{ 0xa139029f6a239f72, 0x1c1fffc1ebc44e81 },
					{ 0xc987434744ac874e, 0xa327ffb266b56221 },
					{ 0xfbe9141915d7a922, 0x4bf1ff9f0062baa9 },
					{ 0x9d71ac8fada6c9b5, 0x6f773fc3603db4aa },
					{ 0xc4ce17b399107c22, 0xcb550fb4384d21d4 },
					{ 0xf6019da07f549b2b, 0x7e2a53a146606a49 },
					{ 0x99c102844f94e0fb, 0x2eda7444cbfc426e },
					{ 0xc0314325637a1939, 0xfa911155fefb5309 },
					{ 0xf03d93eebc589f88, 0x793555ab7eba27cb },
					{ 0x96267c7535b763b5, 0x4bc1558b2f3458df },
					{ 0xbbb01b9283253ca2, 0x9eb1aaedfb016f17 },
					{ 0xea9c227723ee8bcb, 0x465e15a979c1cadd },
					{ 0x92a1958a7675175f, 0x0bfacd89ec191eca },
					{ 0xb749faed14125d36, 0xcef980ec671f667c },
					{ 0xe51c79a85916f484, 0x82b7e12780e7401b },
					{ 0x8f31cc0937ae58d2, 0xd1b2ecb8b0908811 },
					{ 0xb2fe3f0b8599ef07, 0x861fa7e6dcb4aa16 },
					{ 0xdfbdcece67006ac9, 0x67a791e093e1d49b },
					{ 0x8bd6a141006042bd, 0xe0c8bb2c5c6d24e1 },
					{ 0xaecc49914078536d, 0x58fae9f773886e19 },
					{ 0xda7f5bf590966848, 0xaf39a475506a899f },
					{ 0x888f99797a5e012d, 0x6d8406c952429604 },
					{ 0xaab37fd7d8f58178, 0xc8e5087ba6d33b84 },
					{ 0xd5605fcdcf32e1d6, 0xfb1e4a9a90880a65 },
					{ 0x855c3be0a17fcd26, 0x5cf2eea09a550680 },
					{ 0xa6b34ad8c9dfc06f, 0xf42faa48c0ea481f },
					{ 0xd0601d8efc57b08b, 0xf13b94daf124da27 },
					{ 0x823c12795db6ce57, 0x76c53d08d6b70859 },
					{ 0xa2cb1717b52481ed, 0x54768c4b0c64ca6f },
					{ 0xcb7ddcdda26da268, 0xa9942f5dcf7dfd0a },
					{ 0xfe5d54150b090b02, 0xd3f93b35435d7c4d },
					{ 0x9efa548d26e5a6e1, 0xc47bc5014a1a6db0 },
					{ 0xc6b8e9b0709f109a, 0x359ab6419ca1091c },
					{ 0xf867241c8cc6d4c0, 0xc30163d203c94b63 },
					{ 0x9b407691d7fc44f8, 0x79e0de63425dcf1e },
					{ 0xc21094364dfb5636, 0x985915fc12f542e5 },
					{ 0xf294b943e17a2bc4, 0x3e6f5b7b17b2939e },
					{ 0x979cf3ca6cec5b5a, 0xa705992ceecf9c43 },
					{ 0xbd8430bd08277231, 0x50c6ff782a838354 },
					{ 0xece53cec4a314ebd, 0xa4f8bf5635246429 },
					{ 0x940f4613ae5ed136, 0x871b7795e136be9a },
					{ 0xb913179899f68584, 0x28e2557b59846e40 },
					{ 0xe757dd7ec07426e5, 0x331aeada2fe589d0 },
					{ 0x9096ea6f3848984f, 0x3ff0d2c85def7622 },
					{ 0xb4bca50b065abe63, 0x0fed077a756b53aa },
					{ 0xe1ebce4dc7f16dfb, 0xd3e8495912c62895 },
					{ 0x8d3360f09cf6e4bd, 0x64712dd7abbbd95d },
					{ 0xb080392cc4349dec, 0xbd8d794d96aacfb4 },
					{ 0xdca04777f541c567, 0xecf0d7a0fc5583a1 },
					{ 0x89e42caaf9491b60, 0xf41686c49db57245 },
					{ 0xac5d37d5b79b6239, 0x311c2875c522ced6 },
					{ 0xd77485cb25823ac7, 0x7d633293366b828c },
					{ 0x86a8d39ef77164bc, 0xae5dff9c02033198 },
					{ 0xa8530886b54dbdeb, 0xd9f57f830283fdfd },
					{ 0xd267caa862a12d66, 0xd072df63c324fd7c },
					{ 0x8380dea93da4bc60, 0x4247cb9e59f71e6e },
					{ 0xa46116538d0deb78, 0x52d9be85f074e609 },
					{ 0xcd795be870516656, 0x67902e276c921f8c },
					{ 0x806bd9714632dff6, 0x00ba1cd8a3db53b7 },
					{ 0xa086cfcd97bf97f3, 0x80e8a40eccd228a5 },
					{ 0xc8a883c0fdaf7df0, 0x6122cd128006b2ce },
					{ 0xfad2a4b13d1b5d6c, 0x796b805720085f82 },
					{ 0x9cc3a6eec6311a63, 0xcbe3303674053bb1 },
					{ 0xc3f490aa77bd60fc, 0xbedbfc4411068a9d },
					{ 0xf4f1b4d515acb93b, 0xee92fb5515482d45 },
					{ 0x991711052d8bf3c5, 0x751bdd152d4d1c4b },
					{ 0xbf5cd54678eef0b6, 0xd262d45a78a0635e },
					{ 0xef340a98172aace4, 0x86fb897116c87c35 },
					{ 0x9580869f0e7aac0e, 0xd45d35e6ae3d4da1 },
					{ 0xbae0a846d2195712, 0x8974836059cca10a },
					{ 0xe998d258869facd7, 0x2bd1a438703fc94c },
					{ 0x91ff83775423cc06, 0x7b6306a34627ddd0 },
					{ 0xb67f6455292cbf08, 0x1a3bc84c17b1d543 },
					{ 0xe41f3d6a7377eeca, 0x20caba5f1d9e4a94 },
					{ 0x8e938662882af53e, 0x547eb47b7282ee9d },
					{ 0xb23867fb2a35b28d, 0xe99e619a4f23aa44 },
					{ 0xdec681f9f4c31f31, 0x6405fa00e2ec94d5 },
					{ 0x8b3c113c38f9f37e, 0xde83bc408dd3dd05 },
					{ 0xae0b158b4738705e, 0x9624ab50b148d446 },
					{ 0xd98ddaee19068c76, 0x3badd624dd9b0958 },
					{ 0x87f8a8d4cfa417c9, 0xe54ca5d70a80e5d7 },
					{ 0xa9f6d30a038d1dbc, 0x5e9fcf4ccd211f4d },
					{ 0xd47487cc8470652b, 0x7647c32000696720 },
					{ 0x84c8d4dfd2c63f3b, 0x29ecd9f40041e074 },
					{ 0xa5fb0a17c777cf09, 0xf468107100525891 },
					{ 0xcf79cc9db955c2cc, 0x7182148d4066eeb5 },
					{ 0x81ac1fe293d599bf, 0xc6f14cd848405531 },
					{ 0xa21727db38cb002f, 0xb8ada00e5a506a7d },
					{ 0xca9cf1d206fdc03b, 0xa6d90811f0e4851d },
					{ 0xfd442e4688bd304a, 0x908f4a166d1da664 },
					{ 0x9e4a9cec15763e2e, 0x9a598e4e043287ff },
					{ 0xc5dd44271ad3cdba, 0x40eff1e1853f29fe },
					{ 0xf7549530e188c128, 0xd12bee59e68ef47d },
					{ 0x9a94dd3e8cf578b9, 0x82bb74f8301958cf },
					{ 0xc13a148e3032d6e7, 0xe36a52363c1faf02 },
					{ 0xf18899b1bc3f8ca1, 0xdc44e6c3cb279ac2 },
					{ 0x96f5600f15a7b7e5, 0x29ab103a5ef8c0ba },
					{ 0xbcb2b812db11a5de, 0x7415d448f6b6f0e8 },
					{ 0xebdf661791d60f56, 0x111b495b3464ad22 },
					{ 0x936b9fcebb25c995, 0xcab10dd900beec35 },
					{ 0xb84687c269ef3bfb, 0x3d5d514f40eea743 },
					{ 0xe65829b3046b0afa, 0x0cb4a5a3112a5113 },
					{ 0x8ff71a0fe2c2e6dc, 0x47f0e785eaba72ac },
					{ 0xb3f4e093db73a093, 0x59ed216765690f57 },
					{ 0xe0f218b8d25088b8, 0x306869c13ec3532d },
					{ 0x8c974f7383725573, 0x1e414218c73a13fc },
					{ 0xafbd2350644eeacf, 0xe5d1929ef90898fb },
					{ 0xdbac6c247d62a583, 0xdf45f746b74abf3a },
					{ 0x894bc396ce5da772, 0x6b8bba8c328eb784 },
					{ 0xab9eb47c81f5114f, 0x066ea92f3f326565 },
					{ 0xd686619ba27255a2, 0xc80a537b0efefebe },
					{ 0x8613fd0145877585, 0xbd06742ce95f5f37 },
					{ 0xa798fc4196e952e7, 0x2c48113823b73705 },
					{ 0xd17f3b51fca3a7a0, 0xf75a15862ca504c6 },
					{ 0x82ef85133de648c4, 0x9a984d73dbe722fc },
					{ 0xa3ab66580d5fdaf5, 0xc13e60d0d2e0ebbb },
					{ 0xcc963fee10b7d1b3, 0x318df905079926a9 },
					{ 0xffbbcfe994e5c61f, 0xfdf17746497f7053 },
					{ 0x9fd561f1fd0f9bd3, 0xfeb6ea8bedefa634 },
					{ 0xc7caba6e7c5382c8, 0xfe64a52ee96b8fc1 },
					{ 0xf9bd690a1b68637b, 0x3dfdce7aa3c673b1 },
					{ 0x9c1661a651213e2d, 0x06bea10ca65c084f },
					{ 0xc31bfa0fe5698db8, 0x486e494fcff30a63 },
					{ 0xf3e2f893dec3f126, 0x5a89dba3c3efccfb },
					{ 0x986ddb5c6b3a76b7, 0xf89629465a75e01d },
					{ 0xbe89523386091465, 0xf6bbb397f1135824 },
					{ 0xee2ba6c0678b597f, 0x746aa07ded582e2d },
					{ 0x94db483840b717ef, 0xa8c2a44eb4571cdd },
					{ 0xba121a4650e4ddeb, 0x92f34d62616ce414 },
					{ 0xe896a0d7e51e1566, 0x77b020baf9c81d18 },
					{ 0x915e2486ef32cd60, 0x0ace1474dc1d122f },
					{ 0xb5b5ada8aaff80b8, 0x0d819992132456bb },
					{ 0xe3231912d5bf60e6, 0x10e1fff697ed6c6a },
					{ 0x8df5efabc5979c8f, 0xca8d3ffa1ef463c2 },
					{ 0xb1736b96b6fd83b3, 0xbd308ff8a6b17cb3 },
					{ 0xddd0467c64bce4a0, 0xac7cb3f6d05ddbdf },
					{ 0x8aa22c0dbef60ee4, 0x6bcdf07a423aa96c },
					{ 0xad4ab7112eb3929d, 0x86c16c98d2c953c7 },
					{ 0xd89d64d57a607744, 0xe871c7bf077ba8b8 },
					{ 0x87625f056c7c4a8b, 0x11471cd764ad4973 },
					{ 0xa93af6c6c79b5d2d, 0xd598e40d3dd89bd0 },
					{ 0xd389b47879823479, 0x4aff1d108d4ec2c4 },
					{ 0x843610cb4bf160cb, 0xcedf722a585139bb },
					{ 0xa54394fe1eedb8fe, 0xc2974eb4ee658829 },
					{ 0xce947a3da6a9273e, 0x733d226229feea33 },
					{ 0x811ccc668829b887, 0x0806357d5a3f5260 },
					{ 0xa163ff802a3426a8, 0xca07c2dcb0cf26f8 },
					{ 0xc9bcff6034c13052, 0xfc89b393dd02f0b6 },
					{ 0xfc2c3f3841f17c67, 0xbbac2078d443ace3 },
					{ 0x9d9ba7832936edc0, 0xd54b944b84aa4c0e },
					{ 0xc5029163f384a931, 0x0a9e795e65d4df12 },
					{ 0xf64335bcf065d37d, 0x4d4617b5ff4a16d6 },
					{ 0x99ea0196163fa42e, 0x504bced1bf8e4e46 },
					{ 0xc06481fb9bcf8d39, 0xe45ec2862f71e1d7 },
					{ 0xf07da27a82c37088, 0x5d767327bb4e5a4d },
					{ 0x964e858c91ba2655, 0x3a6a07f8d510f870 },
					{ 0xbbe226efb628afea, 0x890489f70a55368c },
					{ 0xeadab0aba3b2dbe5, 0x2b45ac74ccea842f },
					{ 0x92c8ae6b464fc96f, 0x3b0b8bc90012929e },
					{ 0xb77ada0617e3bbcb, 0x09ce6ebb40173745 },
					{ 0xe55990879ddcaabd, 0xcc420a6a101d0516 },
					{ 0x8f57fa54c2a9eab6, 0x9fa946824a12232e },
					{ 0xb32df8e9f3546564, 0x47939822dc96abfa },
					{ 0xdff9772470297ebd, 0x59787e2b93bc56f8 },
					{ 0x8bfbea76c619ef36, 0x57eb4edb3c55b65b },
					{ 0xaefae51477a06b03, 0xede622920b6b23f2 },
					{ 0xdab99e59958885c4, 0xe95fab368e45ecee },
					{ 0x88b402f7fd75539b, 0x11dbcb0218ebb415 },
					{ 0xaae103b5fcd2a881, 0xd652bdc29f26a11a },
					{ 0xd59944a37c0752a2, 0x4be76d3346f04960 },
					{ 0x857fcae62d8493a5, 0x6f70a4400c562ddc },
					{ 0xa6dfbd9fb8e5b88e, 0xcb4ccd500f6bb953 },
					{ 0xd097ad07a71f26b2, 0x7e2000a41346a7a8 },
					{ 0x825ecc24c873782f, 0x8ed400668c0c28c9 },
					{ 0xa2f67f2dfa90563b, 0x728900802f0f32fb },
					{ 0xcbb41ef979346bca, 0x4f2b40a03ad2ffba },
					{ 0xfea126b7d78186bc, 0xe2f610c84987bfa9 },
					{ 0x9f24b832e6b0f436, 0x0dd9ca7d2df4d7ca },
					{ 0xc6ede63fa05d3143, 0x91503d1c79720dbc },
					{ 0xf8a95fcf88747d94, 0x75a44c6397ce912b },
					{ 0x9b69dbe1b548ce7c, 0xc986afbe3ee11abb },
					{ 0xc24452da229b021b, 0xfbe85badce996169 },
					{ 0xf2d56790ab41c2a2, 0xfae27299423fb9c4 },
					{ 0x97c560ba6b0919a5, 0xdccd879fc967d41b },
					{ 0xbdb6b8e905cb600f, 0x5400e987bbc1c921 },
					{ 0xed246723473e3813, 0x290123e9aab23b69 },
					{ 0x9436c0760c86e30b, 0xf9a0b6720aaf6522 },
					{ 0xb94470938fa89bce, 0xf808e40e8d5b3e6a },
					{ 0xe7958cb87392c2c2, 0xb60b1d1230b20e05 },
					{ 0x90bd77f3483bb9b9, 0xb1c6f22b5e6f48c3 },
					{ 0xb4ecd5f01a4aa828, 0x1e38aeb6360b1af4 },
					{ 0xe2280b6c20dd5232, 0x25c6da63c38de1b1 },
					{ 0x8d590723948a535f, 0x579c487e5a38ad0f },
					{ 0xb0af48ec79ace837, 0x2d835a9df0c6d852 },
					{ 0xdcdb1b2798182244, 0xf8e431456cf88e66 },
					{ 0x8a08f0f8bf0f156b, 0x1b8e9ecb641b5900 },
					{ 0xac8b2d36eed2dac5, 0xe272467e3d222f40 },
					{ 0xd7adf884aa879177, 0x5b0ed81dcc6abb10 },
					{ 0x86ccbb52ea94baea, 0x98e947129fc2b4ea },
					{ 0xa87fea27a539e9a5, 0x3f2398d747b36225 },
					{ 0xd29fe4b18e88640e, 0x8eec7f0d19a03aae },
					{ 0x83a3eeeef9153e89, 0x1953cf68300424ad },
					{ 0xa48ceaaab75a8e2b, 0x5fa8c3423c052dd8 },
					{ 0xcdb02555653131b6, 0x3792f412cb06794e },
					{ 0x808e17555f3ebf11, 0xe2bbd88bbee40bd1 },
					{ 0xa0b19d2ab70e6ed6, 0x5b6aceaeae9d0ec5 },
					{ 0xc8de047564d20a8b, 0xf245825a5a445276 },
					{ 0xfb158592be068d2e, 0xeed6e2f0f0d56713 },
					{ 0x9ced737bb6c4183d, 0x55464dd69685606c },
					{ 0xc428d05aa4751e4c, 0xaa97e14c3c26b887 },
					{ 0xf53304714d9265df, 0xd53dd99f4b3066a9 },
					{ 0x993fe2c6d07b7fab, 0xe546a8038efe402a },
					{ 0xbf8fdb78849a5f96, 0xde98520472bdd034 },
					{ 0xef73d256a5c0f77c, 0x963e66858f6d4441 },
					{ 0x95a8637627989aad, 0xdde7001379a44aa9 },
					{ 0xbb127c53b17ec159, 0x5560c018580d5d53 },
					{ 0xe9d71b689dde71af, 0xaab8f01e6e10b4a7 },
					{ 0x9226712162ab070d, 0xcab3961304ca70e9 },
					{ 0xb6b00d69bb55c8d1, 0x3d607b97c5fd0d23 },
					{ 0xe45c10c42a2b3b05, 0x8cb89a7db77c506b },
					{ 0x8eb98a7a9a5b04e3, 0x77f3608e92adb243 },
					{ 0xb267ed1940f1c61c, 0x55f038b237591ed4 },
					{ 0xdf01e85f912e37a3, 0x6b6c46dec52f6689 },
					{ 0x8b61313bbabce2c6, 0x2323ac4b3b3da016 },
					{ 0xae397d8aa96c1b77, 0xabec975e0a0d081b },
					{ 0xd9c7dced53c72255, 0x96e7bd358c904a22 },
					{ 0x881cea14545c7575, 0x7e50d64177da2e55 },
					{ 0xaa242499697392d2, 0xdde50bd1d5d0b9ea },
					{ 0xd4ad2dbfc3d07787, 0x955e4ec64b44e865 },
					{ 0x84ec3c97da624ab4, 0xbd5af13bef0b113f },
					{ 0xa6274bbdd0fadd61, 0xecb1ad8aeacdd58f },
					{ 0xcfb11ead453994ba, 0x67de18eda5814af3 },
					{ 0x81ceb32c4b43fcf4, 0x80eacf948770ced8 },
					{ 0xa2425ff75e14fc31, 0xa1258379a94d028e },
					{ 0xcad2f7f5359a3b3e, 0x096ee45813a04331 },
					{ 0xfd87b5f28300ca0d, 0x8bca9d6e188853fd },
					{ 0x9e74d1b791e07e48, 0x775ea264cf55347e },
					{ 0xc612062576589dda, 0x95364afe032a819e },
					{ 0xf79687aed3eec551, 0x3a83ddbd83f52205 },
					{ 0x9abe14cd44753b52, 0xc4926a9672793543 },
					{ 0xc16d9a0095928a27, 0x75b7053c0f178294 },
					{ 0xf1c90080baf72cb1, 0x5324c68b12dd6339 },
					{ 0x971da05074da7bee, 0xd3f6fc16ebca5e04 },
					{ 0xbce5086492111aea, 0x88f4bb1ca6bcf585 },
					{ 0xec1e4a7db69561a5, 0x2b31e9e3d06c32e6 },
					{ 0x9392ee8e921d5d07, 0x3aff322e62439fd0 },
					{ 0xb877aa3236a4b449, 0x09befeb9fad487c3 },
					{ 0xe69594bec44de15b, 0x4c2ebe687989a9b4 },
					{ 0x901d7cf73ab0acd9, 0x0f9d37014bf60a11 },
					{ 0xb424dc35095cd80f, 0x538484c19ef38c95 },
					{ 0xe12e13424bb40e13, 0x2865a5f206b06fba },
					{ 0x8cbccc096f5088cb, 0xf93f87b7442e45d4 },
					{ 0xafebff0bcb24aafe, 0xf78f69a51539d749 },
					{ 0xdbe6fecebdedd5be, 0xb573440e5a884d1c },
					{ 0x89705f4136b4a597, 0x31680a88f8953031 },
					{ 0xabcc77118461cefc, 0xfdc20d2b36ba7c3e },
					{ 0xd6bf94d5e57a42bc, 0x3d32907604691b4d },
					{ 0x8637bd05af6c69b5, 0xa63f9a49c2c1b110 },
					{ 0xa7c5ac471b478423, 0x0fcf80dc33721d54 },
					{ 0xd1b71758e219652b, 0xd3c36113404ea4a9 },
					{ 0x83126e978d4fdf3b, 0x645a1cac083126ea },
					{ 0xa3d70a3d70a3d70a, 0x3d70a3d70a3d70a4 },
					{ 0xcccccccccccccccc, 0xcccccccccccccccd },
					{ 0x8000000000000000, 0x0000000000000000 },
					{ 0xa000000000000000, 0x0000000000000000 },
					{ 0xc800000000000000, 0x0000000000000000 },
					{ 0xfa00000000000000, 0x0000000000000000 },
					{ 0x9c40000000000000, 0x0000000000000000 },
					{ 0xc350000000000000, 0x0000000000000000 },
					{ 0xf424000000000000, 0x0000000000000000 },
					{ 0x9896800000000000, 0x0000000000000000 },
					{ 0xbebc200000000000, 0x0000000000000000 },
					{ 0xee6b280000000000, 0x0000000000000000 },
					{ 0x9502f90000000000, 0x0000000000000000 },
					{ 0xba43b74000000000, 0x0000000000000000 },
					{ 0xe8d4a51000000000, 0x0000000000000000 },
					{ 0x9184e72a00000000, 0x0000000000000000 },
					{ 0xb5e620f480000000, 0x0000000000000000 },
					{ 0xe35fa931a0000000, 0x0000000000000000 },
					{ 0x8e1bc9bf04000000, 0x0000000000000000 },
					{ 0xb1a2bc2ec5000000, 0x0000000000000000 },
					{ 0xde0b6b3a76400000, 0x0000000000000000 },
					{ 0x8ac7230489e80000, 0x0000000000000000 },
					{ 0xad78ebc5ac620000, 0x0000000000000000 },
					{ 0xd8d726b7177a8000, 0x0000000000000000 },
					{ 0x878678326eac9000, 0x0000000000000000 },
					{ 0xa968163f0a57b400, 0x0000000000000000 },
					{ 0xd3c21bcecceda100, 0x0000000000000000 },
					{ 0x84595161401484a0, 0x0000000000000000 },
					{ 0xa56fa5b99019a5c8, 0x0000000000000000 },
					{ 0xcecb8f27f4200f3a, 0x0000000000000000 },
					{ 0x813f3978f8940984, 0x4000000000000000 },
					{ 0xa18f07d736b90be5, 0x5000000000000000 },
					{ 0xc9f2c9cd04674ede, 0xa400000000000000 },
					{ 0xfc6f7c4045812296, 0x4d00000000000000 },
					{ 0x9dc5ada82b70b59d, 0xf020000000000000 },
					{ 0xc5371912364ce305, 0x6c28000000000000 },
					{ 0xf684df56c3e01bc6, 0xc732000000000000 },
					{ 0x9a130b963a6c115c, 0x3c7f400000000000 },
					{ 0xc097ce7bc90715b3, 0x4b9f100000000000 },
					{ 0xf0bdc21abb48db20, 0x1e86d40000000000 },
					{ 0x96769950b50d88f4, 0x1314448000000000 },
					{ 0xbc143fa4e250eb31, 0x17d955a000000000 },
					{ 0xeb194f8e1ae525fd, 0x5dcfab0800000000 },
					{ 0x92efd1b8d0cf37be, 0x5aa1cae500000000 },
					{ 0xb7abc627050305ad, 0xf14a3d9e40000000 },
					{ 0xe596b7b0c643c719, 0x6d9ccd05d0000000 },
					{ 0x8f7e32ce7bea5c6f, 0xe4820023a2000000 },
					{ 0xb35dbf821ae4f38b, 0xdda2802c8a800000 },
					{ 0xe0352f62a19e306e, 0xd50b2037ad200000 },
					{ 0x8c213d9da502de45, 0x4526f422cc340000 },
					{ 0xaf298d050e4395d6, 0x9670b12b7f410000 },
					{ 0xdaf3f04651d47b4c, 0x3c0cdd765f114000 },
					{ 0x88d8762bf324cd0f, 0xa5880a69fb6ac800 },
					{ 0xab0e93b6efee0053, 0x8eea0d047a457a00 },
					{ 0xd5d238a4abe98068, 0x72a4904598d6d880 },
					{ 0x85a36366eb71f041, 0x47a6da2b7f864750 },
					{ 0xa70c3c40a64e6c51, 0x999090b65f67d924 },
					{ 0xd0cf4b50cfe20765, 0xfff4b4e3f741cf6d },
					{ 0x82818f1281ed449f, 0xbff8f10e7a8921a4 },
					{ 0xa321f2d7226895c7, 0xaff72d52192b6a0d },
					{ 0xcbea6f8ceb02bb39, 0x9bf4f8a69f764490 },
					{ 0xfee50b7025c36a08, 0x02f236d04753d5b4 },
					{ 0x9f4f2726179a2245, 0x01d762422c946590 },
					{ 0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef5 },
					{ 0xf8ebad2b84e0d58b, 0xd2e0898765a7deb2 },
					{ 0x9b934c3b330c8577, 0x63cc55f49f88eb2f },
					{ 0xc2781f49ffcfa6d5, 0x3cbf6b71c76b25fb },
					{ 0xf316271c7fc3908a, 0x8bef464e3945ef7a },
					{ 0x97edd871cfda3a56, 0x97758bf0e3cbb5ac },
					{ 0xbde94e8e43d0c8ec, 0x3d52eeed1cbea317 },
					{ 0xed63a231d4c4fb27, 0x4ca7aaa863ee4bdd },
					{ 0x945e455f24fb1cf8, 0x8fe8caa93e74ef6a },
					{ 0xb975d6b6ee39e436, 0xb3e2fd538e122b44 },
					{ 0xe7d34c64a9c85d44, 0x60dbbca87196b616 },
					{ 0x90e40fbeea1d3a4a, 0xbc8955e946fe31cd },
					{ 0xb51d13aea4a488dd, 0x6babab6398bdbe41 },
					{ 0xe264589a4dcdab14, 0xc696963c7eed2dd1 },
					{ 0x8d7eb76070a08aec, 0xfc1e1de5cf543ca2 },
					{ 0xb0de65388cc8ada8, 0x3b25a55f43294bcb },
					{ 0xdd15fe86affad912, 0x49ef0eb713f39ebe },
					{ 0x8a2dbf142dfcc7ab, 0x6e3569326c784337 },
					{ 0xacb92ed9397bf996, 0x49c2c37f07965404 },
					{ 0xd7e77a8f87daf7fb, 0xdc33745ec97be906 },
					{ 0x86f0ac99b4e8dafd, 0x69a028bb3ded71a3 },
					{ 0xa8acd7c0222311bc, 0xc40832ea0d68ce0c },
					{ 0xd2d80db02aabd62b, 0xf50a3fa490c30190 },
					{ 0x83c7088e1aab65db, 0x792667c6da79e0fa },
					{ 0xa4b8cab1a1563f52, 0x577001b891185938 },
					{ 0xcde6fd5e09abcf26, 0xed4c0226b55e6f86 },
					{ 0x80b05e5ac60b6178, 0x544f8158315b05b4 },
					{ 0xa0dc75f1778e39d6, 0x696361ae3db1c721 },
					{ 0xc913936dd571c84c, 0x03bc3a19cd1e38e9 },
					{ 0xfb5878494ace3a5f, 0x04ab48a04065c723 },
					{ 0x9d174b2dcec0e47b, 0x62eb0d64283f9c76 },
					{ 0xc45d1df942711d9a, 0x3ba5d0bd324f8394 },
					{ 0xf5746577930d6500, 0xca8f44ec7ee36479 },
					{ 0x9968bf6abbe85f20, 0x7e998b13cf4e1ecb },
					{ 0xbfc2ef456ae276e8, 0x9e3fedd8c321a67e },
					{ 0xefb3ab16c59b14a2, 0xc5cfe94ef3ea101e },
					{ 0x95d04aee3b80ece5, 0xbba1f1d158724a12 },
					{ 0xbb445da9ca61281f, 0x2a8a6e45ae8edc97 },
					{ 0xea1575143cf97226, 0xf52d09d71a3293bd },
					{ 0x924d692ca61be758, 0x593c2626705f9c56 },
					{ 0xb6e0c377cfa2e12e, 0x6f8b2fb00c77836c },
					{ 0xe498f455c38b997a, 0x0b6dfb9c0f956447 },
					{ 0x8edf98b59a373fec, 0x4724bd4189bd5eac },
					{ 0xb2977ee300c50fe7, 0x58edec91ec2cb657 },
					{ 0xdf3d5e9bc0f653e1, 0x2f2967b66737e3ed },
					{ 0x8b865b215899f46c, 0xbd79e0d20082ee74 },
					{ 0xae67f1e9aec07187, 0xecd8590680a3aa11 },
					{ 0xda01ee641a708de9, 0xe80e6f4820cc9495 },
					{ 0x884134fe908658b2, 0x3109058d147fdcdd },
					{ 0xaa51823e34a7eede, 0xbd4b46f0599fd415 },
					{ 0xd4e5e2cdc1d1ea96, 0x6c9e18ac7007c91a },
					{ 0x850fadc09923329e, 0x03e2cf6bc604ddb0 },
					{ 0xa6539930bf6bff45, 0x84db8346b786151c },
					{ 0xcfe87f7cef46ff16, 0xe612641865679a63 },
					{ 0x81f14fae158c5f6e, 0x4fcb7e8f3f60c07e },
					{ 0xa26da3999aef7749, 0xe3be5e330f38f09d },
					{ 0xcb090c8001ab551c, 0x5cadf5bfd3072cc5 },
					{ 0xfdcb4fa002162a63, 0x73d9732fc7c8f7f6 },
					{ 0x9e9f11c4014dda7e, 0x2867e7fddcdd9afa },
					{ 0xc646d63501a1511d, 0xb281e1fd541501b8 },
					{ 0xf7d88bc24209a565, 0x1f225a7ca91a4226 },
					{ 0x9ae757596946075f, 0x3375788de9b06958 },
					{ 0xc1a12d2fc3978937, 0x0052d6b1641c83ae },
					{ 0xf209787bb47d6b84, 0xc0678c5dbd23a49a },
					{ 0x9745eb4d50ce6332, 0xf840b7ba963646e0 },
					{ 0xbd176620a501fbff, 0xb650e5a93bc3d898 },
					{ 0xec5d3fa8ce427aff, 0xa3e51f138ab4cebe },
					{ 0x93ba47c980e98cdf, 0xc66f336c36b10137 },
					{ 0xb8a8d9bbe123f017, 0xb80b0047445d4184 },
					{ 0xe6d3102ad96cec1d, 0xa60dc059157491e5 },
					{ 0x9043ea1ac7e41392, 0x87c89837ad68db2f },
					{ 0xb454e4a179dd1877, 0x29babe4598c311fb },
					{ 0xe16a1dc9d8545e94, 0xf4296dd6fef3d67a },
					{ 0x8ce2529e2734bb1d, 0x1899e4a65f58660c },
					{ 0xb01ae745b101e9e4, 0x5ec05dcff72e7f8f },
					{ 0xdc21a1171d42645d, 0x76707543f4fa1f73 },
					{ 0x899504ae72497eba, 0x6a06494a791c53a8 },
					{ 0xabfa45da0edbde69, 0x0487db9d17636892 },
					{ 0xd6f8d7509292d603, 0x45a9d2845d3c42b6 },
					{ 0x865b86925b9bc5c2, 0x0b8a2392ba45a9b2 },
					{ 0xa7f26836f282b732, 0x8e6cac7768d7141e },
					{ 0xd1ef0244af2364ff, 0x3207d795430cd926 },
					{ 0x8335616aed761f1f, 0x7f44e6bd49e807b8 },
					{ 0xa402b9c5a8d3a6e7, 0x5f16206c9c6209a6 },
					{ 0xcd036837130890a1, 0x36dba887c37a8c0f },
					{ 0x802221226be55a64, 0xc2494954da2c9789 },
					{ 0xa02aa96b06deb0fd, 0xf2db9baa10b7bd6c },
					{ 0xc83553c5c8965d3d, 0x6f92829494e5acc7 },
					{ 0xfa42a8b73abbf48c, 0xcb772339ba1f17f9 },
					{ 0x9c69a97284b578d7, 0xff2a760414536efb },
					{ 0xc38413cf25e2d70d, 0xfef5138519684aba },
					{ 0xf46518c2ef5b8cd1, 0x7eb258665fc25d69 },
					{ 0x98bf2f79d5993802, 0xef2f773ffbd97a61 },
					{ 0xbeeefb584aff8603, 0xaafb550ffacfd8fa },
					{ 0xeeaaba2e5dbf6784, 0x95ba2a53f983cf38 },
					{ 0x952ab45cfa97a0b2, 0xdd945a747bf26183 },
					{ 0xba756174393d88df, 0x94f971119aeef9e4 },
					{ 0xe912b9d1478ceb17, 0x7a37cd5601aab85d },
					{ 0x91abb422ccb812ee, 0xac62e055c10ab33a },
					{ 0xb616a12b7fe617aa, 0x577b986b314d6009 },
					{ 0xe39c49765fdf9d94, 0xed5a7e85fda0b80b },
					{ 0x8e41ade9fbebc27d, 0x14588f13be847307 },
					{ 0xb1d219647ae6b31c, 0x596eb2d8ae258fc8 },
					{ 0xde469fbd99a05fe3, 0x6fca5f8ed9aef3bb },
					{ 0x8aec23d680043bee, 0x25de7bb9480d5854 },
					{ 0xada72ccc20054ae9, 0xaf561aa79a10ae6a },
					{ 0xd910f7ff28069da4, 0x1b2ba1518094da04 },
					{ 0x87aa9aff79042286, 0x90fb44d2f05d0842 },
					{ 0xa99541bf57452b28, 0x353a1607ac744a53 },
					{ 0xd3fa922f2d1675f2, 0x42889b8997915ce8 },
					{ 0x847c9b5d7c2e09b7, 0x69956135febada11 },
					{ 0xa59bc234db398c25, 0x43fab9837e699095 },
					{ 0xcf02b2c21207ef2e, 0x94f967e45e03f4bb },
					{ 0x8161afb94b44f57d, 0x1d1be0eebac278f5 },
					{ 0xa1ba1ba79e1632dc, 0x6462d92a69731732 },
					{ 0xca28a291859bbf93, 0x7d7b8f7503cfdcfe },
					{ 0xfcb2cb35e702af78, 0x5cda735244c3d43e },
					{ 0x9defbf01b061adab, 0x3a0888136afa64a7 },
					{ 0xc56baec21c7a1916, 0x088aaa1845b8fdd0 },
					{ 0xf6c69a72a3989f5b, 0x8aad549e57273d45 },
					{ 0x9a3c2087a63f6399, 0x36ac54e2f678864b },
					{ 0xc0cb28a98fcf3c7f, 0x84576a1bb416a7dd },
					{ 0xf0fdf2d3f3c30b9f, 0x656d44a2a11c51d5 },
					{ 0x969eb7c47859e743, 0x9f644ae5a4b1b325 },
					{ 0xbc4665b596706114, 0x873d5d9f0dde1fee },
					{ 0xeb57ff22fc0c7959, 0xa90cb506d155a7ea },
					{ 0x9316ff75dd87cbd8, 0x09a7f12442d588f2 },
					{ 0xb7dcbf5354e9bece, 0x0c11ed6d538aeb2f },
					{ 0xe5d3ef282a242e81, 0x8f1668c8a86da5fa },
					{ 0x8fa475791a569d10, 0xf96e017d694487bc },
					{ 0xb38d92d760ec4455, 0x37c981dcc395a9ac },
					{ 0xe070f78d3927556a, 0x85bbe253f47b1417 },
					{ 0x8c469ab843b89562, 0x93956d7478ccec8e },
					{ 0xaf58416654a6babb, 0x387ac8d1970027b2 },
					{ 0xdb2e51bfe9d0696a, 0x06997b05fcc0319e },
					{ 0x88fcf317f22241e2, 0x441fece3bdf81f03 },
					{ 0xab3c2fddeeaad25a, 0xd527e81cad7626c3 },
					{ 0xd60b3bd56a5586f1, 0x8a71e223d8d3b074 },
					{ 0x85c7056562757456, 0xf6872d5667844e49 },
					{ 0xa738c6bebb12d16c, 0xb428f8ac016561db },
					{ 0xd106f86e69d785c7, 0xe13336d701beba52 },
					{ 0x82a45b450226b39c, 0xecc0024661173473 },
					{ 0xa34d721642b06084, 0x27f002d7f95d0190 },
					{ 0xcc20ce9bd35c78a5, 0x31ec038df7b441f4 },
					{ 0xff290242c83396ce, 0x7e67047175a15271 },
					{ 0x9f79a169bd203e41, 0x0f0062c6e984d386 },
					{ 0xc75809c42c684dd1, 0x52c07b78a3e60868 },
					{ 0xf92e0c3537826145, 0xa7709a56ccdf8a82 },
					{ 0x9bbcc7a142b17ccb, 0x88a66076400bb691 },
					{ 0xc2abf989935ddbfe, 0x6acff893d00ea435 },
					{ 0xf356f7ebf83552fe, 0x0583f6b8c4124d43 },
					{ 0x98165af37b2153de, 0xc3727a337a8b704a },
					{ 0xbe1bf1b059e9a8d6, 0x744f18c0592e4c5c },
					{ 0xeda2ee1c7064130c, 0x1162def06f79df73 },
					{ 0x9485d4d1c63e8be7, 0x8addcb5645ac2ba8 },
					{ 0xb9a74a0637ce2ee1, 0x6d953e2bd7173692 },
					{ 0xe8111c87c5c1ba99, 0xc8fa8db6ccdd0437 },
					{ 0x910ab1d4db9914a0, 0x1d9c9892400a22a2 },
					{ 0xb54d5e4a127f59c8, 0x2503beb6d00cab4b },
					{ 0xe2a0b5dc971f303a, 0x2e44ae64840fd61d },
					{ 0x8da471a9de737e24, 0x5ceaecfed289e5d2 },
					{ 0xb10d8e1456105dad, 0x7425a83e872c5f47 },
					{ 0xdd50f1996b947518, 0xd12f124e28f77719 },
					{ 0x8a5296ffe33cc92f, 0x82bd6b70d99aaa6f },
					{ 0xace73cbfdc0bfb7b, 0x636cc64d1001550b },
					{ 0xd8210befd30efa5a, 0x3c47f7e05401aa4e },
					{ 0x8714a775e3e95c78, 0x65acfaec34810a71 },
					{ 0xa8d9d1535ce3b396, 0x7f1839a741a14d0d },
					{ 0xd31045a8341ca07c, 0x1ede48111209a050 },
					{ 0x83ea2b892091e44d, 0x934aed0aab460432 },
					{ 0xa4e4b66b68b65d60, 0xf81da84d5617853f },
					{ 0xce1de40642e3f4b9, 0x36251260ab9d668e },
					{ 0x80d2ae83e9ce78f3, 0xc1d72b7c6b426019 },
					{ 0xa1075a24e4421730, 0xb24cf65b8612f81f },
					{ 0xc94930ae1d529cfc, 0xdee033f26797b627 },
					{ 0xfb9b7cd9a4a7443c, 0x169840ef017da3b1 },
					{ 0x9d412e0806e88aa5, 0x8e1f289560ee864e },
					{ 0xc491798a08a2ad4e, 0xf1a6f2bab92a27e2 },
					{ 0xf5b5d7ec8acb58a2, 0xae10af696774b1db },
					{ 0x9991a6f3d6bf1765, 0xacca6da1e0a8ef29 },
					{ 0xbff610b0cc6edd3f, 0x17fd090a58d32af3 },
					{ 0xeff394dcff8a948e, 0xddfc4b4cef07f5b0 },
					{ 0x95f83d0a1fb69cd9, 0x4abdaf101564f98e },
					{ 0xbb764c4ca7a4440f, 0x9d6d1ad41abe37f1 },
					{ 0xea53df5fd18d5513, 0x84c86189216dc5ed },
					{ 0x92746b9be2f8552c, 0x32fd3cf5b4e49bb4 },
					{ 0xb7118682dbb66a77, 0x3fbc8c33221dc2a1 },
					{ 0xe4d5e82392a40515, 0x0fabaf3feaa5334a },
					{ 0x8f05b1163ba6832d, 0x29cb4d87f2a7400e },
					{ 0xb2c71d5bca9023f8, 0x743e20e9ef511012 },
					{ 0xdf78e4b2bd342cf6, 0x914da9246b255416 },
					{ 0x8bab8eefb6409c1a, 0x1ad089b6c2f7548e },
					{ 0xae9672aba3d0c320, 0xa184ac2473b529b1 },
					{ 0xda3c0f568cc4f3e8, 0xc9e5d72d90a2741e },
					{ 0x8865899617fb1871, 0x7e2fa67c7a658892 },
					{ 0xaa7eebfb9df9de8d, 0xddbb901b98feeab7 },
					{ 0xd51ea6fa85785631, 0x552a74227f3ea565 },
					{ 0x8533285c936b35de, 0xd53a88958f87275f },
					{ 0xa67ff273b8460356, 0x8a892abaf368f137 },
					{ 0xd01fef10a657842c, 0x2d2b7569b0432d85 },
					{ 0x8213f56a67f6b29b, 0x9c3b29620e29fc73 },
					{ 0xa298f2c501f45f42, 0x8349f3ba91b47b8f },
					{ 0xcb3f2f7642717713, 0x241c70a936219a73 },
					{ 0xfe0efb53d30dd4d7, 0xed238cd383aa0110 },
					{ 0x9ec95d1463e8a506, 0xf4363804324a40aa },
					{ 0xc67bb4597ce2ce48, 0xb143c6053edcd0d5 },
					{ 0xf81aa16fdc1b81da, 0xdd94b7868e94050a },
					{ 0x9b10a4e5e9913128, 0xca7cf2b4191c8326 },
					{ 0xc1d4ce1f63f57d72, 0xfd1c2f611f63a3f0 },
					{ 0xf24a01a73cf2dccf, 0xbc633b39673c8cec },
					{ 0x976e41088617ca01, 0xd5be0503e085d813 },
					{ 0xbd49d14aa79dbc82, 0x4b2d8644d8a74e18 },
					{ 0xec9c459d51852ba2, 0xddf8e7d60ed1219e },
					{ 0x93e1ab8252f33b45, 0xcabb90e5c942b503 },
					{ 0xb8da1662e7b00a17, 0x3d6a751f3b936243 },
					{ 0xe7109bfba19c0c9d, 0x0cc512670a783ad4 },
					{ 0x906a617d450187e2, 0x27fb2b80668b24c5 },
					{ 0xb484f9dc9641e9da, 0xb1f9f660802dedf6 },
					{ 0xe1a63853bbd26451, 0x5e7873f8a0396973 },
					{ 0x8d07e33455637eb2, 0xdb0b487b6423e1e8 },
					{ 0xb049dc016abc5e5f, 0x91ce1a9a3d2cda62 },
					{ 0xdc5c5301c56b75f7, 0x7641a140cc7810fb },
					{ 0x89b9b3e11b6329ba, 0xa9e904c87fcb0a9d },
					{ 0xac2820d9623bf429, 0x546345fa9fbdcd44 },
					{ 0xd732290fbacaf133, 0xa97c177947ad4095 },
					{ 0x867f59a9d4bed6c0, 0x49ed8eabcccc485d },
					{ 0xa81f301449ee8c70, 0x5c68f256bfff5a74 },
					{ 0xd226fc195c6a2f8c, 0x73832eec6fff3111 },
					{ 0x83585d8fd9c25db7, 0xc831fd53c5ff7eab },
					{ 0xa42e74f3d032f525, 0xba3e7ca8b77f5e55 },
					{ 0xcd3a1230c43fb26f, 0x28ce1bd2e55f35eb },
					{ 0x80444b5e7aa7cf85, 0x7980d163cf5b81b3 },
					{ 0xa0555e361951c366, 0xd7e105bcc332621f },
					{ 0xc86ab5c39fa63440, 0x8dd9472bf3fefaa7 },
					{ 0xfa856334878fc150, 0xb14f98f6f0feb951 },
					{ 0x9c935e00d4b9d8d2, 0x6ed1bf9a569f33d3 },
					{ 0xc3b8358109e84f07, 0x0a862f80ec4700c8 },
					{ 0xf4a642e14c6262c8, 0xcd27bb612758c0fa },
					{ 0x98e7e9cccfbd7dbd, 0x8038d51cb897789c },
					{ 0xbf21e44003acdd2c, 0xe0470a63e6bd56c3 },
					{ 0xeeea5d5004981478, 0x1858ccfce06cac74 },
					{ 0x95527a5202df0ccb, 0x0f37801e0c43ebc8 },
					{ 0xbaa718e68396cffd, 0xd30560258f54e6ba },
					{ 0xe950df20247c83fd, 0x47c6b82ef32a2069 },
					{ 0x91d28b7416cdd27e, 0x4cdc331d57fa5441 },
					{ 0xb6472e511c81471d, 0xe0133fe4adf8e952 },
					{ 0xe3d8f9e563a198e5, 0x58180fddd97723a6 },
					{ 0x8e679c2f5e44ff8f, 0x570f09eaa7ea7648 },
					{ 0xb201833b35d63f73, 0x2cd2cc6551e513da },
					{ 0xde81e40a034bcf4f, 0xf8077f7ea65e58d1 },
					{ 0x8b112e86420f6191, 0xfb04afaf27faf782 },
					{ 0xadd57a27d29339f6, 0x79c5db9af1f9b563 },
					{ 0xd94ad8b1c7380874, 0x18375281ae7822bc },
					{ 0x87cec76f1c830548, 0x8f2293910d0b15b5 },
					{ 0xa9c2794ae3a3c69a, 0xb2eb3875504ddb22 },
					{ 0xd433179d9c8cb841, 0x5fa60692a46151eb },
					{ 0x849feec281d7f328, 0xdbc7c41ba6bcd333 },
					{ 0xa5c7ea73224deff3, 0x12b9b522906c0800 },
					{ 0xcf39e50feae16bef, 0xd768226b34870a00 },
					{ 0x81842f29f2cce375, 0xe6a1158300d46640 },
					{ 0xa1e53af46f801c53, 0x60495ae3c1097fd0 },
					{ 0xca5e89b18b602368, 0x385bb19cb14bdfc4 },
					{ 0xfcf62c1dee382c42, 0x46729e03dd9ed7b5 },
					{ 0x9e19db92b4e31ba9, 0x6c07a2c26a8346d1 },
					{ 0xc5a05277621be293, 0xc7098b7305241885 },
					{ 0xf70867153aa2db38, 0xb8cbee4fc66d1ea7 }
				};
				return table[idx];
			}
		};

		// Compressed cache for double
		struct compressed_cache_detail {
			static constexpr int compression_ratio = 27;
			static constexpr std::size_t compressed_table_size =
				(cache_holder<ieee754_binary64>::max_k -
					cache_holder<ieee754_binary64>::min_k + compression_ratio) / compression_ratio;

			STRF_HD static inline wuint::uint128 cache(std::size_t idx) noexcept {
				static const wuint::uint128 table[compressed_table_size] = {
					{ 0xff77b1fcbebcdc4f, 0x25e8e89c13bb0f7b },
					{ 0xce5d73ff402d98e3, 0xfb0a3d212dc81290 },
					{ 0xa6b34ad8c9dfc06f, 0xf42faa48c0ea481f },
					{ 0x86a8d39ef77164bc, 0xae5dff9c02033198 },
					{ 0xd98ddaee19068c76, 0x3badd624dd9b0958 },
					{ 0xafbd2350644eeacf, 0xe5d1929ef90898fb },
					{ 0x8df5efabc5979c8f, 0xca8d3ffa1ef463c2 },
					{ 0xe55990879ddcaabd, 0xcc420a6a101d0516 },
					{ 0xb94470938fa89bce, 0xf808e40e8d5b3e6a },
					{ 0x95a8637627989aad, 0xdde7001379a44aa9 },
					{ 0xf1c90080baf72cb1, 0x5324c68b12dd6339 },
					{ 0xc350000000000000, 0x0000000000000000 },
					{ 0x9dc5ada82b70b59d, 0xf020000000000000 },
					{ 0xfee50b7025c36a08, 0x02f236d04753d5b4 },
					{ 0xcde6fd5e09abcf26, 0xed4c0226b55e6f86 },
					{ 0xa6539930bf6bff45, 0x84db8346b786151c },
					{ 0x865b86925b9bc5c2, 0x0b8a2392ba45a9b2 },
					{ 0xd910f7ff28069da4, 0x1b2ba1518094da04 },
					{ 0xaf58416654a6babb, 0x387ac8d1970027b2 },
					{ 0x8da471a9de737e24, 0x5ceaecfed289e5d2 },
					{ 0xe4d5e82392a40515, 0x0fabaf3feaa5334a },
					{ 0xb8da1662e7b00a17, 0x3d6a751f3b936243 },
					{ 0x95527a5202df0ccb, 0x0f37801e0c43ebc8 }
				};
				return table[idx];
			}

			STRF_HD static inline std::uint64_t pow5(std::size_t idx) noexcept {
				static const std::uint64_t table[compression_ratio] = {
					1ULL, 5ULL, 25ULL, 125ULL, 625ULL, 3125ULL, 15625ULL, 78125ULL,
					390625ULL, 1953125ULL, 9765625ULL, 48828125ULL, 244140625ULL,
					1220703125ULL, 6103515625ULL, 30517578125ULL, 152587890625ULL,
					762939453125ULL, 3814697265625ULL, 19073486328125ULL, 95367431640625ULL,
					476837158203125ULL, 2384185791015625ULL, 11920928955078125ULL,
					59604644775390625ULL, 298023223876953125ULL, 1490116119384765625ULL };
				return table[idx];
			}

			STRF_HD static inline std::uint32_t errors(std::size_t idx) noexcept {
				static const std::uint32_t table[] = {
					0x50001400, 0x54044100, 0x54014555, 0x55954415, 0x54115555,
					0x00000001, 0x50000000, 0x00104000, 0x54010004, 0x05004001,
					0x55555544, 0x41545555, 0x54040551, 0x15445545, 0x51555514,
					0x10000015, 0x00101100, 0x01100015, 0x00000000, 0x00000000,
					0x00000000, 0x00000000, 0x04450514, 0x45414110, 0x55555145,
					0x50544050, 0x15040155, 0x11054140, 0x50111514, 0x11451454,
					0x00400541, 0x00000000, 0x55555450, 0x10056551, 0x10054011,
					0x55551014, 0x69514555, 0x05151109, 0x00155555
				};
				return table[idx];
			}
		};

		template <typename UInt, int kappa>
		STRF_HD constexpr static int do_calculate_max_power_of_10
			(int k, UInt p, UInt limit) noexcept
		{
			return p < limit ? do_calculate_max_power_of_10<UInt, kappa>(k + 1, p * 10, limit) : k;
		}

		template <typename UInt, int kappa,
			UInt max_possible_significand = ((UInt)-1) / mp_compute_power<std::uint32_t, 10, kappa + 1>::value >
		STRF_HD constexpr static int calculate_max_power_of_10() noexcept {
			return do_calculate_max_power_of_10<UInt, kappa>(0, 1, max_possible_significand / 10);
		}
	}  // namespace detail


	////////////////////////////////////////////////////////////////////////////////////////
	// Policies.
	////////////////////////////////////////////////////////////////////////////////////////

	namespace detail {
		// Forward declare the implementation class.
		template <class Float, class FloatTraits = default_float_traits<Float>>
		struct impl;

		namespace policy_impl {
			// Sign policies.
			namespace sign {
				struct base {};

				struct ignore : base {
					using sign_policy = ignore;
					static constexpr bool return_has_sign = false;

					template <class SignedSignificandBits, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void handle_sign(SignedSignificandBits, ReturnType&) noexcept {}
				};

				struct return_sign : base {
					using sign_policy = return_sign;
					static constexpr bool return_has_sign = true;

					template <class SignedSignificandBits, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void handle_sign(SignedSignificandBits s, ReturnType& r) noexcept {
						r.is_negative = s.is_negative();
					}
				};
			}  // namespace sign

			// Trailing zero policies.
			namespace trailing_zero {
				struct base {};

				struct ignore : base {
					using trailing_zero_policy = ignore;
					static constexpr bool report_trailing_zeros = false;

					template <class Impl, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void on_trailing_zeros(ReturnType&) noexcept {}

					template <class Impl, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void no_trailing_zeros(ReturnType&) noexcept {}
				};

				struct remove : base {
					using trailing_zero_policy = remove;
					static constexpr bool report_trailing_zeros = false;

					template <class Impl, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void on_trailing_zeros(ReturnType& r) noexcept {
						r.exponent += Impl::remove_trailing_zeros(r.significand);
					}

					template <class Impl, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void no_trailing_zeros(ReturnType&) noexcept {}
				};

				struct report : base {
					using trailing_zero_policy = report;
					static constexpr bool report_trailing_zeros = true;

					template <class Impl, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void on_trailing_zeros(ReturnType& r) noexcept {
						r.may_have_trailing_zeros = true;
					}

					template <class Impl, class ReturnType>
					STRF_HD static JKJ_FORCEINLINE void no_trailing_zeros(ReturnType& r) noexcept {
						r.may_have_trailing_zeros = false;
					}
				};
			}  // namespace trailing_zero

			// Decimal-to-binary rounding mode policies.
			namespace decimal_to_binary_rounding {
				struct base {};

				enum class tag_t {
					to_nearest,
					left_closed_directed,
					right_closed_directed
				};
				namespace interval_type {
					struct symmetric_boundary {
						STRF_HD symmetric_boundary(bool closed) : is_closed(closed) {
						}
						symmetric_boundary() = default;

						static constexpr bool is_symmetric = true;
						bool is_closed;
						STRF_HD constexpr bool include_left_endpoint() const noexcept {
							return is_closed;
						}
						STRF_HD constexpr bool include_right_endpoint() const noexcept {
							return is_closed;
						}
					};
					// struct asymmetric_boundary {
					// 	static constexpr bool is_symmetric = false;
					// 	bool is_left_closed;
					// 	STRF_HD constexpr bool include_left_endpoint() const noexcept {
					// 		return is_left_closed;
					// 	}
					// 	STRF_HD constexpr bool include_right_endpoint() const noexcept {
					// 		return !is_left_closed;
					// 	}
					// };
					struct closed {
						static constexpr bool is_symmetric = true;
						STRF_HD static constexpr bool include_left_endpoint() noexcept {
							return true;
						}
						STRF_HD static constexpr bool include_right_endpoint() noexcept {
							return true;
						}
					};
					struct open {
						static constexpr bool is_symmetric = true;
						STRF_HD static constexpr bool include_left_endpoint() noexcept {
							return false;
						}
						STRF_HD static constexpr bool include_right_endpoint() noexcept {
							return false;
						}
					};
					struct left_closed_right_open {
						static constexpr bool is_symmetric = false;
						STRF_HD static constexpr bool include_left_endpoint() noexcept {
							return true;
						}
						STRF_HD static constexpr bool include_right_endpoint() noexcept {
							return false;
						}
					};
					struct right_closed_left_open {
						static constexpr bool is_symmetric = false;
						STRF_HD static constexpr bool include_left_endpoint() noexcept {
							return false;
						}
						STRF_HD static constexpr bool include_right_endpoint() noexcept {
							return true;
						}
					};
				}  // namespace interval_type

				struct nearest_to_even : base {
					using decimal_to_binary_rounding_policy = nearest_to_even;
					static constexpr auto tag = tag_t::to_nearest;
					using normal_interval_type = interval_type::symmetric_boundary;
					using shorter_interval_type = interval_type::closed;

					// template <class SignedSignificandBits, class Func>
					// STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept
					// {
					// 	return f(nearest_to_even{});
					// }

					template <class SignedSignificandBits>
					STRF_HD static constexpr normal_interval_type
						get_normal_interval_type(SignedSignificandBits s) noexcept
					{
						return normal_interval_type{s.has_even_significand_bits()};
					}
					template <class SignedSignificandBits>
					STRF_HD static constexpr shorter_interval_type
						get_shorter_interval_type(SignedSignificandBits) noexcept
					{
						return {};
					}

					// template <class SignedSignificandBits, class Func>
					// STRF_HD static constexpr auto
					// 	invoke_normal_interval_case(SignedSignificandBits s, Func&& f) noexcept
					// //-> decltype(f(s.has_even_significand_bits())
					// {
					// 	return f(s.has_even_significand_bits());
					// }
					// template <class SignedSignificandBits, class Func>
					// STRF_HD static constexpr auto
					// 	invoke_shorter_interval_case(SignedSignificandBits, Func&& f) noexcept
					// //-> decltype(f())
					// {
					// 	return f();
					// }
				};
				// struct nearest_to_odd : base {
				// 	using decimal_to_binary_rounding_policy = nearest_to_odd;
				// 	static constexpr auto tag = tag_t::to_nearest;
				// 	using normal_interval_type = interval_type::symmetric_boundary;
				// 	using shorter_interval_type = interval_type::open;

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f(nearest_to_odd{});
				// 	}

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_normal_interval_case(SignedSignificandBits s, Func&& f) noexcept
				// 	{
				// 		return f(!s.has_even_significand_bits());
				// 	}
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_shorter_interval_case(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f();
				// 	}
				// };
				// struct nearest_toward_plus_infinity : base {
				// 	using decimal_to_binary_rounding_policy = nearest_toward_plus_infinity;
				// 	static constexpr auto tag = tag_t::to_nearest;
				// 	using normal_interval_type = interval_type::asymmetric_boundary;
				// 	using shorter_interval_type = interval_type::asymmetric_boundary;

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f(nearest_toward_plus_infinity{});
				// 	}

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_normal_interval_case(SignedSignificandBits s, Func&& f) noexcept
				// 	{
				// 		return f(!s.is_negative());
				// 	}
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_shorter_interval_case(SignedSignificandBits s, Func&& f) noexcept
				// 	{
				// 		return f(!s.is_negative());
				// 	}
				// };
				// struct nearest_toward_minus_infinity : base {
				// 	using decimal_to_binary_rounding_policy = nearest_toward_minus_infinity;
				// 	static constexpr auto tag = tag_t::to_nearest;
				// 	using normal_interval_type = interval_type::asymmetric_boundary;
				// 	using shorter_interval_type = interval_type::asymmetric_boundary;

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f(nearest_toward_minus_infinity{});
				// 	}

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_normal_interval_case(SignedSignificandBits s, Func&& f) noexcept
				// 	{
				// 		return f(s.is_negative());
				// 	}
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_shorter_interval_case(SignedSignificandBits s, Func&& f) noexcept
				// 	{
				// 		return f(s.is_negative());
				// 	}
				// };
				// struct nearest_toward_zero : base {
				// 	using decimal_to_binary_rounding_policy = nearest_toward_zero;
				// 	static constexpr auto tag = tag_t::to_nearest;
				// 	using normal_interval_type = interval_type::right_closed_left_open;
				// 	using shorter_interval_type = interval_type::right_closed_left_open;

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f(nearest_toward_zero{});
				// 	}

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_normal_interval_case(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f();
				// 	}
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_shorter_interval_case(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f();
				// 	}
				// };
				// struct nearest_away_from_zero : base {
				// 	using decimal_to_binary_rounding_policy = nearest_away_from_zero;
				// 	static constexpr auto tag = tag_t::to_nearest;
				// 	using normal_interval_type = interval_type::left_closed_right_open;
				// 	using shorter_interval_type = interval_type::left_closed_right_open;

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f(nearest_away_from_zero{});
				// 	}

				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_normal_interval_case(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f();
				// 	}
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static constexpr auto
				// 		invoke_shorter_interval_case(SignedSignificandBits, Func&& f) noexcept
				// 	{
				// 		return f();
				// 	}
				// };

				namespace detail {
					struct nearest_always_closed {
						static constexpr auto tag = tag_t::to_nearest;
						using normal_interval_type = interval_type::closed;
						using shorter_interval_type = interval_type::closed;

						// template <class SignedSignificandBits, class Func>
						// STRF_HD static constexpr auto
						// 	invoke_normal_interval_case(SignedSignificandBits, Func&& f) noexcept
						// {
						// 	return f();
						// }
						// template <class SignedSignificandBits, class Func>
						// STRF_HD static constexpr auto
						// 	invoke_shorter_interval_case(SignedSignificandBits, Func&& f) noexcept
						// {
						// 	return f();
						// }
					};
					struct nearest_always_open {
						static constexpr auto tag = tag_t::to_nearest;
						using normal_interval_type = interval_type::open;
						using shorter_interval_type = interval_type::open;

						// template <class SignedSignificandBits, class Func>
						// STRF_HD static constexpr auto
						// 	invoke_normal_interval_case(SignedSignificandBits, Func&& f) noexcept
						// {
						// 	return f();
						// }
						// template <class SignedSignificandBits, class Func>
						// STRF_HD static constexpr auto
						// 	invoke_shorter_interval_case(SignedSignificandBits, Func&& f) noexcept
						// {
						// 	return f();
						// }
					};
				}  // namespace detail

				// struct nearest_to_even_static_boundary : base {
				// 	using decimal_to_binary_rounding_policy = nearest_to_even_static_boundary;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits s, Func&& f) noexcept {
				// 		if (s.has_even_significand_bits()) {
				// 			return f(detail::nearest_always_closed{});
				// 		}
				// 		else {
				// 			return f(detail::nearest_always_open{});
				// 		}
				// 	}
				// };
				// struct nearest_to_odd_static_boundary : base {
				// 	using decimal_to_binary_rounding_policy = nearest_to_odd_static_boundary;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits s, Func&& f) noexcept {
				// 		if (s.has_even_significand_bits()) {
				// 			return f(detail::nearest_always_open{});
				// 		}
				// 		else {
				// 			return f(detail::nearest_always_closed{});
				// 		}
				// 	}
				// };
				// struct nearest_toward_plus_infinity_static_boundary : base {
				// 	using decimal_to_binary_rounding_policy = nearest_toward_plus_infinity_static_boundary;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits s, Func&& f) noexcept {
				// 		if (s.is_negative()) {
				// 			return f(nearest_toward_zero{});
				// 		}
				// 		else {
				// 			return f(nearest_away_from_zero{});
				// 		}
				// 	}
				// };
				// struct nearest_toward_minus_infinity_static_boundary : base {
				// 	using decimal_to_binary_rounding_policy = nearest_toward_minus_infinity_static_boundary;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits s, Func&& f) noexcept {
				// 		if (s.is_negative()) {
				// 			return f(nearest_away_from_zero{});
				// 		}
				// 		else {
				// 			return f(nearest_toward_zero{});
				// 		}
				// 	}
				// };

				// namespace detail {
				// 	struct left_closed_directed {
				// 		static constexpr auto tag = tag_t::left_closed_directed;
				// 	};
				// 	struct right_closed_directed {
				// 		static constexpr auto tag = tag_t::right_closed_directed;
				// 	};
				// }

				// struct toward_plus_infinity : base {
				// 	using decimal_to_binary_rounding_policy = toward_plus_infinity;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits s, Func&& f) noexcept {
				// 		if (s.is_negative()) {
				// 			return f(detail::left_closed_directed{});
				// 		}
				// 		else {
				// 			return f(detail::right_closed_directed{});
				// 		}
				// 	}
				// };
				// struct toward_minus_infinity : base {
				// 	using decimal_to_binary_rounding_policy = toward_minus_infinity;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits s, Func&& f) noexcept {
				// 		if (s.is_negative()) {
				// 			return f(detail::right_closed_directed{});
				// 		}
				// 		else {
				// 			return f(detail::left_closed_directed{});
				// 		}
				// 	}
				// };
				// struct toward_zero : base {
				// 	using decimal_to_binary_rounding_policy = toward_zero;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept {
				// 		return f(detail::left_closed_directed{});
				// 	}
				// };
				// struct away_from_zero : base {
				// 	using decimal_to_binary_rounding_policy = away_from_zero;
				// 	template <class SignedSignificandBits, class Func>
				// 	STRF_HD static auto delegate(SignedSignificandBits, Func&& f) noexcept {
				// 		return f(detail::right_closed_directed{});
				// 	}
				// };
			}  // namespace decimal_to_binary_rounding

			// Binary-to-decimal rounding policies.
			// (Always assumes nearest rounding modes.)
			namespace binary_to_decimal_rounding {
				struct base {};

				enum class tag_t {
					do_not_care,
					to_even,
					to_odd,
					away_from_zero,
					toward_zero
				};

				// struct do_not_care : base {
				// 	using binary_to_decimal_rounding_policy = do_not_care;
				// 	static constexpr auto tag = tag_t::do_not_care;

				// 	template <class ReturnType>
				// 	STRF_HD static constexpr void break_rounding_tie(ReturnType&) noexcept {}
				// };

				struct to_even : base {
					using binary_to_decimal_rounding_policy = to_even;
					static constexpr auto tag = tag_t::to_even;

					template <class ReturnType>
					STRF_HD static STRF_CONSTEXPR_IN_CXX14 void break_rounding_tie(ReturnType& r) noexcept
					{
						r.significand = r.significand % 2 == 0 ?
							r.significand : r.significand - 1;
					}
				};

				// struct to_odd : base {
				// 	using binary_to_decimal_rounding_policy = to_odd;
				// 	static constexpr auto tag = tag_t::to_odd;

				// 	template <class ReturnType>
				// 	STRF_HD static constexpr void break_rounding_tie(ReturnType& r) noexcept
				// 	{
				// 		r.significand = r.significand % 2 != 0 ?
				// 			r.significand : r.significand - 1;
				// 	}
				// };

				// struct away_from_zero : base {
				// 	using binary_to_decimal_rounding_policy = away_from_zero;
				// 	static constexpr auto tag = tag_t::away_from_zero;

				// 	template <class ReturnType>
				// 	STRF_HD static constexpr void break_rounding_tie(ReturnType&) noexcept {}
				// };

				// struct toward_zero : base {
				// 	using binary_to_decimal_rounding_policy = toward_zero;
				// 	static constexpr auto tag = tag_t::toward_zero;

				// 	template <class ReturnType>
				// 	STRF_HD static constexpr void break_rounding_tie(ReturnType& r) noexcept
				// 	{
				// 		--r.significand;
				// 	}
				// };
			}  // namespace binary_to_decimal_rounding

			// Cache policies.
			namespace cache {
				struct base {};

				struct full : base {
					using cache_policy = full;
					template <class FloatFormat>
					STRF_HD static constexpr typename cache_holder<FloatFormat>::cache_entry_type
						get_cache(int k) noexcept
					{
						STRF_ASSERT_IN_CONSTEXPR(k >= cache_holder<FloatFormat>::min_k &&
							k <= cache_holder<FloatFormat>::max_k);
						return cache_holder<FloatFormat>::cache(
							std::size_t(k - cache_holder<FloatFormat>::min_k));
					}
				};

				// struct compact : base {
				// 	using cache_policy = compact;
				// 	template <class FloatFormat>
				// 	STRF_HD static constexpr typename cache_holder<FloatFormat>::cache_entry_type
				// 		get_cache(int k) noexcept
				// 	{
				// 		STRF_ASSERT(k >= cache_holder<FloatFormat>::min_k &&
				// 			k <= cache_holder<FloatFormat>::max_k);

				// 		JKJ_IF_CONSTEXPR (std::is_same<FloatFormat, ieee754_binary64>::value)
				// 		{
				// 			// Compute base index.
				// 			auto cache_index = (k - cache_holder<FloatFormat>::min_k) /
				// 				compressed_cache_detail::compression_ratio;
				// 			auto kb = cache_index * compressed_cache_detail::compression_ratio
				// 				+ cache_holder<FloatFormat>::min_k;
				// 			auto offset = k - kb;

				// 			// Get base cache.
				// 			auto base_cache = compressed_cache_detail::cache(cache_index);

				// 			if (offset == 0) {
				// 				return base_cache;
				// 			}
				// 			else {
				// 				// Compute the required amount of bit-shift.
				// 				auto alpha = log::floor_log2_pow10(kb + offset)
				// 					- log::floor_log2_pow10(kb) - offset;
				// 				STRF_ASSERT(alpha > 0 && alpha < 64);

				// 				// Try to recover the real cache.
				// 				auto pow5 = compressed_cache_detail::pow5(offset);
				// 				auto recovered_cache = wuint::umul128(base_cache.high(), pow5);
				// 				auto middle_low = wuint::umul128(base_cache.low() - (kb < 0 ? 1 : 0), pow5);

				// 				recovered_cache += middle_low.high();

				// 				auto high_to_middle = recovered_cache.high() << (64 - alpha);
				// 				auto middle_to_low = recovered_cache.low() << (64 - alpha);

				// 				recovered_cache = wuint::uint128{
				// 					(recovered_cache.low() >> alpha) | high_to_middle,
				// 					((middle_low.low() >> alpha) | middle_to_low)
				// 				};

				// 				if (kb < 0) {
				// 					recovered_cache += 1;
				// 				}

				// 				// Get error.
				// 				auto error_idx = (k - cache_holder<FloatFormat>::min_k) / 16;
				// 				auto error = (compressed_cache_detail::errors(error_idx) >>
				// 					((k - cache_holder<FloatFormat>::min_k) % 16) * 2) & 0x3;

				// 				// Add the error back.
				// 				STRF_ASSERT(recovered_cache.low() + error >= recovered_cache.low());
				// 				recovered_cache = {
				// 					recovered_cache.high(),
				// 					recovered_cache.low() + error
				// 				};

				// 				return recovered_cache;
				// 			}
				// 		}
				// 		else
				// 		{
				// 			// Just use the full cache for anything other than binary64
				// 			return cache_holder<FloatFormat>::cache(
				// 				std::size_t(k - cache_holder<FloatFormat>::min_k));
				// 		}
				// 	}
				// };
			}  // namespace cache
		}  // namespace policy_impl
	}  // namespace detail

	namespace policy {
		namespace sign {
			constexpr auto ignore = detail::policy_impl::sign::ignore{};
		// constexpr auto return_sign = detail::policy_impl::sign::return_sign{};
		} // namespace sign

		namespace trailing_zero {
			constexpr auto ignore = detail::policy_impl::trailing_zero::ignore{};
			constexpr auto remove = detail::policy_impl::trailing_zero::remove{};
			constexpr auto report = detail::policy_impl::trailing_zero::report{};
		} // namespace trailing_zero

		namespace decimal_to_binary_rounding {
			constexpr auto nearest_to_even =
				detail::policy_impl::decimal_to_binary_rounding::nearest_to_even{};
			// constexpr auto nearest_to_odd =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_to_odd{};
			// constexpr auto nearest_toward_plus_infinity =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_toward_plus_infinity{};
			// constexpr auto nearest_toward_minus_infinity =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_toward_minus_infinity{};
			// constexpr auto nearest_toward_zero =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_toward_zero{};
			// constexpr auto nearest_away_from_zero =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_away_from_zero{};

			// constexpr auto nearest_to_even_static_boundary =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_to_even_static_boundary{};
			// constexpr auto nearest_to_odd_static_boundary =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_to_odd_static_boundary{};
			// constexpr auto nearest_toward_plus_infinity_static_boundary =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_toward_plus_infinity_static_boundary{};
			// constexpr auto nearest_toward_minus_infinity_static_boundary =
			// 	detail::policy_impl::decimal_to_binary_rounding::nearest_toward_minus_infinity_static_boundary{};

			// constexpr auto toward_plus_infinity =
			// 	detail::policy_impl::decimal_to_binary_rounding::toward_plus_infinity{};
			// constexpr auto toward_minus_infinity =
			// 	detail::policy_impl::decimal_to_binary_rounding::toward_minus_infinity{};
			// constexpr auto toward_zero =
			// 	detail::policy_impl::decimal_to_binary_rounding::toward_zero{};
			// constexpr auto away_from_zero =
			// 	detail::policy_impl::decimal_to_binary_rounding::away_from_zero{};
		}  // namespace decimal_to_binary_rounding

		namespace binary_to_decimal_rounding {
			// constexpr auto do_not_care =
			// 	detail::policy_impl::binary_to_decimal_rounding::do_not_care{};
			constexpr auto to_even =
				detail::policy_impl::binary_to_decimal_rounding::to_even{};
			// constexpr auto to_odd =
			// 	detail::policy_impl::binary_to_decimal_rounding::to_odd{};
			// constexpr auto away_from_zero =
			// 	detail::policy_impl::binary_to_decimal_rounding::away_from_zero{};
			// constexpr auto toward_zero =
			// 	detail::policy_impl::binary_to_decimal_rounding::toward_zero{};
		}  // namespace binary_to_decimal_rounding

		namespace cache {
			constexpr auto full = detail::policy_impl::cache::full{};
			// constexpr auto compact = detail::policy_impl::cache::compact{};
		} // namespace cache
	}  // namespace policy

	namespace detail {
		////////////////////////////////////////////////////////////////////////////////////////
		// The main algorithm.
		////////////////////////////////////////////////////////////////////////////////////////

		// Get decimal significand/decimal exponent from
		// the bit representation of a floating-point number.
		template <class Float, class Format, class FloatTraits>
		struct impl_base;

		template <class FloatTraits>
		struct impl_base<float, ieee754_binary32, FloatTraits>
		{
			using format = ieee754_binary32;
			using carrier_uint = typename FloatTraits::carrier_uint;
			using cache_entry_type = typename cache_holder<format>::cache_entry_type;
			static constexpr auto significand_bits = format::significand_bits;
			static constexpr auto cache_bits = cache_holder<format>::cache_bits;
			static constexpr auto carrier_bits = FloatTraits::carrier_bits;
			static constexpr int kappa = 1;

			static_assert(std::is_same<typename FloatTraits::format, format>::value, "");

			STRF_HD static carrier_uint compute_mul(carrier_uint u, cache_entry_type const& cache) noexcept
			{
				return wuint::umul96_upper32(u, cache);
			}

			STRF_HD static constexpr std::uint32_t compute_delta(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return std::uint32_t(cache >> (cache_bits - 1 - beta_minus_1));
			}

			STRF_HD static bool compute_mul_parity(carrier_uint two_f,
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				STRF_ASSERT(beta_minus_1 >= 1);
				STRF_ASSERT(beta_minus_1 < 64);
				return ((wuint::umul96_lower64(two_f, cache) >>
						(64 - beta_minus_1)) & 1) != 0;
			}

			STRF_HD static constexpr carrier_uint compute_left_endpoint_for_shorter_interval_case(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return carrier_uint(
					(cache - (cache >> (significand_bits + 2))) >>
					(cache_bits - significand_bits - 1 - beta_minus_1));
			}

			STRF_HD static constexpr carrier_uint compute_right_endpoint_for_shorter_interval_case(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return carrier_uint(
					(cache + (cache >> (significand_bits + 1))) >>
					(cache_bits - significand_bits - 1 - beta_minus_1));
			}

			STRF_HD static constexpr carrier_uint compute_round_up_for_shorter_interval_case(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return (carrier_uint(cache >> (cache_bits - significand_bits - 2 - beta_minus_1)) + 1) / 2;
			}

			// Remove trailing zeros from n and return the number of zeros removed.
			STRF_HD JKJ_FORCEINLINE static int remove_trailing_zeros(carrier_uint& n) noexcept {
				constexpr auto max_power = calculate_max_power_of_10<carrier_uint, kappa>();

				static_assert(max_power == 7, "Assertion failed! Did you change kappa?");

				using divtable = div::division_by_5<carrier_uint>;

				// Perform a binary search.
				carrier_uint quotient;
				int s = 0;

				// Is n divisible by 10^4?
				if ((n & 0xf) == 0) {
					quotient = (n >> 4) * divtable::mod_inv(4);
					if (quotient <= divtable::max_quotients(4)) {
						n = quotient;
						s |= 0x4;
					}
				}

				// Is n divisible by 10^2?
				if ((n & 0x3) == 0) {
					quotient = (n >> 2) * divtable::mod_inv(2);
					if (quotient <= divtable::max_quotients(2)) {
						n = quotient;
						s |= 0x2;
					}
				}

				// Is n divisible by 10^1?
				if ((n & 0x1) == 0) {
					quotient = (n >> 1) * divtable::mod_inv(1);
					if (quotient <= divtable::max_quotients(1)) {
						n = quotient;
						s |= 0x1;
					}
				}

				return s;
			}
		};

		template <class FloatTraits>
		struct impl_base<double, ieee754_binary64, FloatTraits>
		{
			using format = ieee754_binary64;
			using carrier_uint = typename FloatTraits::carrier_uint;
			using cache_entry_type = typename cache_holder<format>::cache_entry_type;
			static constexpr auto significand_bits = format::significand_bits;
			static constexpr auto cache_bits = cache_holder<format>::cache_bits;
			static constexpr auto carrier_bits = FloatTraits::carrier_bits;
			static constexpr int kappa = 2;

			static_assert(std::is_same<typename FloatTraits::format, format>::value, "");

			STRF_HD static carrier_uint compute_mul(carrier_uint u, cache_entry_type const& cache) noexcept
			{
				return wuint::umul192_upper64(u, cache);
			}

			STRF_HD static constexpr std::uint32_t compute_delta(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return std::uint32_t(cache.high() >> (carrier_bits - 1 - beta_minus_1));
			}

			STRF_HD static bool compute_mul_parity(carrier_uint two_f,
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				STRF_ASSERT(beta_minus_1 >= 1);
				STRF_ASSERT(beta_minus_1 < 64);
					return ((wuint::umul192_middle64(two_f, cache) >>
						(64 - beta_minus_1)) & 1) != 0;
			}

			STRF_HD static constexpr carrier_uint compute_left_endpoint_for_shorter_interval_case(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return (cache.high() - (cache.high() >> (significand_bits + 2))) >>
					(carrier_bits - significand_bits - 1 - beta_minus_1);
			}

			STRF_HD static constexpr carrier_uint compute_right_endpoint_for_shorter_interval_case(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return (cache.high() + (cache.high() >> (significand_bits + 1))) >>
					(carrier_bits - significand_bits - 1 - beta_minus_1);
			}

			STRF_HD static constexpr carrier_uint compute_round_up_for_shorter_interval_case(
				cache_entry_type const& cache, int beta_minus_1) noexcept
			{
				return ((cache.high() >> (carrier_bits - significand_bits - 2 - beta_minus_1)) + 1) / 2;
			}

			// Remove trailing zeros from n and return the number of zeros removed.
			STRF_HD JKJ_FORCEINLINE static int remove_trailing_zeros(carrier_uint& n) noexcept {
				// Divide by 10^8 and reduce to 32-bits.
				// Since ret_value.significand <= (2^64 - 1) / 1000 < 10^17,
				// both of the quotient and the r should fit in 32-bits.

				using divtable32 = div::division_by_5<std::uint32_t>;

				// If the number is divisible by 1'0000'0000, work with the quotient.
				auto quotient_by_pow10_8 = std::uint32_t(div::divide_by_pow10<8, 54, 0>(n));
				auto remainder = std::uint32_t(n - 100000000 * quotient_by_pow10_8);

				if (remainder == 0) {
					auto n32 = quotient_by_pow10_8;
					std::uint32_t quotient32 = 0;

					// Is n divisible by 10^8?
					// This branch is extremely unlikely.
					// I suspect it is impossible to get into this branch.
					if ((n32 & 0xff) == 0) {
						quotient32 = (n32 >> 8) * divtable32::mod_inv(8);
						if (quotient32 <= divtable32::max_quotients(8)) {
							n = quotient32;
							return 16;
						}
					}

					// Otherwise, perform a binary search.
					int s = 8;

					// Is n divisible by 10^4?
					if ((n32 & 0xf) == 0) {
						quotient32 = (n32 >> 4) * divtable32::mod_inv(4);
						if (quotient32 <= divtable32::max_quotients(4)) {
							n32 = quotient32;
							s |= 0x4;
						}
					}

					// Is n divisible by 10^2?
					if ((n32 & 0x3) == 0) {
						quotient32 = (n32 >> 2) * divtable32::mod_inv(2);
						if (quotient32 <= divtable32::max_quotients(2)) {
							n32 = quotient32;
							s |= 0x2;
						}
					}

					// Is n divisible by 10^1?
					if ((n32 & 0x1) == 0) {
						quotient32 = (n32 >> 1) * divtable32::mod_inv(1);
						if (quotient32 <= divtable32::max_quotients(1)) {
							n32 = quotient32;
							s |= 0x1;
						}
					}

					n = n32;
					return s;
				}

				// If the number is not divisible by 1'0000'0000, work with the remainder.

				// Perform a binary search.
				std::uint32_t quotient32 = 0;
				std::uint32_t multiplier = 100000000;
				int s = 0;

				// Is n divisible by 10^4?
				if ((remainder & 0xf) == 0) {
					quotient32 = (remainder >> 4) * divtable32::mod_inv(4);
					if (quotient32 <= divtable32::max_quotients(4)) {
						remainder = quotient32;
						multiplier = 10000;
						s |= 0x4;
					}
				}

				// Is n divisible by 10^2?
				if ((remainder & 0x3) == 0) {
					quotient32 = (remainder >> 2) * divtable32::mod_inv(2);
					if (quotient32 <= divtable32::max_quotients(2)) {
						remainder = quotient32;
						multiplier = (s == 4 ? 100 : 1000000);
						s |= 0x2;
					}
				}

				// Is n divisible by 10^1?
				if ((remainder & 0x1) == 0) {
					quotient32 = (remainder >> 1) * divtable32::mod_inv(1);
					if (quotient32 <= divtable32::max_quotients(1)) {
						remainder = quotient32;
						multiplier = (multiplier >> 1) * divtable32::mod_inv(1);
						s |= 0x1;
					}
				}

				n = remainder + quotient_by_pow10_8 * carrier_uint(multiplier);
				return s;
			}
		};


		template <class Float, class FloatTraits>
		struct impl : private FloatTraits, private FloatTraits::format,
			public impl_base<Float, typename FloatTraits::format, FloatTraits>
		{
			using format = typename FloatTraits::format;
			using carrier_uint = typename FloatTraits::carrier_uint;

			using FloatTraits::carrier_bits;
			using format::significand_bits;
			using format::min_exponent;
			using format::max_exponent;
			using format::exponent_bias;
			using format::decimal_digits;

			using base = impl_base<Float, format, FloatTraits>;
			using base::compute_mul;
			using base::compute_delta;
			using base::compute_mul_parity;
			using base::compute_left_endpoint_for_shorter_interval_case;
			using base::compute_right_endpoint_for_shorter_interval_case;
			using base::compute_round_up_for_shorter_interval_case;

			static constexpr int kappa = std::is_same<format, ieee754_binary32>::value ? 1 : 2;
			static_assert(kappa >= 1, "");
			static_assert(carrier_bits >= significand_bits + 2 + log::floor_log2_pow10(kappa + 1), "");

			static constexpr int min_k = strf::detail::min
				( -log::floor_log10_pow2_minus_log10_4_over_3(int(max_exponent - significand_bits))
				, -log::floor_log10_pow2(int(max_exponent - significand_bits)) + kappa );
			static_assert(min_k >= cache_holder<format>::min_k, "");

			static constexpr int max_k = strf::detail::max
				( -log::floor_log10_pow2_minus_log10_4_over_3(int(min_exponent - significand_bits + 1))
				, -log::floor_log10_pow2(int(min_exponent - significand_bits)) + kappa );
			static_assert(max_k <= cache_holder<format>::max_k, "");

			using cache_entry_type =
				typename cache_holder<format>::cache_entry_type;
			static constexpr auto cache_bits =
				cache_holder<format>::cache_bits;

			static constexpr int max_power_of_factor_of_5 = log::floor_log5_pow2(int(significand_bits + 2));
			static constexpr int divisibility_check_by_5_threshold =
				log::floor_log2_pow10(max_power_of_factor_of_5 + kappa + 1);

			static constexpr int case_fc_pm_half_lower_threshold = -kappa - log::floor_log5_pow2(kappa);
			static constexpr int case_fc_pm_half_upper_threshold = log::floor_log2_pow10(kappa + 1);

			static constexpr int case_fc_lower_threshold = -kappa - 1 - log::floor_log5_pow2(kappa + 1);
			static constexpr int case_fc_upper_threshold = log::floor_log2_pow10(kappa + 1);

			static constexpr int case_shorter_interval_left_endpoint_lower_threshold = 2;
			static constexpr int case_shorter_interval_left_endpoint_upper_threshold = 2 +
				log::floor_log2(mp_compute_power
					< unsigned
					, (unsigned)10
					, count_factors<5>((carrier_uint(1) << (significand_bits + 2)) - 1) + 1
					> ::value / 3);

			static constexpr int case_shorter_interval_right_endpoint_lower_threshold = 0;
			static constexpr int case_shorter_interval_right_endpoint_upper_threshold = 2 +
				log::floor_log2(mp_compute_power
					< unsigned
					, 10
					, count_factors<5>((carrier_uint(1) << (significand_bits + 1)) + 1) + 1
					> ::value / 3);

			static constexpr int shorter_interval_tie_lower_threshold =
				-log::floor_log5_pow2_minus_log5_3(significand_bits + 4) - 2 - significand_bits;
			static constexpr int shorter_interval_tie_upper_threshold =
				-log::floor_log5_pow2(significand_bits + 2) - 2 - significand_bits;

			//// The main algorithm assumes the input is a normal/subnormal finite number

			template <class ReturnType, class IntervalType, class TrailingZeroPolicy,
				class BinaryToDecimalRoundingPolicy, class CachePolicy, class... AdditionalArgs>
			STRF_HD JKJ_SAFEBUFFERS static ReturnType compute_nearest_normal(carrier_uint const two_fc,
				int const exponent, IntervalType interval_type) noexcept
			{
				//////////////////////////////////////////////////////////////////////
				// Step 1: Schubfach multiplier calculation
				//////////////////////////////////////////////////////////////////////

				ReturnType ret_value;

				// Compute k and beta.
				int const minus_k = log::floor_log10_pow2(exponent) - kappa;
				auto const cache = CachePolicy::template get_cache<format>(-minus_k);
				int const beta_minus_1 = exponent + log::floor_log2_pow10(-minus_k);

				// Compute zi and deltai.
				// 10^kappa <= deltai < 10^(kappa + 1)
				auto const deltai = compute_delta(cache, beta_minus_1);
				carrier_uint const two_fr = two_fc | 1;
				carrier_uint const zi = compute_mul(two_fr << beta_minus_1, cache);


				//////////////////////////////////////////////////////////////////////
				// Step 2: Try larger divisor; remove trailing zeros if necessary
				//////////////////////////////////////////////////////////////////////

				constexpr auto big_divisor = mp_compute_power<std::uint32_t, 10, kappa + 1>::value;
				constexpr auto small_divisor = mp_compute_power<std::uint32_t, 10, kappa>::value;

				// Using an upper bound on zi, we might be able to optimize the division
				// better than the compiler; we are computing zi / big_divisor here.
				ret_value.significand = div::divide_by_pow10<kappa + 1,
					significand_bits + kappa + 2, kappa + 1>(zi);
				auto r = std::uint32_t(zi - big_divisor * ret_value.significand);

				if (r > deltai) {
					goto small_divisor_case_label;
				}
				else if (r < deltai) {
					// Exclude the right endpoint if necessary.
					if (r == 0 && !interval_type.include_right_endpoint() &&
						is_product_integer<integer_check_case_id::fc_pm_half>(two_fr, exponent, minus_k))
					{
						JKJ_IF_CONSTEXPR (BinaryToDecimalRoundingPolicy::tag ==
							policy_impl::binary_to_decimal_rounding::tag_t::do_not_care)
						{
							ret_value.significand *= 10;
							ret_value.exponent = minus_k + kappa;
							--ret_value.significand;
							return ret_value;
						}
						else {
							--ret_value.significand;
							r = big_divisor;
							goto small_divisor_case_label;
						}
					}
				}
				else {
					// r == deltai; compare fractional parts.
					// Check conditions in the order different from the paper
					// to take advantage of short-circuiting.
					auto const two_fl = two_fc - 1;
					if ((!interval_type.include_left_endpoint() ||
						!is_product_integer<integer_check_case_id::fc_pm_half>(
							two_fl, exponent, minus_k)) &&
						!compute_mul_parity(two_fl, cache, beta_minus_1))
					{
						goto small_divisor_case_label;
					}
				}
				ret_value.exponent = minus_k + kappa + 1;

				// We may need to remove trailing zeros.
				TrailingZeroPolicy::template on_trailing_zeros<impl>(ret_value);
				return ret_value;


				//////////////////////////////////////////////////////////////////////
				// Step 3: Find the significand with the smaller divisor
				//////////////////////////////////////////////////////////////////////

			small_divisor_case_label:
				TrailingZeroPolicy::template no_trailing_zeros<impl>(ret_value);
				ret_value.significand *= 10;
				ret_value.exponent = minus_k + kappa;

				JKJ_IF_CONSTEXPR (BinaryToDecimalRoundingPolicy::tag ==
					policy_impl::binary_to_decimal_rounding::tag_t::do_not_care)
				{
					// Normally, we want to compute
					// ret_value.significand += r / small_divisor
					// and return, but we need to take care of the case that the resulting
					// value is exactly the right endpoint, while that is not included in the interval.
					(void) small_divisor;
					if (!interval_type.include_right_endpoint()) {
						// Is r divisible by 10^kappa?
						if (div::check_divisibility_and_divide_by_pow10<kappa>(r) &&
							is_product_integer<integer_check_case_id::fc_pm_half>(two_fr, exponent, minus_k))
						{
							// This should be in the interval.
							ret_value.significand += r - 1;
						}
						else {
							ret_value.significand += r;
						}
					}
					else {
						ret_value.significand += div::small_division_by_pow10<kappa>(r);
					}
				}
				else
				{
					auto dist = r - (deltai / 2) + (small_divisor / 2);
					bool const approx_y_parity = ((dist ^ (small_divisor / 2)) & 1) != 0;

					// Is dist divisible by 10^kappa?
					const bool divisible_by_10_to_the_kappa =
						div::check_divisibility_and_divide_by_pow10<kappa>(dist);

					// Add dist / 10^kappa to the significand.
					ret_value.significand += dist;

					if (divisible_by_10_to_the_kappa) {
						// Check z^(f) >= epsilon^(f)
						// We have either yi == zi - epsiloni or yi == (zi - epsiloni) - 1,
						// where yi == zi - epsiloni if and only if z^(f) >= epsilon^(f)
						// Since there are only 2 possibilities, we only need to care about the parity.
						// Also, zi and r should have the same parity since the divisor
						// is an even number.
						if (compute_mul_parity(two_fc, cache, beta_minus_1) != approx_y_parity) {
							--ret_value.significand;
						}
						else {
							// If z^(f) >= epsilon^(f), we might have a tie
							// when z^(f) == epsilon^(f), or equivalently, when y is an integer.
							// For tie-to-up case, we can just choose the upper one.
							JKJ_IF_CONSTEXPR (BinaryToDecimalRoundingPolicy::tag !=
								policy_impl::binary_to_decimal_rounding::tag_t::away_from_zero)
							{
								if (is_product_integer<integer_check_case_id::fc>(
									two_fc, exponent, minus_k))
								{
									BinaryToDecimalRoundingPolicy::break_rounding_tie(ret_value);
								}
							}
						}
					}
				}
				return ret_value;
			}

			template <class ReturnType, class IntervalType, class TrailingZeroPolicy,
				class BinaryToDecimalRoundingPolicy, class CachePolicy, class... AdditionalArgs>
			STRF_HD JKJ_SAFEBUFFERS static ReturnType compute_nearest_shorter(
				int const exponent, IntervalType interval_type) noexcept
			{
				ReturnType ret_value;
				STRF_MAYBE_UNUSED(interval_type);

				// Compute k and beta.
				int const minus_k = log::floor_log10_pow2_minus_log10_4_over_3(exponent);
				int const beta_minus_1 = exponent + log::floor_log2_pow10(-minus_k);

				// Compute xi and zi.
				auto const cache = CachePolicy::template get_cache<format>(-minus_k);

				auto xi = compute_left_endpoint_for_shorter_interval_case(cache, beta_minus_1);
				auto zi = compute_right_endpoint_for_shorter_interval_case(cache, beta_minus_1);

				// If we don't accept the right endpoint and
				// if the right endpoint is an integer, decrease it.
				if (!interval_type.include_right_endpoint() &&
					is_right_endpoint_integer_shorter_interval(exponent))
				{
					--zi;
				}
				// If we don't accept the left endpoint or
				// if the left endpoint is not an integer, increase it.
				if (!interval_type.include_left_endpoint() ||
					!is_left_endpoint_integer_shorter_interval(exponent))
				{
					++xi;
				}

				// Try bigger divisor.
				ret_value.significand = zi / 10;

				// If succeed, remove trailing zeros if necessary and return.
				if (ret_value.significand * 10 >= xi) {
					ret_value.exponent = minus_k + 1;
					TrailingZeroPolicy::template on_trailing_zeros<impl>(ret_value);
					return ret_value;
				}

				// Otherwise, compute the round-up of y.
				TrailingZeroPolicy::template no_trailing_zeros<impl>(ret_value);
				ret_value.significand = compute_round_up_for_shorter_interval_case(cache, beta_minus_1);
				ret_value.exponent = minus_k;

				// When tie occurs, choose one of them according to the rule.
				JKJ_IF_CONSTEXPR (BinaryToDecimalRoundingPolicy::tag !=
					policy_impl::binary_to_decimal_rounding::tag_t::do_not_care &&
					BinaryToDecimalRoundingPolicy::tag !=
					policy_impl::binary_to_decimal_rounding::tag_t::away_from_zero)
				{
					if (exponent >= shorter_interval_tie_lower_threshold &&
						exponent <= shorter_interval_tie_upper_threshold)
					{
						BinaryToDecimalRoundingPolicy::break_rounding_tie(ret_value);
					}
					else if (ret_value.significand < xi) {
						++ret_value.significand;
					}
				}
				else
				{
					if (ret_value.significand < xi) {
						++ret_value.significand;
					}
				}
				return ret_value;
			}

			// template <class ReturnType, class TrailingZeroPolicy, class CachePolicy>
			// STRF_HD JKJ_SAFEBUFFERS static ReturnType compute_left_closed_directed(
			// 		carrier_uint const two_fc, int exponent) noexcept
			// {
			// 	//////////////////////////////////////////////////////////////////////
			// 	// Step 1: Schubfach multiplier calculation
			// 	//////////////////////////////////////////////////////////////////////

			// 	ReturnType ret_value;

			// 	// Compute k and beta.
			// 	int const minus_k = log::floor_log10_pow2(exponent) - kappa;
			// 	auto const cache = CachePolicy::template get_cache<format>(-minus_k);
			// 	int const beta_minus_1 = exponent + log::floor_log2_pow10(-minus_k);

			// 	// Compute xi and deltai.
			// 	// 10^kappa <= deltai < 10^(kappa + 1)
			// 	auto const deltai = compute_delta(cache, beta_minus_1);
			// 	carrier_uint xi = compute_mul(two_fc << beta_minus_1, cache);

			// 	if (!is_product_integer<integer_check_case_id::fc>(two_fc, exponent, minus_k)) {
			// 		++xi;
			// 	}

			// 	//////////////////////////////////////////////////////////////////////
			// 	// Step 2: Try larger divisor; remove trailing zeros if necessary
			// 	//////////////////////////////////////////////////////////////////////

			// 	constexpr auto big_divisor = mp_compute_power<std::uint32_t, 10, kappa + 1>::value;

			// 	// Using an upper bound on xi, we might be able to optimize the division
			// 	// better than the compiler; we are computing xi / big_divisor here.
			// 	ret_value.significand = div::divide_by_pow10<kappa + 1,
			// 		significand_bits + kappa + 2, kappa + 1>(xi);
			// 	auto r = std::uint32_t(xi - big_divisor * ret_value.significand);

			// 	if (r != 0) {
			// 		++ret_value.significand;
			// 		r = big_divisor - r;
			// 	}

			// 	if (r > deltai) {
			// 		goto small_divisor_case_label;
			// 	}
			// 	else if (r == deltai) {
			// 		// Compare the fractional parts.
			// 		if (compute_mul_parity(two_fc + 2, cache, beta_minus_1) ||
			// 			is_product_integer<integer_check_case_id::fc>(two_fc + 2, exponent, minus_k))
			// 		{
			// 			goto small_divisor_case_label;
			// 		}
			// 	}

			// 	// The ceiling is inside, so we are done.
			// 	ret_value.exponent = minus_k + kappa + 1;
			// 	TrailingZeroPolicy::template on_trailing_zeros<impl>(ret_value);
			// 	return ret_value;


			// 	//////////////////////////////////////////////////////////////////////
			// 	// Step 3: Find the significand with the smaller divisor
			// 	//////////////////////////////////////////////////////////////////////

			// small_divisor_case_label:
			// 	ret_value.significand *= 10;
			// 	ret_value.significand -= div::small_division_by_pow10<kappa>(r);
			// 	ret_value.exponent = minus_k + kappa;
			// 	TrailingZeroPolicy::template no_trailing_zeros<impl>(ret_value);
			// 	return ret_value;
			// }

			// template <class ReturnType, class TrailingZeroPolicy, class CachePolicy>
			// STRF_HD JKJ_SAFEBUFFERS static ReturnType compute_right_closed_directed(
			// 		carrier_uint const two_fc, int const exponent, bool shorter_interval) noexcept
			// {
			// 	//////////////////////////////////////////////////////////////////////
			// 	// Step 1: Schubfach multiplier calculation
			// 	//////////////////////////////////////////////////////////////////////

			// 	ReturnType ret_value;

			// 	// Compute k and beta.
			// 	int const minus_k = log::floor_log10_pow2(exponent - (shorter_interval ? 1 : 0)) - kappa;
			// 	auto const cache = CachePolicy::template get_cache<format>(-minus_k);
			// 	int const beta_minus_1 = exponent + log::floor_log2_pow10(-minus_k);

			// 	// Compute zi and deltai.
			// 	// 10^kappa <= deltai < 10^(kappa + 1)
			// 	auto const deltai = shorter_interval ?
			// 		compute_delta(cache, beta_minus_1 - 1) :
			// 		compute_delta(cache, beta_minus_1);
			// 	carrier_uint const zi = compute_mul(two_fc << beta_minus_1, cache);


			// 	//////////////////////////////////////////////////////////////////////
			// 	// Step 2: Try larger divisor; remove trailing zeros if necessary
			// 	//////////////////////////////////////////////////////////////////////

			// 	constexpr auto big_divisor = mp_compute_power<std::uint32_t, 10, kappa + 1>::value;

			// 	// Using an upper bound on zi, we might be able to optimize the division
			// 	// better than the compiler; we are computing zi / big_divisor here.
			// 	ret_value.significand = div::divide_by_pow10<kappa + 1,
			// 		significand_bits + kappa + 2, kappa + 1>(zi);
			// 	auto const r = std::uint32_t(zi - big_divisor * ret_value.significand);

			// 	if (r > deltai) {
			// 		goto small_divisor_case_label;
			// 	}
			// 	else if (r == deltai) {
			// 		// Compare the fractional parts.
			// 		if (shorter_interval) {
			// 			if (!compute_mul_parity((two_fc * 2) - 1, cache, beta_minus_1 - 1))
			// 			{
			// 				goto small_divisor_case_label;
			// 			}
			// 		}
			// 		else {
			// 			if (!compute_mul_parity(two_fc - 1, cache, beta_minus_1))
			// 			{
			// 				goto small_divisor_case_label;
			// 			}
			// 		}
			// 	}

			// 	// The floor is inside, so we are done.
			// 	ret_value.exponent = minus_k + kappa + 1;
			// 	TrailingZeroPolicy::template on_trailing_zeros<impl>(ret_value);
			// 	return ret_value;


			// 	//////////////////////////////////////////////////////////////////////
			// 	// Step 3: Find the significand with the small divisor
			// 	//////////////////////////////////////////////////////////////////////

			// small_divisor_case_label:
			// 	ret_value.significand *= 10;
			// 	ret_value.significand += div::small_division_by_pow10<kappa>(r);
			// 	ret_value.exponent = minus_k + kappa;
			// 	TrailingZeroPolicy::template no_trailing_zeros<impl>(ret_value);
			// 	return ret_value;
			// }

			STRF_HD static constexpr bool is_right_endpoint_integer_shorter_interval(int exponent) noexcept {
				return exponent >= case_shorter_interval_right_endpoint_lower_threshold &&
					exponent <= case_shorter_interval_right_endpoint_upper_threshold;
			}

			STRF_HD static constexpr bool is_left_endpoint_integer_shorter_interval(int exponent) noexcept {
				return exponent >= case_shorter_interval_left_endpoint_lower_threshold &&
					exponent <= case_shorter_interval_left_endpoint_upper_threshold;
			}

			enum class integer_check_case_id {
				fc_pm_half,
				fc
			};
			STRF_HD static bool is_product_integer(std::integral_constant<integer_check_case_id, integer_check_case_id::fc_pm_half>,
				carrier_uint two_f, int exponent, int minus_k) noexcept
			{
				if (exponent < case_fc_pm_half_lower_threshold) {
					return false;
				}
				// For k >= 0
				if (exponent <= case_fc_pm_half_upper_threshold) {
					return true;
				}
				// For k < 0
				if (exponent > divisibility_check_by_5_threshold) {
					return false;
				}
				return div::divisible_by_power_of_5<max_power_of_factor_of_5 + 1>(two_f, minus_k);
			}
			STRF_HD static bool is_product_integer(std::integral_constant<integer_check_case_id, integer_check_case_id::fc>,
				carrier_uint two_f, int exponent, int minus_k) noexcept
			{
				// Case II: f = fc + 1
				// Case III: f = fc
				// Exponent for 5 is negative
				if (exponent > divisibility_check_by_5_threshold) {
					return false;
				}
				if (exponent > case_fc_upper_threshold) {
					return div::divisible_by_power_of_5<max_power_of_factor_of_5 + 1>(two_f, minus_k);
				}
				// Both exponents are nonnegative
				if (exponent >= case_fc_lower_threshold) {
					return true;
				}
				// Exponent for 2 is negative
				return div::divisible_by_power_of_2(two_f, minus_k - exponent + 1);
			}

			template <integer_check_case_id case_id>
			STRF_HD static bool is_product_integer(carrier_uint two_f, int exponent, int minus_k) noexcept
			{
				return is_product_integer(std::integral_constant<integer_check_case_id, case_id>(), two_f, exponent, minus_k);
			}
		};
	}  // namespace detail

	template <typename Float>
	using return_type = decimal_fp<
		typename default_float_traits<Float>::carrier_uint,
		false,   // return_has_sign,
		false >; // report_trailing_zeros;

	template <class Float>
	STRF_HD JKJ_FORCEINLINE return_type<Float> to_decimal
		( unsigned exponent_bits
		, typename std::conditional
			< sizeof(Float) == 4, std::uint32_t, std::uint64_t>
			::type mantissa_bits )
	{
		using FloatTraits = default_float_traits<Float>;
		using format = typename FloatTraits::format;
		const signed_significand_bits<Float, FloatTraits>
			significand_bits(mantissa_bits);

		using dec_to_bin_rounding_poli
			= detail::policy_impl::decimal_to_binary_rounding::nearest_to_even;
		using bin_to_dec_rounding_poli
			= detail::policy_impl::binary_to_decimal_rounding::to_even;
		using trailing_zero_poli = detail::policy_impl::trailing_zero::remove;
		using cache_poli = detail::policy_impl::cache::full;

		auto two_fc = significand_bits.remove_sign_bit_and_shift();
		auto exponent = int(exponent_bits);
			// Is the input a normal number?
		if (exponent != 0) {
			exponent += format::exponent_bias - format::significand_bits;

			// Shorter interval case; proceed like Schubfach.
			// One might think this condition is wrong,
			// since when exponent_bits == 1 and two_fc == 0,
			// the interval is actullay regular.
			// However, it turns out that this seemingly wrong condition
			// is actually fine, because the end result is anyway the same.
			//
			// [binary32]
			// floor( (fc-1/2) * 2^e ) = 1.175'494'28... * 10^-38
			// floor( (fc-1/4) * 2^e ) = 1.175'494'31... * 10^-38
			// floor(    fc    * 2^e ) = 1.175'494'35... * 10^-38
			// floor( (fc+1/2) * 2^e ) = 1.175'494'42... * 10^-38
			//
			// Hence, shorter_interval_case will return 1.175'494'4 * 10^-38.
			// 1.175'494'3 * 10^-38 is also a correct shortest representation
			// that will be rejected if we assume shorter interval,
			// but 1.175'494'4 * 10^-38 is closer to the true value so it doesn't matter.
			//
			// [binary64]
			// floor( (fc-1/2) * 2^e ) = 2.225'073'858'507'201'13... * 10^-308
			// floor( (fc-1/4) * 2^e ) = 2.225'073'858'507'201'25... * 10^-308
			// floor(    fc    * 2^e ) = 2.225'073'858'507'201'38... * 10^-308
			// floor( (fc+1/2) * 2^e ) = 2.225'073'858'507'201'63... * 10^-308
			//
			// Hence, shorter_interval_case will return 2.225'073'858'507'201'4 * 10^-308.
			// This is indeed of the shortest length, and it is the unique one
			// closest to the true value among valid representations of the same length.
			static_assert(std::is_same<format, ieee754_binary32>::value ||
				std::is_same<format, ieee754_binary64>::value, "");

			if (two_fc == 0) {
				return detail::impl<Float, FloatTraits>::template compute_nearest_shorter
						< return_type<Float>
						, typename dec_to_bin_rounding_poli::shorter_interval_type
						, trailing_zero_poli
						, bin_to_dec_rounding_poli
						, cache_poli >
					( exponent
					, dec_to_bin_rounding_poli::get_shorter_interval_type(significand_bits) );
			}
			two_fc |= (decltype(two_fc)(1) << (format::significand_bits + 1));
		}
		// Is the input a subnormal number?
		else {
			exponent = format::min_exponent - format::significand_bits;
		}

		return detail::impl<Float, FloatTraits>::template compute_nearest_normal
				< return_type<Float>
				, typename dec_to_bin_rounding_poli::normal_interval_type
				, trailing_zero_poli
				, bin_to_dec_rounding_poli
				, cache_poli >
			( two_fc, exponent
			, dec_to_bin_rounding_poli::get_normal_interval_type(significand_bits) );
	}

} // namespace dragonbox
} // namespace jkj
} // namespace detail
} // namespace strf

#undef JKJ_HAS_COUNTR_ZERO_INTRINSIC
#undef JKJ_FORCEINLINE
#undef JKJ_SAFEBUFFERS
#undef JKJ_IF_CONSTEXPR
#undef JKJ_WITH_INTERNAL_INT128
#if defined(JKJ_MSC_WARNING_4127_DISABLED)
#pragma warning(pop)
#undef JKJ_MSC_WARNING_4127_DISABLED
#endif

#endif

