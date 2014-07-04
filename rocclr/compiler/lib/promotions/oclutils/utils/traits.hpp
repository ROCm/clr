//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef TRAITS_HPP_
#define TRAITS_HPP_

namespace amd {

// Type traits:

//! \cond ignore
template <typename T>
struct is_pointer
{ static const bool value = false; };

template <typename T>
struct is_pointer<T*>
{ static const bool value = true; };

template <typename T>
struct remove_reference
{ typedef T type; };

template <typename T>
struct remove_reference<T&>
{ typedef T type; };

template <typename T>
struct remove_volatile
{ typedef T type; };

template <typename T>
struct remove_volatile<T volatile>
{ typedef T type; };

template <typename T>
struct remove_const
{ typedef T type; };

template <typename T>
struct remove_const<T const>
{ typedef T type; };

template <typename T>
struct remove_pointer
{ typedef T type; };

template <typename T>
struct remove_pointer<T*>
{ typedef T type; };

template <typename T>
struct add_const
{ typedef T const type; };

template <typename T>
struct add_const<T&>
{ typedef T& type; };

template <typename T>
struct add_volatile
{ typedef T volatile type; };

template <typename T>
struct add_volatile<T&>
{ typedef T& type; };

template <typename T>
struct add_pointer
{ typedef typename remove_reference<T>::type* type; };

template <typename T>
struct add_reference
{ typedef typename remove_reference<T>::type& type; };

template <>
struct add_reference<void>
{ typedef void type; };

template <>
struct add_reference<const void>
{ typedef const void type; };

template <>
struct add_reference<volatile void>
{ typedef volatile void type; };

template <>
struct add_reference<const volatile void>
{ typedef const volatile void type; };

template <typename T>
struct make_arithmetic
{ typedef typename remove_volatile<T>::type type; };

template <typename T>
struct make_arithmetic<T*>
{ typedef long int type; };

template <typename T>
struct make_arithmetic<T&>
{ typedef typename make_arithmetic<T>::type type; };
//! \endcond

} // namespace amd

#endif /* TRAITS_HPP_ */
