// An independent statement of the cubic-spline SPH kernel, for expected values
// in the tests (kernel_test.cpp checks the code's kernel against it):
//   W(r, h) = sigma_D / h^D f(r/h),
//   f(q) = 1 - 3/2 q^2 + 3/4 q^3 (q < 1),  1/4 (2 - q)^3 (1 <= q < 2),  0 beyond,
//   sigma_1 = 2/3, sigma_2 = 10/(7 pi), sigma_3 = 1/pi.
// Distances are Euclidean in the coordinates of the code ((x, y, eta_s) in 3+1D).
#ifndef CCAKE_UNIT_TESTS_KERNEL_REFERENCE_H
#define CCAKE_UNIT_TESTS_KERNEL_REFERENCE_H

#include <cmath>

namespace kernel_reference {

inline double sigma(int D){
  return D == 1 ? 2./3. : (D == 2 ? 10./(7.*M_PI) : 1./M_PI);
}
inline double f(double q){
  if (q >= 2.) return 0.;
  if (q >= 1.) return 0.25*(2.-q)*(2.-q)*(2.-q);
  return 1. - 1.5*q*q + 0.75*q*q*q;
}
inline double fprime(double q){
  if (q >= 2.) return 0.;
  if (q >= 1.) return -0.75*(2.-q)*(2.-q);
  return -3.*q + 2.25*q*q;
}
/// W(r, h) and dW/dr in D dimensions.
inline double W(int D, double r, double h){ return sigma(D)/std::pow(h, D)*f(r/h); }
inline double dW(int D, double r, double h){ return sigma(D)/std::pow(h, D+1)*fprime(r/h); }

} // namespace kernel_reference

#endif
