#include "kernel_reference.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "kernel.h"
#include "vector.h"

#define TOL 1.E-14

TEST( KernelTest, Kernel1D_q_0p5 )
{
  double dist = .5;
  double h = 1.;
  EXPECT_LT( fabs(ccake::SPHkernel<1>::kernel(dist,h)-23./48.) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<1>::kernel(dist,h) 
  << " instead of expected "  << 23./48.;
}

TEST( KernelTest, Kernel1D_q_p10 )
{
  double dist = 1.;
  double h = 1.;
  EXPECT_LT( fabs(ccake::SPHkernel<1>::kernel(dist,h)-1./6.) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<1>::kernel(dist,h) 
  << " instead of expected "  << 1./6.;
}

TEST( KernelTest, Kernel1D_q_1p5 )
{
  double dist = 1.5;
  double h = 1;
  EXPECT_LT( fabs(ccake::SPHkernel<1>::kernel(dist,h)-1./48.) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<1>::kernel(dist,h) 
  << " instead of expected "  << 1./48.;
}

TEST( KernelTest, Kernel1D_q_2p0 )
{
  double dist = 2.0;
  double h = 1;
  EXPECT_LT( fabs(ccake::SPHkernel<1>::kernel(dist,h)) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<1>::kernel(dist,h) 
  << " instead of expected "  << 0.;
}


TEST( KernelTest, Kernel2D_q_0p5 )
{
  double dist = .5;
  double h = 1.;
  EXPECT_LT( fabs(ccake::SPHkernel<2>::kernel(dist,h)-115./112./M_PI) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<2>::kernel(dist,h) 
  << " instead of expected "  << 115./112./M_PI;
}

TEST( KernelTest, Kernel2D_q_1p0 )
{
  double dist = 1.;
  double h = 1.;
  EXPECT_LT( fabs(ccake::SPHkernel<2>::kernel(dist,h)-5./14./M_PI) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<2>::kernel(dist,h) 
  << " instead of expected "  << 115./112./M_PI;
}

TEST( KernelTest, Kernel2D_q_1p5 )
{
  double dist = 1.5;
  double h = 1;
  EXPECT_LT( fabs(ccake::SPHkernel<2>::kernel(dist,h)-5./112./M_PI) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<2>::kernel(dist,h) 
  << " instead of expected "  << 5./112./M_PI;
}

TEST( KernelTest, Kernel2D_q_2p0 )
{
  double dist = 2.0;
  double h = 1;
  EXPECT_LT( fabs(ccake::SPHkernel<2>::kernel(dist,h)) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<2>::kernel(dist,h) 
  << " instead of expected "  << 0.;
}


TEST( KernelTest, Kernel3D_q_0p5 )
{
  double dist = .5;
  double h = 1.;
  EXPECT_LT( fabs(ccake::SPHkernel<3>::kernel(dist,h)-23./32./M_PI) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<3>::kernel(dist,h) 
  << " instead of expected "  << 23./32./M_PI;
}

TEST( KernelTest, Kernel3D_q_1p0 )
{
  double dist = 1.;
  double h = 1.;
  EXPECT_LT( fabs(ccake::SPHkernel<3>::kernel(dist,h)-1./4./M_PI) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<3>::kernel(dist,h) 
  << " instead of expected "  << 1./4./M_PI;
}

TEST( KernelTest, Kernel3D_q_1p5 )
{
  double dist = 1.5;
  double h = 1;
  EXPECT_LT( fabs(ccake::SPHkernel<3>::kernel(dist,h)-1./32./M_PI) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<3>::kernel(dist,h) 
  << " instead of expected "  << 1./32./M_PI;
}

TEST( KernelTest, Kernel3D_q_2p0 )
{
  double dist = 2.0;
  double h = 1;
  EXPECT_LT( fabs(ccake::SPHkernel<3>::kernel(dist,h)) , TOL )
  << "Kernel value is "<< ccake::SPHkernel<3>::kernel(dist,h) 
  << " instead of expected "  << 0.;
}

TEST( GradKernelTest, GradKernel2D_q_0p5 )
{
  double pos_a[2] = {.25,.0};
  double pos_b[2] = {-.25,.0};
  double rel_pos[2];
  double const target_val = -75./56./M_PI;
  double h = 1;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;
  
  double pos_a2[2] = {0.,.25};
  double pos_b2[2] = {0.,-.25};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
  
}

TEST( GradKernelTest, GradKernel2D_q_0p5_reversed )
{
  double pos_b[2] = {.25,.0};
  double pos_a[2] = {-.25,.0};
  double rel_pos[2];
  double const target_val = 75./56./M_PI;
  double h = 1;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;
  
  double pos_b2[2] = {0.,.25};
  double pos_a2[2] = {0.,-.25};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
  
}

TEST( GradKernelTest, GradKernel2D_q_1p0 )
{
  double pos_a[2] = {.5,.0};
  double pos_b[2] = {-.5,.0};
  double rel_pos[2];
  double const target_val = -15./14./M_PI;
  double h = 1;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;
  
  double pos_a2[2] = {0.,.5};
  double pos_b2[2] = {0.,-.5};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
  
}

TEST( GradKernelTest, GradKernel2D_q_1p5 )
{
  double pos_a[2] = {.75,.0};
  double pos_b[2] = {-.75,.0};
  double rel_pos[2];
  double const target_val = -15./56./M_PI;
  double h = 1;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;
  
  double pos_a2[2] = {0.,.75};
  double pos_b2[2] = {0.,-.75};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
  
}

TEST( GradKernelTest, GradKernel2D_q_1p5_reversed )
{
  double pos_b[2] = {.75,.0};
  double pos_a[2] = {-.75,.0};
  double rel_pos[2];
  double const target_val = 15./56./M_PI;
  double h = 1;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;

  double pos_b2[2] = {0.,.75};
  double pos_a2[2] = {0.,-.75};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
}

TEST( GradKernelTest, GradKernel2D_q_0p0 )
{
  double pos_a[2] = {.75,.0};
  double pos_b[2] = {.75,.0};
  double rel_pos[2];
  double const target_val = 0;
  double h = 1;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;
  
  double pos_a2[2] = {0.,.75};
  double pos_b2[2] = {0.,.75};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
  
}

TEST( GradKernelTest, GradKernel2D_q_1p5_half_h )
{
  double pos_a[2] = {.75/2,.0};
  double pos_b[2] = {-.75/2,.0};
  double rel_pos[2];
  double const h = .5;
  double const target_val = -15./56./M_PI/h/h/h;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;
  
  double pos_a2[2] = {0.,.75/2};
  double pos_b2[2] = {0.,-.75/2};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
  
}

TEST( GradKernelTest, GradKernel2D_q_0p5_half_h )
{
  double pos_a[2] = {.25/2,.0};
  double pos_b[2] = {-.25/2,.0};
  double rel_pos[2];
  double const h = .5;
  double const target_val = -75./56./M_PI/h/h/h;
  double gradK[2];
  
  double dist = ccake::SPHkernel<2>::distance(pos_a,pos_b);
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a[idir] - pos_b[idir];
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  
  EXPECT_LT( fabs(gradK[0] - target_val), TOL) << "gradK[0] = " << gradK[0] <<". Expected " << target_val;
  EXPECT_LT( fabs(gradK[1]), TOL) << "gradK[1] = " << gradK[1] <<". Expected 0" << endl;
  
  double pos_a2[2] = {0.,.25/2};
  double pos_b2[2] = {0.,-.25/2};
  for (int idir=0; idir<2;++idir) rel_pos[idir] = pos_a2[idir] - pos_b2[idir];
  dist = ccake::SPHkernel<2>::distance(pos_a2,pos_b2);
  ccake::SPHkernel<2>::gradKernel(rel_pos,dist,h,gradK);
  EXPECT_LT( fabs(gradK[0]), TOL) << "gradK[0] = " << gradK[0] <<". Expected 0" << endl;
  EXPECT_LT( fabs(gradK[1] - target_val), TOL) << "gradK[1] = " << gradK[1] <<". Expected " << target_val;
  
}

////////////////////////////////////////////////////////////////////////////////
// Checks against an independent statement of the cubic spline kernel
//   W(r, h) = sigma_D / h^D * f(q),  q = r/h,
//   f(q) = 1 - 3/2 q^2 + 3/4 q^3  (q < 1),   1/4 (2 - q)^3  (1 <= q < 2),   0  (q >= 2),
//   sigma_1 = 2/3, sigma_2 = 10/(7 pi), sigma_3 = 1/pi,
// and its gradient  grad W = sigma_D / h^(D+1) * f'(q) * (r_a - r_b)/r.
// All comparisons are at round-off level.
////////////////////////////////////////////////////////////////////////////////
// sigma, f, fprime, W and dW come from kernel_reference.h
namespace kernel_reference {

// relative comparison that also works near zero
::testing::AssertionResult close(double value, double expected){
  if (std::fabs(value - expected) <= TOL*(1. + std::fabs(expected)))
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << value << " instead of " << expected;
}

// unit vectors in D dimensions used as directions of separation
std::vector<std::vector<double>> directions(int D){
  if (D == 1) return {{1.}, {-1.}};
  if (D == 2) return {{1., 0.}, {0., -1.}, {0.6, 0.8}, {-0.28, 0.96}};
  return {{1., 0., 0.}, {0., 0., -1.}, {2./3., -1./3., 2./3.}, {0.48, 0.6, 0.64}};
}

template <unsigned int D>
void check_values_and_gradients(){
  for (double h : {0.3, 1.0, 2.5})
  for (double q = 0.; q <= 2.3; q += 0.05)
  {
    const double r = q*h;
    EXPECT_TRUE(close(ccake::SPHkernel<D>::kernel(r, h), W(D, r, h)))
        << "W, D = " << D << ", q = " << q << ", h = " << h;
    for (const std::vector<double>& n : directions(D)){
      double rel[D], grad[D];
      for (unsigned i = 0; i < D; ++i) rel[i] = r*n[i];
      ccake::SPHkernel<D>::gradKernel(rel, r, h, grad);
      for (unsigned i = 0; i < D; ++i){
        const double expected = (r > 0.) ? dW(D, r, h)*n[i] : 0.;
        EXPECT_TRUE(close(grad[i], expected))
            << "grad W component " << i << ", D = " << D << ", q = " << q << ", h = " << h;
      }
    }
  }
}

// Integral of W over all space, by Gauss-Legendre quadrature on [0, h] and [h, 2h].
// The integrand S_D r^(D-1) f(r/h) is a polynomial of degree <= 5 on each piece,
// which 4-point Gauss-Legendre integrates exactly.
template <unsigned int D>
double integral(double h){
  const double node[4]   = {-0.8611363115940526, -0.3399810435848563,
                             0.3399810435848563,  0.8611363115940526};
  const double weight[4] = { 0.3478548451374538,  0.6521451548625461,
                             0.6521451548625461,  0.3478548451374538};
  const double surface = D == 1 ? 2. : (D == 2 ? 2.*M_PI : 4.*M_PI);
  double sum = 0.;
  for (double a : {0., h}){
    for (int k = 0; k < 4; ++k){
      const double r = a + 0.5*h*(node[k] + 1.);
      sum += 0.5*h*weight[k]*surface*std::pow(r, D-1)*ccake::SPHkernel<D>::kernel(r, h);
    }
  }
  return sum;
}

} // namespace kernel_reference

TEST( KernelReference, Values1D ) { kernel_reference::check_values_and_gradients<1>(); }
TEST( KernelReference, Values2D ) { kernel_reference::check_values_and_gradients<2>(); }
TEST( KernelReference, Values3D ) { kernel_reference::check_values_and_gradients<3>(); }

/// The kernel integrates to one in every dimension and for every smoothing length.
TEST( KernelNormalization, IntegratesToOne )
{
  for (double h : {0.1, 0.3, 1.0, 2.5}){
    EXPECT_TRUE(kernel_reference::close(kernel_reference::integral<1>(h), 1.)) << "1D, h = " << h;
    EXPECT_TRUE(kernel_reference::close(kernel_reference::integral<2>(h), 1.)) << "2D, h = " << h;
    EXPECT_TRUE(kernel_reference::close(kernel_reference::integral<3>(h), 1.)) << "3D, h = " << h;
  }
}

/// W and its derivative are continuous where the two pieces of the spline meet.
TEST( KernelProperties, ContinuousAtQEqualsOne )
{
  const double h = 0.7;
  // value and slope of the inner piece, evaluated at q = 1
  const double inner_W  = kernel_reference::sigma(2)/(h*h)*(1. - 1.5 + 0.75);
  const double inner_dW = kernel_reference::sigma(2)/(h*h*h)*(-3. + 2.25);
  EXPECT_TRUE(kernel_reference::close(ccake::SPHkernel<2>::kernel(h, h), inner_W));
  double rel[2] = {h, 0.}, grad[2];
  ccake::SPHkernel<2>::gradKernel(rel, h, h, grad);
  EXPECT_TRUE(kernel_reference::close(grad[0], inner_dW));
}

/// Swapping the two particles reverses the gradient, exactly.
TEST( KernelProperties, GradientIsAntisymmetric )
{
  const double h = 0.8;
  for (double q : {0.2, 0.9, 1.0, 1.4, 1.99}){
    double rel[3] = {0.48*q*h, 0.6*q*h, 0.64*q*h};
    double minus_rel[3] = {-rel[0], -rel[1], -rel[2]};
    double grad[3], minus_grad[3];
    ccake::SPHkernel<3>::gradKernel(rel, q*h, h, grad);
    ccake::SPHkernel<3>::gradKernel(minus_rel, q*h, h, minus_grad);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(minus_grad[i], -grad[i]) << "q = " << q;
  }
}

/// Nothing beyond two smoothing lengths.
TEST( KernelProperties, CompactSupport )
{
  for (double q : {2.0, 2.000001, 3.5}){
    const double h = 1.3, r = q*h;
    EXPECT_EQ(ccake::SPHkernel<1>::kernel(r, h), 0.);
    EXPECT_EQ(ccake::SPHkernel<2>::kernel(r, h), 0.);
    EXPECT_EQ(ccake::SPHkernel<3>::kernel(r, h), 0.);
    double rel[2] = {r, 0.}, grad[2] = {1., 1.};
    ccake::SPHkernel<2>::gradKernel(rel, r, h, grad);
    EXPECT_EQ(grad[0], 0.);
    EXPECT_EQ(grad[1], 0.);
  }
}

/// In 3+1D the kernel is the product of a transverse 2D kernel (width hT) and a
/// longitudinal 1D kernel in eta_s (width hEta).
TEST( FactorizedKernel3D, IsTransverseTimesLongitudinal )
{
  using namespace kernel_reference;
  const double hT = 0.4, hEta = 0.25;
  const double r1[3] = {0.10, -0.20, 0.30};
  for (double dx : {0.0, 0.15, 0.5})
  for (double dy : {0.0, -0.3})
  for (double deta : {0.0, 0.1, -0.3, 0.6}){
    const double r2[3] = {r1[0] + dx, r1[1] + dy, r1[2] + deta};
    const double expected = W(2, std::hypot(dx, dy), hT)*W(1, std::fabs(deta), hEta);
    EXPECT_TRUE(close(ccake::SPHkernel<3>::kernel(r1, r2, hT, hEta), expected))
        << "dx = " << dx << ", dy = " << dy << ", deta = " << deta;
  }
}

/// Its gradient follows the product rule.
TEST( FactorizedKernel3D, GradientFollowsProductRule )
{
  using namespace kernel_reference;
  const double hT = 0.4, hEta = 0.25;
  const double pos_b[3] = {0.0, 0.0, 0.0};
  for (double x : {0.12, -0.5})
  for (double y : {0.0, 0.3})
  for (double eta : {0.05, -0.2, 0.4}){
    const double pos_a[3] = {x, y, eta};
    const double rel[3] = {x, y, eta};
    double grad[3];
    ccake::SPHkernel<3>::gradKernel(rel, pos_a, pos_b, hT, hEta, grad);
    const double rT = std::hypot(x, y), re = std::fabs(eta);
    const double WT = W(2, rT, hT), WE = W(1, re, hEta);
    EXPECT_TRUE(close(grad[0], dW(2, rT, hT)*x/rT*WE));
    EXPECT_TRUE(close(grad[1], (rT > 0. ? dW(2, rT, hT)*y/rT : 0.)*WE));
    EXPECT_TRUE(close(grad[2], WT*dW(1, re, hEta)*eta/re));
  }
}

/// ... and it integrates to one, by the same exact quadrature in r_T and eta_s.
TEST( FactorizedKernel3D, IntegratesToOne )
{
  const double node[4]   = {-0.8611363115940526, -0.3399810435848563,
                             0.3399810435848563,  0.8611363115940526};
  const double weight[4] = { 0.3478548451374538,  0.6521451548625461,
                             0.6521451548625461,  0.3478548451374538};
  const double hT = 0.4, hEta = 0.25;
  const double origin[3] = {0., 0., 0.};
  double sum = 0.;
  for (double aT : {0., hT})
  for (int i = 0; i < 4; ++i){
    const double rT = aT + 0.5*hT*(node[i] + 1.);
    for (double aE : {0., hEta})
    for (int j = 0; j < 4; ++j){
      const double eta = aE + 0.5*hEta*(node[j] + 1.);
      const double point[3] = {rT, 0., eta};
      // x2 for negative eta_s, 2 pi r_T for the transverse ring
      sum += 0.5*hT*weight[i]*0.5*hEta*weight[j]*2.*M_PI*rT*2.
             *ccake::SPHkernel<3>::kernel(point, origin, hT, hEta);
    }
  }
  EXPECT_TRUE(kernel_reference::close(sum, 1.));
}
