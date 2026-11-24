// Vector and matrix algebra of the device kernels (include/milne.hpp and
// include/cartesian.hpp). CCAKE uses the mostly-minus metric; in Milne
// coordinates (tau, x, y, eta_s) it is g = diag(1, -1, -1, -tau^2), in Cartesian
// coordinates diag(1, -1, -1, -1). Every expected value is exact.
#include <gtest/gtest.h>

#include "cartesian.hpp"
#include "milne.hpp"

namespace m = ccake::milne;
namespace c = ccake::cartesian;

/// Lowering a spatial index: x and y change sign, eta_s also picks up tau^2.
TEST( MilneVector, LoweringAndRaisingIndices )
{
  const double t2 = 4.;   // tau = 2
  m::Vector<double,3> u3;
  u3(0) = 1.; u3(1) = 2.; u3(2) = 3.;
  m::Vector<double,3> lowered = u3;
  lowered.make_covariant(t2);
  EXPECT_EQ( lowered(0), -1. );
  EXPECT_EQ( lowered(1), -2. );
  EXPECT_EQ( lowered(2), -12. );
  lowered.make_contravariant(t2);
  for (int i = 0; i < 3; ++i) EXPECT_EQ( lowered(i), u3(i) );

  // 2+1D: only x and y, no tau^2 anywhere
  m::Vector<double,2> u2;
  u2(0) = 1.; u2(1) = 2.;
  u2.make_covariant(t2);
  EXPECT_EQ( u2(0), -1. );
  EXPECT_EQ( u2(1), -2. );

  // 1+1D: the only direction is eta_s
  m::Vector<double,1> u1;
  u1(0) = 3.;
  u1.make_covariant(t2);
  EXPECT_EQ( u1(0), -12. );
}

/// u_i u^i = -(u^x^2 + u^y^2 + tau^2 u^eta^2), so gamma = sqrt(1 - u_i u^i).
TEST( MilneVector, ContractionUsesTheMetric )
{
  const double t2 = 4.;
  m::Vector<double,3> u, u_cov;
  u(0) = 1.; u(1) = 2.; u(2) = 3.;
  u_cov = u;
  u_cov.make_covariant(t2);
  EXPECT_EQ( m::contract(u_cov, u), -(1. + 4. + 4.*9.) );
}

/// delta_i_eta marks the eta_s direction; there is none in 2+1D.
TEST( MilneVector, EtaDirection )
{
  const m::Vector<double,1> d1 = m::delta_i_eta<1>();
  const m::Vector<double,2> d2 = m::delta_i_eta<2>();
  const m::Vector<double,3> d3 = m::delta_i_eta<3>();
  EXPECT_EQ( d1(0), 1. );
  EXPECT_EQ( d2(0), 0. );
  EXPECT_EQ( d2(1), 0. );
  EXPECT_EQ( d3(0), 0. );
  EXPECT_EQ( d3(1), 0. );
  EXPECT_EQ( d3(2), 1. );
}

/// Lowering both indices of a 4x4 tensor: the time row and column keep their
/// sign (g_00 = +1), x and y change it, eta_s picks up -tau^2 per index.
TEST( MilneMatrix, LoweringBothIndicesOfASpacetimeTensor )
{
  const double t2 = 4.;
  m::Matrix<double,4,4> pi;
  for (int i = 0; i < 4; ++i)
  for (int j = 0; j < 4; ++j)
    pi(i,j) = 1. + i + j;            // symmetric, no zero entries
  const m::Matrix<double,4,4> upper = pi;
  pi.make_covariant(0, t2);
  pi.make_covariant(1, t2);
  const double g[4] = {1., -1., -1., -t2};
  for (int i = 0; i < 4; ++i)
  for (int j = 0; j < 4; ++j)
    EXPECT_EQ( pi(i,j), g[i]*g[j]*upper(i,j) ) << "component " << i << j;
  pi.make_contravariant(0, t2);
  pi.make_contravariant(1, t2);
  for (int i = 0; i < 4; ++i)
  for (int j = 0; j < 4; ++j)
    EXPECT_EQ( pi(i,j), upper(i,j) ) << "component " << i << j;
}

/// The trace of a mixed 3x3 spatial matrix carries tau^2 on the eta_s entry.
TEST( MilneMatrix, TraceWeightsEta )
{
  m::Matrix<double,3,3> a;
  a(0,0) = 1.; a(1,1) = 2.; a(2,2) = 3.;
  EXPECT_EQ( m::tr(a, 4.), 1. + 2. + 4.*3. );
  m::Matrix<double,2,2> b;
  b(0,0) = 1.; b(1,1) = 2.;
  EXPECT_EQ( m::tr(b, 4.), 3. );
}

/// Inverses of the 2x2 and 3x3 matrices the equations of motion invert.
TEST( MilneMatrix, Inverse )
{
  m::Matrix<double,2,2> a;
  a(0,0) = 2.; a(0,1) = 1.; a(1,0) = 1.; a(1,1) = 3.;
  const m::Matrix<double,2,2> ai = m::inverse(a);
  EXPECT_DOUBLE_EQ( ai(0,0),  3./5. );
  EXPECT_DOUBLE_EQ( ai(0,1), -1./5. );
  EXPECT_DOUBLE_EQ( ai(1,0), -1./5. );
  EXPECT_DOUBLE_EQ( ai(1,1),  2./5. );

  // determinant 1, integer inverse
  m::Matrix<double,3,3> b;
  const double bv[3][3]  = {{1., 2., 3.}, {0., 1., 4.}, {5., 6., 0.}};
  const double inv[3][3] = {{-24., 18., 5.}, {20., -15., -4.}, {-5., 4., 1.}};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) b(i,j) = bv[i][j];
  const m::Matrix<double,3,3> bi = m::inverse(b);
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
    EXPECT_DOUBLE_EQ( bi(i,j), inv[i][j] ) << "component " << i << j;
}

/// The Cartesian classes have the same interface; the metric has no tau.
TEST( CartesianVector, LoweringIgnoresTau )
{
  c::Vector<double,3> u;
  u(0) = 1.; u(1) = 2.; u(2) = 3.;
  u.make_covariant(4.);
  EXPECT_EQ( u(0), -1. );
  EXPECT_EQ( u(1), -2. );
  EXPECT_EQ( u(2), -3. );
}

TEST( CartesianMatrix, LoweringBothIndicesOfASpacetimeTensor )
{
  c::Matrix<double,4,4> pi;
  for (int i = 0; i < 4; ++i)
  for (int j = 0; j < 4; ++j)
    pi(i,j) = 1. + i + j;
  const c::Matrix<double,4,4> upper = pi;
  pi.make_covariant(0, 4.);
  pi.make_covariant(1, 4.);
  const double g[4] = {1., -1., -1., -1.};
  for (int i = 0; i < 4; ++i)
  for (int j = 0; j < 4; ++j)
    EXPECT_EQ( pi(i,j), g[i]*g[j]*upper(i,j) ) << "component " << i << j;
}

TEST( CartesianMatrix, Inverse )
{
  c::Matrix<double,3,3> b;
  const double bv[3][3]  = {{1., 2., 3.}, {0., 1., 4.}, {5., 6., 0.}};
  const double inv[3][3] = {{-24., 18., 5.}, {20., -15., -4.}, {-5., 4., 1.}};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) b(i,j) = bv[i][j];
  const c::Matrix<double,3,3> bi = c::inverse(b);
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
    EXPECT_DOUBLE_EQ( bi(i,j), inv[i][j] ) << "component " << i << j;
}
