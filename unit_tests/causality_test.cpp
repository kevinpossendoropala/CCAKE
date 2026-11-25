#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "milne.hpp"          // eom_causality.hpp is templated on the Matrix type
#include "eom_causality.hpp"

/// @file causality_test.cpp
/// @brief Regression tests for the shear-eigenvalue extraction behind the nonlinear
/// causality check (eom_causality.hpp).

using namespace ccake::causality;

//==============================================================================
// Lambda extraction
//==============================================================================

/// pi^mu_nu is traceless, so the extracted eigenvalues must sum to zero -- by
/// construction of the depressed cubic, not merely to round-off luck.
TEST( CausalityLambda, Traceless )
{
  std::mt19937_64 rng(20260716);
  std::uniform_real_distribution<double> u(-0.5, 0.5);

  for ( int trial = 0; trial < 200; ++trial )
  {
    const double t2 = 1.0 + 3.0*std::fabs(u(rng));

    // a symmetric pi^{mu nu} that is traceless w.r.t. g = diag(1,-1,-1,-t2):
    // pi^33 = (pi^00 - pi^11 - pi^22)/t2, exactly as reset_pi_tensor builds it.
    ccake::milne::Matrix<double,4,4> shv;
    for ( int i = 0; i < 4; ++i )
      for ( int j = 0; j <= i; ++j )
      {
        const double val = u(rng);
        shv(i,j) = val; shv(j,i) = val;
      }
    shv(3,3) = (shv(0,0) - shv(1,1) - shv(2,2))/t2;

    double Lam[3];
    shear_eigenvalues(shv, t2, Lam);

    EXPECT_NEAR( Lam[0] + Lam[1] + Lam[2], 0.0, 1.0e-10 ) << "trial " << trial;
    EXPECT_LE( Lam[0], Lam[1] );
    EXPECT_LE( Lam[1], Lam[2] );
  }
}

/// Pins the metric-signature convention, which is the subtlest part of this code.
///
/// The paper is mostly-plus, CCAKE is mostly-minus. The contravariant pi^{ij} is the
/// same physical object in both (it is the deviation of T^{ij} from the isotropic
/// part), but the MIXED tensor pi^mu_nu flips sign between them -- so the paper's
/// Lambda_a are MINUS the eigenvalues of CCAKE's mixed tensor. Equivalently, and
/// this is what we assert: in the local rest frame the paper's Lambda_a are just the
/// eigenvalues of the physical spatial shear stress pi^{ij}.
///
/// Get this backwards and every condition, old and new, is silently evaluated on
/// -Lambda. Nothing else in the suite would catch it, because the conditions are not
/// symmetric under Lambda -> -Lambda but they are all still *computable*.
TEST( CausalityLambda, RestFrameSignConvention )
{
  const double t2 = 1.0;   // tau = 1, so the eta-eta slot carries no extra factor

  ccake::milne::Matrix<double,4,4> shv;
  for ( int i = 0; i < 4; ++i ) for ( int j = 0; j < 4; ++j ) shv(i,j) = 0.0;

  // A diagonal rest-frame shear stress. Traceless w.r.t. g = diag(1,-1,-1,-1):
  //   g_{mn} pi^{mn} = 0 - 0.3 - 0.5 + 0.8 = 0
  shv(1,1) =  0.3;
  shv(2,2) =  0.5;
  shv(3,3) = -0.8;

  double Lam[3];
  shear_eigenvalues(shv, t2, Lam);

  // the eigenvalues of pi^{ij} itself, ascending -- NOT their negatives
  EXPECT_NEAR( Lam[0], -0.8, 1.0e-12 );
  EXPECT_NEAR( Lam[1],  0.3, 1.0e-12 );
  EXPECT_NEAR( Lam[2],  0.5, 1.0e-12 );
}

/// Vanishing shear stress must give Lambda = 0 rather than a division blow-up.
TEST( CausalityLambda, ZeroShear )
{
  ccake::milne::Matrix<double,4,4> shv;
  for ( int i = 0; i < 4; ++i ) for ( int j = 0; j < 4; ++j ) shv(i,j) = 0.0;

  double Lam[3];
  shear_eigenvalues(shv, 1.0, Lam);
  for ( int k = 0; k < 3; ++k ) EXPECT_DOUBLE_EQ( Lam[k], 0.0 );
}

//==============================================================================
// Verdict of the necessary and sufficient conditions (classify)
//==============================================================================

namespace {

/// A state with e + p = 4 (e = 3, p = 1), c_s^2 = 1/3, no bulk pressure, and no
/// second-order coefficients; the shear eigenvalues are given.
Inputs state(double eta, double tau_pi, double zeta, double tau_Pi,
             double L1 = 0., double L2 = 0., double L3 = 0.)
{
  Inputs in{};
  in.Lam[0] = L1; in.Lam[1] = L2; in.Lam[2] = L3;
  in.cs2 = 1./3.;
  in.e = 3.; in.p = 1.; in.Pi = 0.;
  in.eta = eta; in.zeta = zeta; in.tau_pi = tau_pi; in.tau_Pi = tau_Pi;
  return in;
}

} // namespace

// At equilibrium (no shear stress, no bulk pressure) the conditions reduce to the
// linear causality bound of Israel-Stewart theory,
//   (4/3) eta/(tau_pi w) + zeta/(tau_Pi w) + c_s^2 <= 1,   w = e + p,
// which does not depend on how the full conditions are written out.

/// eta/(tau_pi w) = 0.2 (tau_pi = 5 eta/w): 4/3*0.2 + 1/3 = 0.6 < 1.
TEST( CausalityVerdict, EquilibriumWithinTheLinearBoundIsCausal )
{
  EXPECT_EQ( classify(state(0.8, 1.0, 0.0, 1.0)), CAUSAL );
}

/// eta/(tau_pi w) = 0.6: 4/3*0.6 + 1/3 = 1.13 > 1, signals faster than light.
TEST( CausalityVerdict, TooLargeShearRatioIsAcausal )
{
  EXPECT_EQ( classify(state(2.4, 1.0, 0.0, 1.0)), ACAUSAL );
}

/// zeta/(tau_Pi w) = 0.8 with small shear: 1/3 + 0.8 > 1.
TEST( CausalityVerdict, TooLargeBulkRatioIsAcausal )
{
  EXPECT_EQ( classify(state(0.04, 1.0, 3.2, 1.0)), ACAUSAL );
}

/// Condition (4a): 2 eta >= tau_pi |Lambda_1| / 2. With eta/tau_pi = 0.2 a
/// shear eigenvalue of -1 violates it, however small the bulk part is.
TEST( CausalityVerdict, StrongShearStressViolatesNecessaryCondition )
{
  EXPECT_EQ( classify(state(0.2, 1.0, 0.0, 1.0, -1.0, 0.25, 0.75)), ACAUSAL );
}

/// A mild shear stress, Lambda = (-0.1, 0.025, 0.075), with eta/tau_pi = 0.8:
/// every sufficient condition (5a)-(5h) holds (checked by hand for this state).
TEST( CausalityVerdict, MildShearStressIsCausal )
{
  EXPECT_EQ( classify(state(0.8, 1.0, 0.0, 1.0, -0.1, 0.025, 0.075)), CAUSAL );
}

/// Condition (5c), tau_pipi <= 6 delta_pipi, is only sufficient: violating it
/// at equilibrium leaves the state undetermined, not acausal.
TEST( CausalityVerdict, MissingSufficientConditionIsUndetermined )
{
  Inputs in = state(0.8, 1.0, 0.0, 1.0);
  in.tau_pipi = 0.5;
  in.delta_pipi = 0.0;
  EXPECT_EQ( classify(in), UNDETERMINED );
}

/// Negative transport coefficients fail the sign conditions.
TEST( CausalityVerdict, NegativeViscosityIsAcausal )
{
  EXPECT_EQ( classify(state(-0.1, 1.0, 0.0, 1.0)), ACAUSAL );
}

/// Without a relaxation time the verdict cannot be computed.
TEST( CausalityVerdict, ZeroRelaxationTimeIsAnError )
{
  EXPECT_EQ( classify(state(0.8, 0.0, 0.0, 1.0)), ERROR );
  EXPECT_EQ( classify(state(0.8, 1.0, 0.0, 0.0)), ERROR );
}
