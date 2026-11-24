// The analytic conformal equations of state (eos_type "conformal" and
// "conformal_diagonal") and the online inversion from (e or s, rho_B, rho_S, rho_Q)
// back to (T, mu_B, mu_Q, mu_S).
//
// Units are those of the code: fm^-1 for T and mu, fm^-4 for e and p, fm^-3 for
// s and the charge densities.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "eos.h"
#include "settings.h"

namespace {

constexpr double TOL = 1.e-12;   // round-off for the analytic identities
// The root finder stops when |e(T,mu) - e| + sum |rho(T,mu) - rho| < 1e-10
// (TOLERANCE in include/rootfinder.h, absolute, in the units above). That is its
// own contract; how close (T, mu) come back depends on the conditioning and is
// only reported.
constexpr double ROOTFINDER_TOLERANCE = 1.e-10;

struct Point { double T, muB, muQ, muS; };

const std::vector<Point> points = {
    {1.0,  0.0,  0.0,  0.0},
    {0.8,  0.5,  0.1,  0.2},
    {1.5, -0.7,  0.3, -0.2},
    {0.6,  0.9, -0.2,  0.1},
};

std::shared_ptr<ccake::EquationOfState> make_eos(const std::string& type){
    auto settingsPtr = std::make_shared<Settings>();
    settingsPtr->eos_type = type;
    auto eos = std::make_shared<ccake::EquationOfState>();
    eos->set_SettingsPtr(settingsPtr);
    eos->init();
    return eos;
}

::testing::AssertionResult close(double value, double expected){
    if (std::fabs(value - expected) <= TOL*(1. + std::fabs(expected)))
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << value << " instead of " << expected;
}

// For a conformal fluid the trace of the energy-momentum tensor vanishes,
// e = 3p, the speed of sound is c_s^2 = 1/3, and, p being homogeneous of degree
// 4 in (T, mu), e + p = 4p = T s + mu_B n_B + mu_Q n_Q + mu_S n_S.
void check_conformal_identities(const std::string& type){
    auto eos = make_eos(type);
    for (const Point& x : points){
        eos->tbqs(x.T, x.muB, x.muQ, x.muS, type);
        const std::string at = type + " at T = " + std::to_string(x.T)
                             + ", muB = " + std::to_string(x.muB);
        EXPECT_TRUE(close(eos->e(), 3.*eos->p())) << "e = 3p, " << at;
        EXPECT_TRUE(close(eos->cs2(), 1./3.)) << "cs2 = 1/3, " << at;
        EXPECT_TRUE(close(eos->w(), eos->e() + eos->p())) << "w = e + p, " << at;
        EXPECT_TRUE(close(eos->e() + eos->p(),
                          x.T*eos->s() + x.muB*eos->B() + x.muQ*eos->Q() + x.muS*eos->S()))
            << "e + p = Ts + mu n, " << at;
        EXPECT_GT(eos->p(), 0.) << at;
        EXPECT_GT(eos->s(), 0.) << at;
    }
}

double residual(const ccake::EquationOfState& eos, double target, bool energy,
                double B, double S, double Q){
    const double first = energy ? eos.e() : eos.s();
    return std::fabs(first - target) + std::fabs(eos.B() - B)
         + std::fabs(eos.S() - S) + std::fabs(eos.Q() - Q);
}

} // namespace

TEST(ConformalEoS, thermodynamic_identities){ check_conformal_identities("conformal"); }
TEST(ConformalDiagonalEoS, thermodynamic_identities){ check_conformal_identities("conformal_diagonal"); }

/// Densities computed at a known (T, mu) are inverted back by the online root
/// finder, from the energy density and from the entropy density.
TEST(EoSInversion, round_trip_through_the_root_finder){
    auto eos = make_eos("conformal");
    for (const Point& x : points){
        eos->tbqs(x.T, x.muB, x.muQ, x.muS, "conformal");
        const double e = eos->e(), s = eos->s();
        const double B = eos->B(), S = eos->S(), Q = eos->Q();
        const std::string at = "T = " + std::to_string(x.T) + ", muB = " + std::to_string(x.muB);

        bool found = false;
        const double s_from_e = eos->s_out(e, B, S, Q, found);
        ASSERT_TRUE(found) << "no solution from e at " << at;
        EXPECT_LE(residual(*eos, e, true, B, S, Q), ROOTFINDER_TOLERANCE) << "from e at " << at;
        std::printf("[ MEASURED ] from e at %s: |dT|/T = %.2e, |dmuB| = %.2e fm^-1, |ds|/s = %.2e\n",
                    at.c_str(), std::fabs(eos->T() - x.T)/x.T, std::fabs(eos->muB() - x.muB),
                    std::fabs(s_from_e - s)/s);

        ASSERT_TRUE(eos->update_s(s, B, S, Q)) << "no solution from s at " << at;
        EXPECT_LE(residual(*eos, s, false, B, S, Q), ROOTFINDER_TOLERANCE) << "from s at " << at;
        std::printf("[ MEASURED ] from s at %s: |dT|/T = %.2e, |dmuB| = %.2e fm^-1\n",
                    at.c_str(), std::fabs(eos->T() - x.T)/x.T, std::fabs(eos->muB() - x.muB));
    }
}
