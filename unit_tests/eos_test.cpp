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

#include "constants.h"
#include "eos.h"
#include "eos_tanh_conformal.h"
#include "rootfinder.h"
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

// --------------------------------------------------------------- tanh conformal
// p = t(T) p_c with p_c = c T0^4 x^2, x = (T/T0)^2 + sum (mu/mu0)^2, the conformal
// pressure, and t = [1 + tanh((T - Tc)/Ts)]/2. So s = t s_c + t' p_c, n = t n_c,
// chi_XY = t d2p_c, chi_TX = t d2p_c/dTdX + t' n_X,c, chi_TT = t d2p_c/dT2 + 2 t' s_c + t'' p_c.

namespace {

const std::vector<double> everywhere_min = {0.0, -1e10, -1e10, -1e10};
const std::vector<double> everywhere_max = {1e10, 1e10, 1e10, 1e10};

std::vector<double> full_thermo(EoS_tanh_conformal& eos, double T, double muB, double muQ = 0.,
                                double muS = 0.){
    double point[4] = {T, muB, muQ, muS}, out[17];
    eos.get_full_thermo(point, out);
    return std::vector<double>(out, out + 17);
}

} // namespace

/// c = T0 = mu0 = 1, Tc = 1, Ts = 1/2, at (T, muB) = (1, 1): x = 2, p_c = 4, s_c = n_B,c = 8,
/// t = 1/2, t' = 1, t'' = 0. Every number is exact in binary.
TEST(TanhConformalEoS, values_at_the_transition){
    EoS_tanh_conformal eos(1., 1., 1., 1., 1., 1., 0.5, everywhere_min, everywhere_max);
    const auto r = full_thermo(eos, 1., 1.);
    EXPECT_EQ(r[0], 2.);   // p = t p_c
    EXPECT_EQ(r[1], 8.);   // s = t s_c + t' p_c = 4 + 4
    EXPECT_EQ(r[2], 4.);   // n_B
    EXPECT_EQ(r[3], 0.);   // n_S
    EXPECT_EQ(r[4], 0.);   // n_Q
    EXPECT_EQ(r[5], 10.);  // e = T s + muB n_B - p
    EXPECT_EQ(r[7], 8.);   // chi_BB = t 4c (x + 2 muB^2)
    EXPECT_EQ(r[8], 4.);   // chi_QQ = t 4c x
    EXPECT_EQ(r[9], 4.);   // chi_SS
    EXPECT_EQ(r[10], 0.);  // chi_BQ
    EXPECT_EQ(r[13], 12.); // chi_TB = t 8c T muB + t' n_B,c
    EXPECT_EQ(r[16], 24.); // chi_TT = t 4c (x + 2T^2) + 2 t' s_c + t'' p_c
    // With n_Q = n_S = 0 and no cross susceptibilities, the four-charge sound speed is the
    // (T, muB) one: (chi_BB s^2 - 2 chi_TB s n + chi_TT n^2)/(w det), w = 12, det = 48.
    EXPECT_TRUE(close(r[6], 2./9.));
    // and e - 3p = T t' p_c = 4: not conformal
    EXPECT_EQ(r[5] - 3.*r[0], 4.);

    double point[4] = {1., 1., 0., 0.}, e_side[4], s_side[4];
    eos.get_eBSQ(point, e_side);
    eos.get_sBSQ(point, s_side);
    EXPECT_EQ(e_side[0], 10.);
    EXPECT_EQ(e_side[1], 4.);
    EXPECT_EQ(s_side[0], 8.);
    EXPECT_EQ(s_side[1], 4.);
}

/// Far above Tc, t = 1 and t' = t'' = 0 exactly: the conformal identities hold.
TEST(TanhConformalEoS, conformal_far_above_the_transition){
    EoS_tanh_conformal eos(1., 1., 1., 1., 1., 0., 1e-3, everywhere_min, everywhere_max);
    for (const Point& x : points){
        const auto r = full_thermo(eos, x.T, x.muB, x.muQ, x.muS);
        EXPECT_TRUE(close(r[5], 3.*r[0])) << "e = 3p at T = " << x.T;
        EXPECT_TRUE(close(r[6], 1./3.)) << "cs2 = 1/3 at T = " << x.T;
        EXPECT_TRUE(close(r[5] + r[0], x.T*r[1] + x.muB*r[2] + x.muS*r[3] + x.muQ*r[4]));
    }
}

/// Far below Tc, t = t' = t'' = 0: nothing is left.
TEST(TanhConformalEoS, empty_far_below_the_transition){
    EoS_tanh_conformal eos(1., 1., 1., 1., 1., 1., 1e-3, everywhere_min, everywhere_max);
    const auto r = full_thermo(eos, 0.1, 0.2, 0.1, -0.1);
    for (int i : {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
        EXPECT_EQ(r[i], 0.) << "field " << i;
}

/// The production parameters (eos_type "tanh_conformal"): at mu = 0,
/// cs2 = s/(T chi_TT) = (2 + r/2)/(6 + 4r) at T = Tc, r = Tc/Ts = 220/120 = 11/6: 7/32.
TEST(TanhConformalEoS, production_speed_of_sound_at_Tc){
    auto eos = make_eos("tanh_conformal");
    const double Tc = 220./constants::hbarc_MeVfm;
    EXPECT_TRUE(close(eos->cs2out(Tc, "tanh_conformal"), 7./32.));
    // and the identities of any EoS
    std::vector<double> at{Tc, 0.3, 0.1, -0.1};  // (T, muB, muQ, muS)
    const auto r = eos->get_thermodynamics(at, "tanh_conformal");
    EXPECT_TRUE(close(r[5] + r[0], Tc*r[1] + 0.3*r[2] - 0.1*r[3] + 0.1*r[4]));
}

TEST(CoshConformalEoS, thermodynamic_identities){
    auto eos = make_eos("cosh_conformal");
    for (const Point& x : points){
        eos->tbqs(x.T, x.muB, x.muQ, x.muS, "cosh_conformal");
        EXPECT_TRUE(close(eos->w(), eos->e() + eos->p())) << "T = " << x.T;
        EXPECT_TRUE(close(eos->e() + eos->p(),
                          x.T*eos->s() + x.muB*eos->B() + x.muQ*eos->Q() + x.muS*eos->S()))
            << "e + p = Ts + mu n at T = " << x.T;
    }
}

TEST(EoSSetup, unknown_type_is_refused){
    EXPECT_THROW(make_eos("no such eos"), std::runtime_error);
}

// ------------------------------------------------- closed forms of the conformal EoS
/// p = c T^4 at mu = 0 with c = pi^2 (2 (Nc^2 - 1) + 7/2 Nc Nf)/90, Nc = 3, Nf = 2.5.
TEST(ConformalEoS, closed_forms_at_zero_density){
    auto eos = make_eos("conformal");
    const double c = M_PI*M_PI*(16. + 3.5*3.*2.5)/90.;
    const double T = 0.8;
    EXPECT_TRUE(close(eos->cs2out(T, "conformal"), 1./3.));
    EXPECT_TRUE(close(eos->cs2out(T, 0., 0., 0., "conformal"), 1./3.));
    EXPECT_TRUE(close(eos->wfz(T, "conformal"), 4.*c*std::pow(T, 4)));
    EXPECT_TRUE(close(eos->wfz(T, 0., 0., 0., "conformal"), 4.*c*std::pow(T, 4)));
    EXPECT_TRUE(close(eos->s_terms_T(T, "conformal"), 4.*c*std::pow(T, 3)));
    EXPECT_TRUE(close(eos->s_terms_T(T, 0., 0., 0., "conformal"), 4.*c*std::pow(T, 3)));
    EXPECT_TRUE(close(eos->sfreeze(T, "conformal"), 4.*c*std::pow(T, 3)));
    EXPECT_TRUE(close(eos->efreeze(T, "conformal"), 3.*c*std::pow(T, 4)));
    // w = 4cT^4, s = 4cT^3: dw/ds = 4T/3, and A = w - s dw/ds = -(4/3) c T^4
    eos->tbqs(T, 0., 0., 0., "conformal");
    EXPECT_TRUE(close(eos->dwds(), 4.*T/3.));
    EXPECT_TRUE(close(eos->A(), -4./3.*c*std::pow(T, 4)));
    // "default" is the configured EoS
    std::vector<double> at{T, 0., 0., 0.};
    EXPECT_TRUE(close(eos->get_thermodynamics(at, "default")[0], c*std::pow(T, 4)));
}

TEST(EoSRange, points_outside_an_eos){
    auto eos = make_eos("conformal");
    auto bounded = std::make_shared<EoS_tanh_conformal>(1., 1., 1., 1., 1., 1., 0.5,
        std::vector<double>{0.1, -1., -2., -3.}, std::vector<double>{2., 1., 2., 3.});
    EXPECT_FALSE(eos->point_not_in_range(1., 0., 0., 0., bounded));
    EXPECT_TRUE(eos->point_not_in_range(0.05, 0., 0., 0., bounded));
    EXPECT_TRUE(eos->point_not_in_range(3., 0., 0., 0., bounded));
    EXPECT_TRUE(eos->point_not_in_range(1., 1.5, 0., 0., bounded));
    EXPECT_TRUE(eos->point_not_in_range(1., 0., -2.5, 0., bounded));
    EXPECT_TRUE(eos->point_not_in_range(1., 0., 0., 3.5, bounded));
}

TEST(EoSRangeDeathTest, a_point_outside_stops_the_run){
    auto eos = make_eos("conformal");
    EXPECT_EXIT(eos->tbqs(-1., 0., 0., 0., "conformal"), ::testing::ExitedWithCode(8),
                "out of range");
}

// ------------------------------------------------------------ failed inversions
/// No EoS has a negative entropy density: every seed and every fallback EoS fails.
TEST(EoSInversion, impossible_entropy_is_refused){
    auto eos = make_eos("conformal");
    eos->tbqs(1., 0., 0., 0., "conformal");
    EXPECT_THROW(eos->update_s(-1., 0., 0., 0.), std::invalid_argument);
}

TEST(EoSInversion, impossible_energy_is_reported){
    auto eos = make_eos("conformal");
    eos->tbqs(1., 0., 0., 0., "conformal");
    bool found = true;
    eos->s_out(-1., 0., 0., 0., found);
    EXPECT_FALSE(found);
}

// ------------------------------------------------------------------- root finder
// The linear map (T, muB, muQ, muS) -> (2T, 3muB, 5muS, 7muQ), in the order the EoS
// functions return (e or s, n_B, n_S, n_Q). Its root is known exactly.
namespace {
void linear_map(double x[], double out[]){
    out[0] = 2.*x[0]; out[1] = 3.*x[1]; out[2] = 5.*x[3]; out[3] = 7.*x[2];
}
double linear_residual(const std::vector<double>& x, double e, double B, double S, double Q){
    double in[4] = {x[0], x[1], x[2], x[3]}, out[4];
    linear_map(in, out);
    return std::fabs(out[0] - e) + std::fabs(out[1] - B) + std::fabs(out[2] - S)
         + std::fabs(out[3] - Q);
}
} // namespace

TEST(Rootfinder, finds_the_root_of_a_linear_map){
    Rootfinder rf;
    std::vector<double> lo{0., -10., -10., -10.}, hi{10., 10., 10., 10.};
    for (const char* mode : {"energy", "e", "entropy", "s"}){
        std::vector<double> x{0.5, 0.5, 0.5, 0.5};
        ASSERT_TRUE(rf.find_root(mode, 2., 3., 10., -7., linear_map, lo, hi, x)) << mode;
        EXPECT_LE(linear_residual(x, 2., 3., 10., -7.), ROOTFINDER_TOLERANCE) << mode;
    }
}

TEST(Rootfinder, reduced_modes_keep_frozen_potentials_at_zero){
    Rootfinder rf;
    // muQ and muS frozen at 0 and no Q, S: baryon-only
    std::vector<double> lo{0., -10., 0., 0.}, hi{10., 10., 0., 0.};
    std::vector<double> x{0.5, 0.5, 0.3, 0.3};
    ASSERT_TRUE(rf.find_root("energy", 2., 3., 0., 0., linear_map, lo, hi, x));
    EXPECT_EQ(x[2], 0.);
    EXPECT_EQ(x[3], 0.);
    EXPECT_LE(linear_residual(x, 2., 3., 0., 0.), ROOTFINDER_TOLERANCE);
    // every potential frozen and no charge: temperature only
    std::vector<double> lo_T{0., 0., 0., 0.}, hi_T{10., 0., 0., 0.};
    std::vector<double> y{0.5, 0.3, 0.3, 0.3};
    ASSERT_TRUE(rf.find_root("entropy", 4., 0., 0., 0., linear_map, lo_T, hi_T, y));
    EXPECT_EQ(y[1], 0.);
    EXPECT_EQ(y[2], 0.);
    EXPECT_EQ(y[3], 0.);
    EXPECT_LE(linear_residual(y, 4., 0., 0., 0.), ROOTFINDER_TOLERANCE);
}

TEST(Rootfinder, a_root_outside_the_box_is_not_found){
    Rootfinder rf;
    std::vector<double> lo{0., -10., -10., -10.}, hi{10., 10., 10., 10.};
    for (double e : {100., -4.}){  // T = 50 above the box, T = -2 below it
        std::vector<double> x{0.5, 0.5, 0.5, 0.5};
        EXPECT_FALSE(rf.find_root("energy", e, 3., 10., -7., linear_map, lo, hi, x)) << e;
    }
}

TEST(RootfinderDeathTest, an_unknown_mode_stops_the_run){
    Rootfinder rf;
    std::vector<double> lo{0., -10., -10., -10.}, hi{10., 10., 10., 10.}, x{1., 1., 1., 1.};
    EXPECT_DEATH(rf.find_root("enthalpy", 2., 3., 10., -7., linear_map, lo, hi, x),
                 "invalid choice");
}

// ------------------------------------------- extension beyond a table (eos_extension)
// Not used in runs (use_nonconformal_extension is false in include/eos_header.h); kept for
// comparisons. It writes sqrt(p) as a quadratic in (T, muB, muQ, muS) whose 15
// coefficients match p, its 4 first and 10 second derivatives at one point. The conformal
// pressure p = c x^2, x = T^2 + muB^2 + muQ^2 + muS^2, has a quadratic sqrt(p), so matched
// anywhere the extension must give the conformal EoS everywhere. (eos.h includes eos_extension.h.)

TEST(EoSExtension, continues_a_conformal_pressure_exactly){
    auto eos = make_eos("conformal");
    std::vector<double> at{points[1].T, points[1].muB, points[1].muQ, points[1].muS};
    std::vector<double> matched = eos->get_thermodynamics(at, "conformal");
    const double point0[4] = {at[0], at[1], at[2], at[3]};
    eos_extension::set_coeffs(point0, matched.data());

    for (const Point& x : points){
        std::vector<double> there{x.T, x.muB, x.muQ, x.muS};
        const std::vector<double> conformal = eos->get_thermodynamics(there, "conformal");
        const double point[4] = {x.T, x.muB, x.muQ, x.muS};
        double extended[17];
        eos_extension::get_full_thermo(point, extended);
        for (int i = 0; i < 17; ++i)
            EXPECT_TRUE(close(extended[i], conformal[i])) << "field " << i << " at T = " << x.T;
    }
}

/// The same through the call the table would make: match at the table edge, evaluate at
/// the point outside.
TEST(EoSExtension, continuation_from_the_table_edge){
    auto eos = make_eos("conformal");
    const double outside[4] = {1.5, -0.7, 0.3, -0.2};
    const double edge[4] = {0.75, -0.35, 0.15, -0.1};
    std::vector<double> at_edge{edge[0], edge[1], edge[2], edge[3]};
    std::vector<double> results = eos->get_thermodynamics(at_edge, "conformal");
    eos_extension::get_nonconformal_extension(outside, edge, results.data());
    std::vector<double> there{outside[0], outside[1], outside[2], outside[3]};
    const std::vector<double> conformal = eos->get_thermodynamics(there, "conformal");
    for (int i : {0, 1, 2, 3, 4, 5, 6, 7, 16})
        EXPECT_TRUE(close(results[i], conformal[i])) << "field " << i;
}

/// A point outside is scaled towards the origin until it lies on the table edge: all four
/// coordinates are divided by the largest ratio to the limit on their side.
TEST(EoSExtension, projection_onto_the_table_edge){
    const double minima[4] = {0., -2., -1., -4.}, maxima[4] = {1., 2., 1., 4.};
    double point[4] = {2., 1., -0.5, -1.};   // T/T_max = 2 is the largest ratio
    eos_extension::project_to_boundary(point, minima, maxima);
    EXPECT_EQ(point[0], 1.);
    EXPECT_EQ(point[1], 0.5);
    EXPECT_EQ(point[2], -0.25);
    EXPECT_EQ(point[3], -0.5);
    double below[4] = {0.5, -6., 0., 0.};   // muB/muB_min = 3 on the negative side
    eos_extension::project_to_boundary(below, minima, maxima);
    EXPECT_EQ(below[0], 0.5/3.);
    EXPECT_EQ(below[1], -2.);
    // a frozen axis (limit 0) does not constrain
    const double frozen_min[4] = {0., -2., 0., 0.}, frozen_max[4] = {1., 2., 0., 0.};
    double reduced[4] = {0.5, 4., 0., 0.};
    eos_extension::project_to_boundary(reduced, frozen_min, frozen_max);
    EXPECT_EQ(reduced[0], 0.25);
    EXPECT_EQ(reduced[1], 2.);
}
