// Equations of motion in Milne coordinates (src/eom_default.cpp).
//
// The physics cases run in 3+1D and in 2+1D (typed tests EoMTest, CTest names
// ending in <In3D> and <In2D>). The metric is g = diag(1, -1, -1, -tau^2) with
// coordinates (tau, x, y, eta_s). One particle at proper time tau = 2, with its
// inputs set by hand but as a run would leave them (see set_conformal_state and
// set_sigma_lab). The cases of the original 2+1D test with hand-derived values,
// the regression snapshot and two 1+1D cases follow at the end.
#include <iomanip>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <Kokkos_Core.hpp>
//#include <gmock/gmock.h> //To mock settings pointers

#include "sph_workstation.h"
#include "system_state.h"
#include "settings.h"
#include "particle.h"
#include "eom_default.h"
#define TOL 5.E-13

// The dimensions, 3+1D first. CTest names end in <In3D> or <In2D>.
struct In3D { static constexpr unsigned int D = 3; };
struct In2D { static constexpr unsigned int D = 2; };
using Dimensions = ::testing::Types<In3D, In2D>;

namespace {

constexpr double TAU = 2.;

::testing::AssertionResult near(double value, double expected){
    if (std::fabs(value - expected) <= 1.e-12*(1. + std::fabs(expected)))
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << std::setprecision(17) << value
                                         << " instead of " << expected;
}

// A conformal thermodynamic state at zero chemical potential, as the equation
// of state gives it: w = e + p = T s, e = 3p, c_s^2 = 1/3, dw/ds = 4T/3.
template <unsigned int D>
void set_conformal_state(ccake::Particle<D>& p, double T, double s){
    p.thermo.T = T;
    p.thermo.s = s;
    p.thermo.w = T*s;
    p.thermo.p = T*s/4.;
    p.thermo.e = 3.*T*s/4.;
    p.thermo.cs2 = 1./3.;
    p.thermo.dwds = 4.*T/3.;
}

// The smoothing step of a run leaves sigma_lab and sigma = sigma_lab/(gamma t)
// together, gamma being the Lorentz factor of u. The tests choose sigma; this
// sets sigma_lab to match.
template <unsigned int D>
void set_sigma_lab(std::shared_ptr<ccake::SystemState<D>> systemPtr){
    const double t = systemPtr->t;
    for (auto& p : systemPtr->particles){
        double u[D];
        for (unsigned i = 0; i < D; ++i) u[i] = p.hydro.u(i);
        p.hydro.sigma_lab = p.hydro.sigma*ccake::EoM_default<D>::gamma_calc(u, t*t)*t;
    }
}

// One particle at (x, y, eta_s) = (-1.5, 2.5, 0.2) (eta_s only in 1+1D) at
// proper time t, at rest, with sigma = 2. tau_Pi must not be zero (it divides).
template <unsigned int D>
std::shared_ptr<ccake::SystemState<D>> one_particle(double t = TAU){
    auto settingsPtr = std::make_shared<Settings>();
    auto sysPtr = std::make_shared<ccake::SystemState<D>>(settingsPtr);
    sysPtr->particles.resize(1);
    ccake::Particle<D>& p = sysPtr->particles[0];
    const double position[3] = {-1.5, 2.5, 0.2};
    if (D == 1) p.r(0) = position[2];
    else for (unsigned i = 0; i < D; ++i) p.r(i) = position[i];
    p.ID = 0;
    p.hydro.sigma = 2.;
    p.hydro.tau_Pi = 1.;
    sysPtr->n_particles = 1;
    sysPtr->t = t;
    return sysPtr;
}

// Ideal fluid: a conformal state at T = pi/2.
template <unsigned int D>
std::shared_ptr<ccake::SystemState<D>> ideal_particle(double t = TAU){
    auto sysPtr = one_particle<D>(t);
    set_conformal_state(sysPtr->particles[0], M_PI/2., 7.);
    return sysPtr;
}

// Viscous fluid at rest: conformal at T = pi/2 with w = e + p = T s = 3.
template <unsigned int D>
std::shared_ptr<ccake::SystemState<D>> viscous_particle(){
    auto sysPtr = one_particle<D>();
    sysPtr->particles[0].hydro.tau_pi = 1.;
    set_conformal_state(sysPtr->particles[0], M_PI/2., 6./M_PI);
    return sysPtr;
}

std::shared_ptr<Settings> eom_settings(bool shear){
    auto settingsPtr = std::make_shared<Settings>();
    settingsPtr->using_shear = shear;
    settingsPtr->using_diffusion = false;
    return settingsPtr;
}

// gamma and v from u, the full shear tensor, then all time derivatives: the
// order used by the code in every time step.
template <unsigned int D>
void evaluate(std::shared_ptr<ccake::SystemState<D>> systemPtr, std::shared_ptr<Settings> settingsPtr){
    set_sigma_lab(systemPtr);
    systemPtr->allocate_cabana_particles();
    ccake::EoM_default<D>::update_velocity(systemPtr);
    ccake::EoM_default<D>::reset_pi_tensor(systemPtr);
    ccake::EoM_default<D>::evaluate_time_derivatives(systemPtr, settingsPtr);
    systemPtr->copy_device_to_host();
}

// Diagonal of the metric, (tau, x, y, eta_s).
double metric(int mu, double t){
    return mu == 0 ? 1. : (mu == 3 ? -t*t : -1.);
}

} // namespace

template <typename T> class EoMTest : public ::testing::Test {};
TYPED_TEST_SUITE(EoMTest, Dimensions);

/// gamma = u^tau = sqrt(1 + u^x^2 + u^y^2 + tau^2 u^eta^2) and v^i = u^i/gamma.
/// With u = (3, 4, 1/2) at tau = 2: gamma = sqrt(27) in 3+1D, sqrt(26) in 2+1D.
TYPED_TEST(EoMTest, gamma_and_velocity_from_u){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = one_particle<D>();
    const double u[3] = {3., 4., 0.5};
    for (unsigned i = 0; i < D; ++i) systemPtr->particles[0].hydro.u(i) = u[i];
    const double gamma = (D == 3) ? std::sqrt(27.) : std::sqrt(26.);
    systemPtr->allocate_cabana_particles();
    ccake::EoM_default<D>::update_velocity(systemPtr);
    systemPtr->copy_device_to_host();
    const auto& q = systemPtr->particles[0];
    EXPECT_TRUE(near(q.hydro.gamma, gamma));
    for (unsigned i = 0; i < D; ++i) EXPECT_TRUE(near(q.hydro.v(i), u[i]/gamma)) << "v^" << i;
}

/// The shear tensor is evolved through its spatial components except pi^{eta eta};
/// the others follow from transversality, pi^{mu nu} u_nu = 0, and tracelessness,
/// g_{mu nu} pi^{mu nu} = 0. Checked for a moving fluid with u^eta != 0 in 3+1D.
TYPED_TEST(EoMTest, shear_tensor_is_completed_transverse_and_traceless){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = one_particle<D>();
    auto& p = systemPtr->particles[0];
    const double u[3] = {3., 4., 0.5};
    for (unsigned i = 0; i < D; ++i) p.hydro.u(i) = u[i];
    // stored per unit of sigma = 2: pi^{xx} = 3, pi^{xy} = 4, pi^{yy} = 5,
    // and in 3+1D pi^{x eta} = 0.6, pi^{y eta} = -0.4; Pi = 6
    p.hydro.extensive_shv(0,0) = 3./2.;
    p.hydro.extensive_shv(0,1) = 4./2.;
    p.hydro.extensive_shv(1,1) = 5./2.;
    if (D == 3){
        p.hydro.extensive_shv(0,2) = 0.6/2.;
        p.hydro.extensive_shv(1,2) = -0.4/2.;
    }
    p.hydro.extensive_bulk = 6./2.;
    systemPtr->allocate_cabana_particles();
    ccake::EoM_default<D>::update_velocity(systemPtr);
    ccake::EoM_default<D>::reset_pi_tensor(systemPtr);
    systemPtr->copy_device_to_host();
    const auto& q = systemPtr->particles[0];

    EXPECT_TRUE(near(q.hydro.shv(1,1), 3.));
    EXPECT_TRUE(near(q.hydro.shv(1,2), 4.));
    EXPECT_TRUE(near(q.hydro.shv(2,2), 5.));
    EXPECT_TRUE(near(q.hydro.shv(1,3), D == 3 ? 0.6 : 0.));
    EXPECT_TRUE(near(q.hydro.shv(2,3), D == 3 ? -0.4 : 0.));
    EXPECT_TRUE(near(q.hydro.bulk, 6.));

    double u_up[4] = {q.hydro.gamma, 0., 0., 0.};
    for (unsigned i = 0; i < D; ++i) u_up[i+1] = u[i];
    double trace = 0.;
    for (int mu = 0; mu < 4; ++mu){
        double transverse = 0.;
        for (int nu = 0; nu < 4; ++nu){
            EXPECT_TRUE(near(q.hydro.shv(mu,nu), q.hydro.shv(nu,mu))) << "symmetry " << mu << nu;
            transverse += q.hydro.shv(mu,nu)*metric(nu, TAU)*u_up[nu];
        }
        EXPECT_TRUE(near(transverse, 0.)) << "pi^{" << mu << " nu} u_nu";
        trace += metric(mu, TAU)*q.hydro.shv(mu,mu);
    }
    EXPECT_TRUE(near(trace, 0.)) << "trace";
}

/// A fluid at rest expands at theta = d_i v^i + 1/tau: the trace of the velocity
/// gradient plus the 1/tau of the longitudinal (Bjorken) expansion. Off-diagonal
/// gradients do not enter. Nothing accelerates it.
TYPED_TEST(EoMTest, expansion_rate_at_rest){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    auto& p = systemPtr->particles[0];
    const double diagonal[3] = {1., 2., 0.5};
    for (unsigned i = 0; i < D; ++i)
        for (unsigned j = 0; j < D; ++j) p.hydro.gradV(i,j) = (i == j) ? diagonal[i] : 4.5 + i - j;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    double theta = 1./TAU;
    for (unsigned i = 0; i < D; ++i) theta += diagonal[i];
    EXPECT_TRUE(near(q.hydro.theta, theta));
    for (unsigned i = 0; i < D; ++i) EXPECT_EQ(q.hydro.du_dt(i), 0.) << "du^" << i << "/dt";
}

/// An ideal fluid at rest without gradients is the Bjorken solution: no
/// acceleration, no change of entropy, charges or bulk pressure, theta = 1/tau.
TYPED_TEST(EoMTest, ideal_fluid_at_rest_is_stationary){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    for (unsigned i = 0; i < D; ++i) EXPECT_EQ(q.hydro.du_dt(i), 0.) << "du^" << i << "/dt";
    EXPECT_EQ(q.d_dt_extensive.s, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoB, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoS, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoQ, 0.);
    EXPECT_EQ(q.hydro.d_extensive_bulk_dt, 0.);
    EXPECT_TRUE(near(q.hydro.theta, 1./TAU));
}

/// Euler equation for an ideal fluid at rest, (e + p) du^i/dt = g^{ij} d_j p:
/// du^x/dt = -d_x p/w, du^y/dt = -d_y p/w, du^eta/dt = -d_eta p/(tau^2 w).
TYPED_TEST(EoMTest, pressure_gradient_accelerates_ideal_fluid){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    auto& p = systemPtr->particles[0];
    const double gradP[3] = {7./4., 9./4., 9./8.};
    for (unsigned i = 0; i < D; ++i) p.hydro.gradP(i) = gradP[i];
    const double w = p.thermo.w;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    for (unsigned i = 0; i < D; ++i){
        const double g_ii = (D == 3 && i == 2) ? -TAU*TAU : -1.;
        EXPECT_TRUE(near(q.hydro.du_dt(i), gradP[i]/(g_ii*w))) << "du^" << i << "/dt";
    }
    EXPECT_EQ(q.d_dt_extensive.s, 0.);   // acceleration produces no entropy
}

/// Without viscosity, diffusion or sources, entropy and the three charges carried
/// by each SPH particle are conserved, also while the fluid moves (with u^eta in
/// 3+1D), accelerates and expands.
TYPED_TEST(EoMTest, ideal_flow_conserves_entropy_and_charges){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    auto& p = systemPtr->particles[0];
    const double u[3] = {3., 4., 0.5}, gradP[3] = {7./4., 9./4., 9./8.};
    for (unsigned i = 0; i < D; ++i){
        p.hydro.u(i) = u[i];
        p.hydro.gradP(i) = gradP[i];
        for (unsigned j = 0; j < D; ++j) p.hydro.gradV(i,j) = (i == j) ? 1. + i : 4.5 + i - j;
    }
    // Charges, chemical potentials and dw/drho: arbitrary values (w is not
    // updated for them), so that the charge terms of the equations are nonzero.
    // Conservation must hold for any state.
    p.thermo.rhoB = 11.; p.thermo.rhoQ = 12.; p.thermo.rhoS = 13.;
    p.thermo.muB = 0.1;  p.thermo.muS = 0.2;  p.thermo.muQ = 0.3;
    p.thermo.dwdB = 8.;  p.thermo.dwdQ = 9.;  p.thermo.dwdS = 10.;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    for (unsigned i = 0; i < D; ++i) ASSERT_TRUE(std::isfinite(q.hydro.du_dt(i)));
    EXPECT_NE(q.hydro.du_dt(0), 0.);   // the state really is accelerating
    EXPECT_EQ(q.d_dt_extensive.s, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoB, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoS, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoQ, 0.);
    EXPECT_EQ(q.hydro.d_extensive_bulk_dt, 0.);
}

/// Bulk pressure at rest, shear off. The Israel-Stewart equation
///   tau_Pi dPi/dt + Pi = -zeta theta - delta_PiPi Pi theta + phi_1 Pi^2,
/// with dsigma/dt = -sigma theta for the reference density, gives the rate of
/// the evolved variable Pi/sigma:  d(Pi/sigma)/dt = (dPi/dt + Pi theta)/sigma.
/// Entropy is produced at the rate -Pi theta/T:  d(s/sigma)/dt = -Pi theta/(sigma T).
TYPED_TEST(EoMTest, bulk_relaxation_at_rest_follows_israel_stewart){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = viscous_particle<D>();
    auto& p = systemPtr->particles[0];
    const double Pi = 0.3, zeta = 0.4, tau_Pi = 1.5, delta_PiPi = 0.7, phi1 = 0.2;
    p.hydro.extensive_bulk = Pi/p.hydro.sigma;
    p.hydro.zeta_Pi = zeta;
    p.hydro.tau_Pi = tau_Pi;
    p.hydro.delta_PiPi = delta_PiPi;
    p.hydro.phi1 = phi1;
    const double diagonal[3] = {1., 2., 0.5};
    double theta = 1./TAU;
    for (unsigned i = 0; i < D; ++i){ p.hydro.gradV(i,i) = diagonal[i]; theta += diagonal[i]; }
    const double sigma = p.hydro.sigma, T = p.thermo.T;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    const double dPi_dt = (-Pi - zeta*theta - delta_PiPi*Pi*theta + phi1*Pi*Pi)/tau_Pi;
    EXPECT_TRUE(near(q.hydro.theta, theta));
    EXPECT_TRUE(near(q.hydro.d_extensive_bulk_dt, (dPi_dt + Pi*theta)/sigma));
    EXPECT_TRUE(near(q.d_dt_extensive.s, -Pi*theta/(sigma*T)));
    for (unsigned i = 0; i < D; ++i) EXPECT_EQ(q.hydro.du_dt(i), 0.);
}

/// With a bulk pressure the effective pressure is p + Pi (Euler equation at rest):
///   (e + p + Pi) du^i/dt = g^{ij} d_j (p + Pi).
TYPED_TEST(EoMTest, bulk_pressure_gradient_accelerates){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = viscous_particle<D>();
    auto& p = systemPtr->particles[0];
    const double Pi = -0.4;
    const double gradP[3] = {7./4., 9./4., 9./8.}, gradBulk[3] = {0.3, -0.2, 0.1};
    p.hydro.extensive_bulk = Pi/p.hydro.sigma;
    for (unsigned i = 0; i < D; ++i){ p.hydro.gradP(i) = gradP[i]; p.hydro.gradBulk(i) = gradBulk[i]; }
    const double w = p.thermo.w;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    for (unsigned i = 0; i < D; ++i){
        const double g_ii = (D == 3 && i == 2) ? -TAU*TAU : -1.;
        EXPECT_TRUE(near(q.hydro.du_dt(i), (gradP[i] + gradBulk[i])/(g_ii*(w + Pi)))) << "du^" << i << "/dt";
    }
}

/// Bjorken flow, no shear stress yet: the shear tensor starts to relax towards
/// its Navier-Stokes value 2 eta sigma^{mu nu}. For a fluid at rest that only
/// expands longitudinally sigma^{xx} = sigma^{yy} = 1/(3 tau) (transverse
/// pressure up, longitudinal down) and the other evolved components vanish, so
///   d(pi^{xx}/sigma)/dt = d(pi^{yy}/sigma)/dt = 2 eta/(3 tau tau_pi sigma).
TYPED_TEST(EoMTest, bjorken_shear_starts_towards_navier_stokes){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = viscous_particle<D>();
    auto& p = systemPtr->particles[0];
    const double eta = 0.6, tau_pi = 1.2;
    p.hydro.eta_pi = eta;
    p.hydro.tau_pi = tau_pi;
    const double sigma = p.hydro.sigma;
    evaluate(systemPtr, eom_settings(true));
    const auto& q = systemPtr->particles[0];
    const double expected = 2.*eta/(3.*TAU*tau_pi*sigma);
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,0), expected));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(1,1), expected));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,1), 0.));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,2), 0.));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(1,2), 0.));
}

/// The Navier-Stokes source with velocity gradients d_i v^j = gradV(i,j), at rest:
///   sigma^{mu nu} = (nabla^mu u^nu + nabla^nu u^mu)/2 - Delta^{mu nu} theta/3,
///   sigma^{xx} = -d_x v^x + theta/3,  sigma^{yy} = -d_y v^y + theta/3,
///   sigma^{xy} = -(d_x v^y + d_y v^x)/2,
///   sigma^{x eta} = -(d_x v^eta + d_eta v^x/tau^2)/2,  likewise y eta,
///   theta = d_x v^x + d_y v^y + d_eta v^eta + 1/tau,
/// and d(pi^{ij}/sigma)/dt = 2 eta sigma^{ij}/(tau_pi sigma) while pi = 0.
TYPED_TEST(EoMTest, shear_source_with_velocity_gradients){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = viscous_particle<D>();
    auto& p = systemPtr->particles[0];
    const double eta = 0.6, tau_pi = 1.2;
    // d_i v^j, i = derivative (x, y, eta_s), j = component
    const double dv[3][3] = {{0.3, 0.2, 0.07}, {0.05, -0.1, -0.04}, {0.4, -0.3, 0.15}};
    p.hydro.eta_pi = eta;
    p.hydro.tau_pi = tau_pi;
    for (unsigned i = 0; i < D; ++i)
        for (unsigned j = 0; j < D; ++j) p.hydro.gradV(i,j) = dv[i][j];
    const double sigma = p.hydro.sigma;
    evaluate(systemPtr, eom_settings(true));
    const auto& q = systemPtr->particles[0];

    double theta = dv[0][0] + dv[1][1] + 1./TAU;
    if (D == 3) theta += dv[2][2];
    const double factor = 2.*eta/(tau_pi*sigma);
    EXPECT_TRUE(near(q.hydro.theta, theta));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,0), factor*(-dv[0][0] + theta/3.)));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(1,1), factor*(-dv[1][1] + theta/3.)));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,1), factor*(-(dv[0][1] + dv[1][0])/2.)));
    if (D == 3){
        EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,2),
                         factor*(-(dv[0][2] + dv[2][0]/(TAU*TAU))/2.))) << "x eta";
        EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(1,2),
                         factor*(-(dv[1][2] + dv[2][1]/(TAU*TAU))/2.))) << "y eta";
    }
}

/// Entropy production by shear in Bjorken flow. With pi^{xx} = pi^{yy} = a
/// (hence tau^2 pi^{eta eta} = -2a), pi^{mu nu} sigma_{mu nu} = 2a/tau, and
///   d(s/sigma)/dt = pi^{mu nu} sigma_{mu nu}/(sigma T) = 2a/(tau sigma T) > 0.
TYPED_TEST(EoMTest, bjorken_shear_produces_entropy){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = viscous_particle<D>();
    auto& p = systemPtr->particles[0];
    const double a = 0.05;
    p.hydro.extensive_shv(0,0) = a/p.hydro.sigma;
    p.hydro.extensive_shv(1,1) = a/p.hydro.sigma;
    p.hydro.eta_pi = 0.6;
    p.hydro.tau_pi = 1.2;
    const double sigma = p.hydro.sigma, T = p.thermo.T;
    evaluate(systemPtr, eom_settings(true));
    const auto& q = systemPtr->particles[0];
    EXPECT_TRUE(near(q.hydro.shv(3,3), -2.*a/(TAU*TAU)));
    EXPECT_TRUE(near(q.hydro.shv_nabla_u, 2.*a/TAU));
    EXPECT_TRUE(near(q.d_dt_extensive.s, 2.*a/(TAU*sigma*T)));
}

//==============================================================================
// 1+1D: Bjorken flow and flow along eta_s
//==============================================================================

/// A fluid at rest in 1+1D (eta_s only) is the Bjorken solution: it stays at
/// rest, expands at 1/tau and keeps its entropy.
TEST(eom_default, bjorken_at_rest_1d){
    auto systemPtr = ideal_particle<1>();
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    EXPECT_EQ(q.hydro.du_dt(0), 0.);
    EXPECT_TRUE(near(q.hydro.theta, 1./TAU));
    EXPECT_EQ(q.d_dt_extensive.s, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoB, 0.);
}

/// A fluid moving along eta_s without dissipation keeps its entropy (1+1D).
TEST(eom_default, ideal_flow_along_eta_conserves_entropy_1d){
    auto systemPtr = ideal_particle<1>();
    systemPtr->particles[0].hydro.u(0) = 0.3;          // u^eta in fm^-1
    systemPtr->particles[0].hydro.gradP(0) = 0.5;
    systemPtr->particles[0].hydro.gradV(0,0) = 0.2;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    ASSERT_TRUE(std::isfinite(q.hydro.du_dt(0)));
    EXPECT_NE(q.hydro.du_dt(0), 0.);
    EXPECT_EQ(q.d_dt_extensive.s, 0.);
}

//==============================================================================
// The original 2+1D cases, with their hand-derived values
//==============================================================================

std::shared_ptr<ccake::SystemState<2>> prepare_system(){
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings> ();
    settingsPtr->xmin = -4;
    settingsPtr->ymin = -6;
    settingsPtr->etamin = -2;
    settingsPtr->hT = 1;
    settingsPtr->using_shear = true;

    std::shared_ptr<ccake::SystemState<2>> sysPtr = std::make_shared<ccake::SystemState<2>> (settingsPtr);
    // (the smoothing length is read from the settings; SystemState has no hT of its own)
    const int n_sph_particles = 1;
    std::array<ccake::Particle<2>,n_sph_particles> particles_array_in;
    particles_array_in[0].r = {-1.5,2.5};
    for (auto p : particles_array_in) sysPtr->add_particle(p);
    sysPtr->n_particles = n_sph_particles;
    sysPtr->t = 2.;

    return sysPtr;
}

TEST(eom_default,get_LRF){

    ASSERT_DOUBLE_EQ(ccake::EoM_default<2>::get_LRF(4.,2.,2.),1.);
}

TEST(eom_default, gamma_and_v){
    std::shared_ptr<ccake::SystemState<2>> systemPtr = prepare_system();
    //Setup test
    systemPtr->particles[0].ID = 0;
    systemPtr->particles[0].hydro.sigma = 2.;
    systemPtr->particles[0].hydro.u(0) = 3.;
    systemPtr->particles[0].hydro.u(1) = 4.;
    double expected_gamma = sqrt(26.0);
    double expected_vx = 3./sqrt(26.0);
    double expected_vy = 4./sqrt(26.0);

    systemPtr->particles[0].hydro.sigma = 2.;
    systemPtr->particles[0].hydro.gradV(0,0) = 1.;
    systemPtr->particles[0].hydro.gradV(1,1) = 2.;
    systemPtr->particles[0].hydro.gradV(0,1) = 4.5; //Should not matter
    systemPtr->particles[0].hydro.gradV(1,0) = 4.5; //Should not matter

    systemPtr->allocate_cabana_particles();
    // gamma and v are now computed by update_velocity, no longer by reset_pi_tensor
    ccake::EoM_default<2>::update_velocity(systemPtr);
    ccake::EoM_default<2>::reset_pi_tensor(systemPtr);
    systemPtr->copy_device_to_host();
    ASSERT_DOUBLE_EQ(systemPtr->particles[0].hydro.gamma, expected_gamma);
    ASSERT_DOUBLE_EQ(systemPtr->particles[0].hydro.v(0), expected_vx);
    ASSERT_DOUBLE_EQ(systemPtr->particles[0].hydro.v(1), expected_vy);
}

/// The shear tensor is evolved through its independent spatial components;
/// the others follow from transversality (pi^{0i} = -pi^{ij} u_j / gamma) and
/// tracelessness. The expected values are the hand-derived ones of the original
/// version of this test (spatial components 3, 4, 5, u = (3, 4), t = 2).
TEST(eom_default, complete_shear_tensor){
    std::shared_ptr<ccake::SystemState<2>> systemPtr = prepare_system();
    ccake::Particle<2>& p = systemPtr->particles[0];
    p.ID = 0;
    p.hydro.sigma = 2.;
    p.hydro.u(0) = 3.;
    p.hydro.u(1) = 4.;
    // stored per unit of sigma: pi^{xx} = 3, pi^{xy} = 4, pi^{yy} = 5, Pi = 6
    p.hydro.extensive_shv(0,0) = 3./2.;
    p.hydro.extensive_shv(0,1) = 4./2.;
    p.hydro.extensive_shv(1,1) = 5./2.;
    p.hydro.extensive_bulk = 6./2.;
    const double gamma = sqrt(26.0);

    systemPtr->allocate_cabana_particles();
    ccake::EoM_default<2>::update_velocity(systemPtr);
    ccake::EoM_default<2>::reset_pi_tensor(systemPtr);
    systemPtr->copy_device_to_host();
    const ccake::Particle<2>& q = systemPtr->particles[0];

    EXPECT_DOUBLE_EQ(q.hydro.shv(1,1), 3.);
    EXPECT_DOUBLE_EQ(q.hydro.shv(1,2), 4.);
    EXPECT_DOUBLE_EQ(q.hydro.shv(2,2), 5.);
    EXPECT_DOUBLE_EQ(q.hydro.shv(2,1), q.hydro.shv(1,2));
    EXPECT_DOUBLE_EQ(q.hydro.shv(0,1), 25./gamma);
    EXPECT_DOUBLE_EQ(q.hydro.shv(1,0), 25./gamma);
    EXPECT_DOUBLE_EQ(q.hydro.shv(0,2), 32./gamma);
    EXPECT_DOUBLE_EQ(q.hydro.shv(2,0), 32./gamma);
    EXPECT_DOUBLE_EQ(q.hydro.shv(0,0), 203./26.);
    EXPECT_LE(std::fabs(q.hydro.shv(3,3) - (-5./104.)), TOL);
    EXPECT_DOUBLE_EQ(q.hydro.bulk, 6.);
}

// The input state of the original full time-derivative test, with shear and
// bulk viscosity. Its thermodynamics is not that of an equation of state; it
// is kept as it was because it is the input of the snapshot below.
std::shared_ptr<Settings> make_viscous(ccake::Particle<2>& p){
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings>();
    settingsPtr->using_shear = true;
    settingsPtr->using_diffusion = false;
    p.ID = 0;
    p.hydro.sigma = 2.;                           // sigma_lab follows (evaluate)
    p.hydro.u = {3., 4.};
    p.hydro.extensive_bulk = 6./2.;               // Pi = 6
    p.hydro.extensive_shv(0,0) = 3./2.;           // pi^{xx} = 3
    p.hydro.extensive_shv(0,1) = 4./2.;           // pi^{xy} = 4
    p.hydro.extensive_shv(1,1) = 5./2.;           // pi^{yy} = 5
    p.hydro.eta_pi = 2.*M_PI;
    p.hydro.tau_pi = 3.*M_PI;
    p.hydro.zeta_Pi = 4.*M_PI;
    p.hydro.tau_Pi = 2.*M_PI;
    p.thermo.s = 7.;
    p.thermo.T = M_PI/2.;
    p.thermo.w = M_PI/2.;
    p.thermo.dwds = M_PI;
    p.thermo.dwdB = 8.;
    p.thermo.dwdQ = 9.;
    p.thermo.dwdS = 10.;
    p.thermo.rhoB = 11.;
    p.thermo.rhoQ = 12.;
    p.thermo.rhoS = 13.;
    p.hydro.gradshear = {7./2., 9./2.};
    p.hydro.gradBulk = {7./5., 9./5.};
    p.hydro.gradP = {7./4., 9./4.};
    p.hydro.divshear = {7./8., 9./8.};
    p.hydro.gradV(0,0) = 1.;
    p.hydro.gradV(1,1) = 2.;
    p.hydro.gradV(0,1) = 4.5;
    p.hydro.gradV(1,0) = 5.5;
    return settingsPtr;
}

/// REGRESSION SNAPSHOT, not an independent check. The original version of this
/// file compared the full viscous time derivatives of this state with values
/// derived by hand (unit_tests/UnitTestsCalculations.nb). Those values belong to
/// the equations of motion before they were rewritten in the current M/R/F form
/// and no longer apply. The numbers below are what the code computes at commit
/// 2a359e0e (GCC 13, -O3; an -O0 build agrees to 1e-13 relative). They detect
/// any change in the viscous equations of motion; they do not show that the
/// equations are right.
/// If you change the equations on purpose, update the numbers and say why in
/// the pull request. Replacing them by derived values is still to be done.
TEST(eom_default, viscous_time_derivatives_snapshot){
    std::shared_ptr<ccake::SystemState<2>> systemPtr = prepare_system();
    std::shared_ptr<Settings> settingsPtr = make_viscous(systemPtr->particles[0]);
    evaluate(systemPtr, settingsPtr);
    const ccake::Particle<2>& q = systemPtr->particles[0];

    // On a mismatch the new value is printed with all its digits, ready to be
    // pasted here if the change of the equations is intended.
    auto expect_close = [](double value, double expected, const char* what){
        EXPECT_LE(std::fabs(value/expected - 1.), TOL)
            << std::setprecision(17) << what << " is " << value
            << ", the snapshot has " << expected;
    };
    expect_close(q.hydro.du_dt(0), -11.048954486938108, "du^x/dt");
    expect_close(q.hydro.du_dt(1), -15.143042657788019, "du^y/dt");
    expect_close(q.d_dt_extensive.s, -0.20952074652782482, "d(extensive s)/dt");
    expect_close(q.hydro.d_extensive_bulk_dt, -0.30279502559476157, "d(extensive Pi)/dt");
    expect_close(q.hydro.d_extensive_shv_dt(0,0), -31.802968675732981, "d(extensive pi^xx)/dt");
    expect_close(q.hydro.d_extensive_shv_dt(0,1), -40.543948333105121, "d(extensive pi^xy)/dt");
    expect_close(q.hydro.d_extensive_shv_dt(1,1), -45.251870870696308, "d(extensive pi^yy)/dt");
    expect_close(q.hydro.theta, -0.53324645742541232, "theta");
    expect_close(q.hydro.shv_nabla_u, -6.5558004342629035, "pi^{ij} nabla_i u_j");
    // exact: no eta components in 2+1D, and no diffusion to move charge
    EXPECT_EQ(q.hydro.d_extensive_shv_dt(0,2), 0.);
    EXPECT_EQ(q.hydro.d_extensive_shv_dt(1,2), 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoB, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoS, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoQ, 0.);
}

// Define the main function
int main(int argc, char* argv[]) {

    // Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);

    // Run the tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
