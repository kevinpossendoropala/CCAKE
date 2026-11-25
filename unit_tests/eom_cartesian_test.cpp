// Equations of motion in Cartesian coordinates (src/eom_cartesian.cpp,
// coordinate_system: cartesian).
//
// The cases of eom_default_test.cpp, in the Cartesian metric g = diag(1, -1, -1, -1) with
// coordinates (t, x, y, z): no longitudinal expansion, so a fluid at rest has theta = 0
// and nothing changes unless a gradient drives it. 3+1D and 2+1D (typed tests, CTest names
// ending in <In3D> and <In2D>). One particle at lab time t = 2, its inputs set by hand but
// as a run would leave them.
//
// Two terms of the Cartesian equations are known to differ from the Milne ones (the sign of
// lambda_Pipi in the bulk equation and of the charge-diffusion term); no case here turns
// either on.
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <iomanip>
#include <memory>

#include "eom_cartesian.h"
#include "particle.h"
#include "settings.h"
#include "system_state.h"

struct In3D { static constexpr unsigned int D = 3; };
struct In2D { static constexpr unsigned int D = 2; };
using Dimensions = ::testing::Types<In3D, In2D>;

namespace {

constexpr double T_LAB = 2.;

::testing::AssertionResult near(double value, double expected){
    if (std::fabs(value - expected) <= 1.e-12*(1. + std::fabs(expected)))
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << std::setprecision(17) << value
                                         << " instead of " << expected;
}

// A conformal state at zero chemical potential, as the EoS gives it.
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

// sigma_lab = sigma gamma, the smoothing step's relation in Cartesian coordinates.
template <unsigned int D>
void set_sigma_lab(std::shared_ptr<ccake::SystemState<D>> systemPtr){
    const double t = systemPtr->t;
    for (auto& p : systemPtr->particles){
        double u[D];
        for (unsigned i = 0; i < D; ++i) u[i] = p.hydro.u(i);
        p.hydro.sigma_lab = p.hydro.sigma*ccake::EoM_cartesian<D>::gamma_calc(u, t*t);
    }
}

template <unsigned int D>
std::shared_ptr<ccake::SystemState<D>> one_particle(){
    auto settingsPtr = std::make_shared<Settings>();
    auto sysPtr = std::make_shared<ccake::SystemState<D>>(settingsPtr);
    sysPtr->particles.resize(1);
    ccake::Particle<D>& p = sysPtr->particles[0];
    const double position[3] = {-1.5, 2.5, 0.2};
    for (unsigned i = 0; i < D; ++i) p.r(i) = position[i];
    p.ID = 0;
    p.hydro.sigma = 2.;
    p.hydro.tau_Pi = 1.;
    sysPtr->n_particles = 1;
    sysPtr->t = T_LAB;
    return sysPtr;
}

template <unsigned int D>
std::shared_ptr<ccake::SystemState<D>> ideal_particle(){
    auto sysPtr = one_particle<D>();
    set_conformal_state(sysPtr->particles[0], M_PI/2., 7.);
    return sysPtr;
}

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
    settingsPtr->coordinate_system = "cartesian";
    return settingsPtr;
}

template <unsigned int D>
void evaluate(std::shared_ptr<ccake::SystemState<D>> systemPtr, std::shared_ptr<Settings> settingsPtr){
    set_sigma_lab(systemPtr);
    systemPtr->allocate_cabana_particles();
    ccake::EoM_cartesian<D>::update_velocity(systemPtr);
    ccake::EoM_cartesian<D>::reset_pi_tensor(systemPtr);
    ccake::EoM_cartesian<D>::evaluate_time_derivatives(systemPtr, settingsPtr);
    systemPtr->copy_device_to_host();
}

double metric(int mu){ return mu == 0 ? 1. : -1.; }

} // namespace

template <typename T> class CartesianEoMTest : public ::testing::Test {};
TYPED_TEST_SUITE(CartesianEoMTest, Dimensions);

/// gamma = u^t = sqrt(1 + |u|^2), v^i = u^i/gamma. With u = (3, 4, 1/2): sqrt(26.25) in
/// 3+1D, sqrt(26) in 2+1D.
TYPED_TEST(CartesianEoMTest, gamma_and_velocity_from_u){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = one_particle<D>();
    const double u[3] = {3., 4., 0.5};
    for (unsigned i = 0; i < D; ++i) systemPtr->particles[0].hydro.u(i) = u[i];
    const double gamma = (D == 3) ? std::sqrt(26.25) : std::sqrt(26.);
    systemPtr->allocate_cabana_particles();
    ccake::EoM_cartesian<D>::update_velocity(systemPtr);
    systemPtr->copy_device_to_host();
    const auto& q = systemPtr->particles[0];
    EXPECT_TRUE(near(q.hydro.gamma, gamma));
    for (unsigned i = 0; i < D; ++i) EXPECT_TRUE(near(q.hydro.v(i), u[i]/gamma)) << "v^" << i;
    double ua[D];
    for (unsigned i = 0; i < D; ++i) ua[i] = u[i];
    EXPECT_TRUE(near(ccake::EoM_cartesian<D>::gamma_calc(ua, T_LAB*T_LAB), gamma));
}

/// The shear tensor is completed from its evolved components by transversality,
/// pi^{mu nu} u_nu = 0, and tracelessness, g_{mu nu} pi^{mu nu} = 0.
TYPED_TEST(CartesianEoMTest, shear_tensor_is_completed_transverse_and_traceless){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = one_particle<D>();
    auto& p = systemPtr->particles[0];
    const double u[3] = {3., 4., 0.5};
    for (unsigned i = 0; i < D; ++i) p.hydro.u(i) = u[i];
    p.hydro.extensive_shv(0,0) = 3./2.;
    p.hydro.extensive_shv(0,1) = 4./2.;
    p.hydro.extensive_shv(1,1) = 5./2.;
    if (D == 3){
        p.hydro.extensive_shv(0,2) = 0.6/2.;
        p.hydro.extensive_shv(1,2) = -0.4/2.;
    }
    p.hydro.extensive_bulk = 6./2.;
    systemPtr->allocate_cabana_particles();
    ccake::EoM_cartesian<D>::update_velocity(systemPtr);
    ccake::EoM_cartesian<D>::reset_pi_tensor(systemPtr);
    systemPtr->copy_device_to_host();
    const auto& q = systemPtr->particles[0];

    EXPECT_TRUE(near(q.hydro.shv(1,1), 3.));
    EXPECT_TRUE(near(q.hydro.shv(1,2), 4.));
    EXPECT_TRUE(near(q.hydro.shv(2,2), 5.));
    EXPECT_TRUE(near(q.hydro.bulk, 6.));

    double u_up[4] = {q.hydro.gamma, 0., 0., 0.};
    for (unsigned i = 0; i < D; ++i) u_up[i+1] = u[i];
    double trace = 0.;
    for (int mu = 0; mu < 4; ++mu){
        double transverse = 0.;
        for (int nu = 0; nu < 4; ++nu){
            EXPECT_TRUE(near(q.hydro.shv(mu,nu), q.hydro.shv(nu,mu))) << "symmetry " << mu << nu;
            transverse += q.hydro.shv(mu,nu)*metric(nu)*u_up[nu];
        }
        EXPECT_TRUE(near(transverse, 0.)) << "pi^{" << mu << " nu} u_nu";
        trace += metric(mu)*q.hydro.shv(mu,mu);
    }
    EXPECT_TRUE(near(trace, 0.)) << "trace";
}

/// At rest the expansion rate is the trace of the velocity gradient alone.
TYPED_TEST(CartesianEoMTest, expansion_rate_at_rest){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    auto& p = systemPtr->particles[0];
    const double diagonal[3] = {1., 2., 0.5};
    for (unsigned i = 0; i < D; ++i)
        for (unsigned j = 0; j < D; ++j) p.hydro.gradV(i,j) = (i == j) ? diagonal[i] : 4.5 + i - j;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    double theta = 0.;
    for (unsigned i = 0; i < D; ++i) theta += diagonal[i];
    EXPECT_TRUE(near(q.hydro.theta, theta));
    for (unsigned i = 0; i < D; ++i) EXPECT_EQ(q.hydro.du_dt(i), 0.) << "du^" << i << "/dt";
}

/// A uniform ideal fluid at rest stays as it is: in Cartesian coordinates there is no
/// expansion to drive it.
TYPED_TEST(CartesianEoMTest, uniform_fluid_at_rest_is_static){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    for (unsigned i = 0; i < D; ++i) EXPECT_EQ(q.hydro.du_dt(i), 0.) << "du^" << i << "/dt";
    EXPECT_EQ(q.d_dt_extensive.s, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoB, 0.);
    EXPECT_EQ(q.hydro.d_extensive_bulk_dt, 0.);
    EXPECT_EQ(q.hydro.theta, 0.);
}

/// Euler equation at rest: (e + p) du^i/dt = -d_i p.
TYPED_TEST(CartesianEoMTest, pressure_gradient_accelerates_ideal_fluid){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    auto& p = systemPtr->particles[0];
    const double gradP[3] = {7./4., 9./4., 9./8.};
    for (unsigned i = 0; i < D; ++i) p.hydro.gradP(i) = gradP[i];
    const double w = p.thermo.w;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    for (unsigned i = 0; i < D; ++i)
        EXPECT_TRUE(near(q.hydro.du_dt(i), -gradP[i]/w)) << "du^" << i << "/dt";
    EXPECT_EQ(q.d_dt_extensive.s, 0.);
}

/// Without viscosity, diffusion or sources, entropy and charges per particle are conserved
/// while the fluid moves, accelerates and expands.
TYPED_TEST(CartesianEoMTest, ideal_flow_conserves_entropy_and_charges){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = ideal_particle<D>();
    auto& p = systemPtr->particles[0];
    const double u[3] = {3., 4., 0.5}, gradP[3] = {7./4., 9./4., 9./8.};
    for (unsigned i = 0; i < D; ++i){
        p.hydro.u(i) = u[i];
        p.hydro.gradP(i) = gradP[i];
        for (unsigned j = 0; j < D; ++j) p.hydro.gradV(i,j) = (i == j) ? 1. + i : 4.5 + i - j;
    }
    // arbitrary charge state (w is not updated for it): conservation must hold for any
    p.thermo.rhoB = 11.; p.thermo.rhoQ = 12.; p.thermo.rhoS = 13.;
    p.thermo.muB = 0.1;  p.thermo.muS = 0.2;  p.thermo.muQ = 0.3;
    p.thermo.dwdB = 8.;  p.thermo.dwdQ = 9.;  p.thermo.dwdS = 10.;
    evaluate(systemPtr, eom_settings(false));
    const auto& q = systemPtr->particles[0];
    for (unsigned i = 0; i < D; ++i) ASSERT_TRUE(std::isfinite(q.hydro.du_dt(i)));
    EXPECT_NE(q.hydro.du_dt(0), 0.);
    EXPECT_EQ(q.d_dt_extensive.s, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoB, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoS, 0.);
    EXPECT_EQ(q.d_dt_extensive.rhoQ, 0.);
    EXPECT_EQ(q.hydro.d_extensive_bulk_dt, 0.);
}

/// Israel-Stewart bulk relaxation at rest, shear off:
///   tau_Pi dPi/dt + Pi = -zeta theta - delta_PiPi Pi theta + phi_1 Pi^2,
///   d(Pi/sigma)/dt = (dPi/dt + Pi theta)/sigma,  d(s/sigma)/dt = -Pi theta/(sigma T).
TYPED_TEST(CartesianEoMTest, bulk_relaxation_at_rest_follows_israel_stewart){
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
    double theta = 0.;
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

/// Effective pressure p + Pi in the Euler equation at rest: (e + p + Pi) du^i/dt = -d_i(p + Pi).
TYPED_TEST(CartesianEoMTest, bulk_pressure_gradient_accelerates){
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
    for (unsigned i = 0; i < D; ++i)
        EXPECT_TRUE(near(q.hydro.du_dt(i), -(gradP[i] + gradBulk[i])/(w + Pi))) << "du^" << i << "/dt";
}

/// Navier-Stokes source at rest, pi = 0: d(pi^{ij}/sigma)/dt = 2 eta sigma^{ij}/(tau_pi sigma),
///   sigma^{ij} = -(d_i v^j + d_j v^i)/2 + delta^{ij} theta/3,  theta = d_i v^i.
TYPED_TEST(CartesianEoMTest, shear_source_with_velocity_gradients){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = viscous_particle<D>();
    auto& p = systemPtr->particles[0];
    const double eta = 0.6, tau_pi = 1.2;
    const double dv[3][3] = {{0.3, 0.2, 0.07}, {0.05, -0.1, -0.04}, {0.4, -0.3, 0.15}};
    p.hydro.eta_pi = eta;
    p.hydro.tau_pi = tau_pi;
    for (unsigned i = 0; i < D; ++i)
        for (unsigned j = 0; j < D; ++j) p.hydro.gradV(i,j) = dv[i][j];
    const double sigma = p.hydro.sigma;
    evaluate(systemPtr, eom_settings(true));
    const auto& q = systemPtr->particles[0];

    double theta = dv[0][0] + dv[1][1];
    if (D == 3) theta += dv[2][2];
    const double factor = 2.*eta/(tau_pi*sigma);
    EXPECT_TRUE(near(q.hydro.theta, theta));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,0), factor*(-dv[0][0] + theta/3.)));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(1,1), factor*(-dv[1][1] + theta/3.)));
    EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,1), factor*(-(dv[0][1] + dv[1][0])/2.)));
    if (D == 3){
        EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(0,2), factor*(-(dv[0][2] + dv[2][0])/2.)));
        EXPECT_TRUE(near(q.hydro.d_extensive_shv_dt(1,2), factor*(-(dv[1][2] + dv[2][1])/2.)));
    }
}

/// Entropy production by shear: d(s/sigma)/dt = pi^{mu nu} sigma_{mu nu}/(sigma T). With
/// pi^{xx} = pi^{yy} = a (so pi^{zz} = -2a) and d_i v^j = diag(g_x, g_y, g_z):
/// pi^{mu nu} sigma_{mu nu} = a (2 g_z - g_x - g_y), with g_z = 0 in 2+1D.
TYPED_TEST(CartesianEoMTest, shear_produces_entropy){
    constexpr unsigned int D = TypeParam::D;
    auto systemPtr = viscous_particle<D>();
    auto& p = systemPtr->particles[0];
    const double a = 0.05, g[3] = {-0.3, -0.1, 0.25};
    p.hydro.extensive_shv(0,0) = a/p.hydro.sigma;
    p.hydro.extensive_shv(1,1) = a/p.hydro.sigma;
    p.hydro.eta_pi = 0.6;
    p.hydro.tau_pi = 1.2;
    for (unsigned i = 0; i < D; ++i) p.hydro.gradV(i,i) = g[i];
    const double sigma = p.hydro.sigma, T = p.thermo.T;
    evaluate(systemPtr, eom_settings(true));
    const auto& q = systemPtr->particles[0];
    const double gz = (D == 3) ? g[2] : 0.;
    EXPECT_TRUE(near(q.hydro.shv(3,3), -2.*a));
    EXPECT_TRUE(near(q.hydro.shv_nabla_u, a*(2.*gz - g[0] - g[1])));
    EXPECT_TRUE(near(q.d_dt_extensive.s, a*(2.*gz - g[0] - g[1])/(sigma*T)));
}

int main(int argc, char* argv[]){
    Kokkos::ScopeGuard guard(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
