// Transport coefficients (src/transport_coefficients.cpp): every shear, bulk and diffusion
// mode and the second-order coefficients, each against its formula on a fixed thermodynamic
// state. The functions are plain functions of a thermo array and a parameters struct, so
// the state below is the whole input; nothing here is computed by another part of CCAKE.
//
// Expected values are written out by hand from the formulas in the source (and the papers
// they cite). EXPECT_DOUBLE_EQ allows only round-off; values that are exact by
// construction (clamps, zeros, cancellations) use EXPECT_EQ.

#include <cmath>
#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "constants.h"
#include "settings.h"
#include "thermodynamic_info.h"
#include "../src/transport_coefficients.cpp"  // all inline, as src/sph_workstation.cpp includes it

using namespace ccake;
namespace tc = ccake::transport_coefficients;
namespace ti = ccake::thermo_info;

namespace {

/// A thermodynamic state: T = 0.5 fm^-1, s = 4, so w = e + p = T s = 2 at mu = 0; p = 0.5,
/// e = 1.5, cs2 = 1/4 (below 1/3, so every bulk factor (1/3 - cs2)^p is non-zero).
struct State
{
  double v[ti::NUM_THERMO_INFO] = {};
  State()
  {
    v[ti::T] = 0.5;
    v[ti::s] = 4.0;
    v[ti::w] = 2.0;
    v[ti::p] = 0.5;
    v[ti::e] = 1.5;
    v[ti::cs2] = 0.25;
    v[ti::chiBB] = 1.0;
    v[ti::chiBB0] = 1.0;
  }
  /// Baryon chemical potential and density, keeping w = T s + muB nB.
  State& baryon(double muB, double nB)
  {
    v[ti::muB] = muB;
    v[ti::rhoB] = nB;
    v[ti::w] = v[ti::T] * v[ti::s] + muB * nB;
    return *this;
  }
  const double* operator()() const { return v; }
};

/// Settings with every mode named; the tests change what they need.
std::shared_ptr<Settings> settings()
{
  auto s = std::make_shared<Settings>();
  s->etaMode = "constant";
  s->constant_eta_over_s = 0.2;
  s->shearRelaxMode = "default";
  s->zetaMode = "constant";
  s->constant_zeta_over_s = 0.1;
  s->bulkRelaxMode = "default";
  s->modulate_zeta_with_tanh = false;
  s->cs2_dependent_zeta_A = 2.0;
  s->cs2_dependent_zeta_p = 2.0;
  s->bulk_tau_pi_coeff = 5.0;
  s->critical_scaling_bulk = false;
  s->diffusionMode = "constant_over_T2";
  s->diffusionRelaxMode = "constant_over_T";
  for (auto& row : s->constant_kappa_over_T2) row = {0.0, 0.0, 0.0};
  s->delta_pipi_mode = "default";
  s->tau_pipi_mode = "default";
  s->lambda_piPi_mode = "default";
  s->phi6_mode = "default";
  s->phi7_mode = "default";
  s->delta_PiPi_mode = "default";
  s->lambda_Pipi_mode = "default";
  s->phi1_mode = "default";
  s->phi3_mode = "default";
  s->critical_scaling_diffusion = false;
  return s;
}

tc::parameters params(const std::shared_ptr<Settings>& s) { return tc::setup_parameters(s); }

}  // namespace

// ------------------------------------------------------------------ mode names
TEST(TransportModes, EveryNameSelectsItsMode)
{
  struct Case { std::string Settings::*field; const char* name; int tc::parameters::*mode; int value; };
  const Case cases[] = {
    {&Settings::etaMode, "default", &tc::parameters::shear_mode, tc::SHEAR_DEFAULT},
    {&Settings::etaMode, "constant", &tc::parameters::shear_mode, tc::SHEAR_CONST},
    {&Settings::etaMode, "JakiParam", &tc::parameters::shear_mode, tc::SHEAR_JAKI},
    {&Settings::etaMode, "LinearMus", &tc::parameters::shear_mode, tc::SHEAR_LINEAR_MUS},
    {&Settings::etaMode, "piecewise_muB", &tc::parameters::shear_mode, tc::SHEAR_PIECEWISE_MUB},
    {&Settings::etaMode, "interpolate", &tc::parameters::shear_mode, tc::SHEAR_INTERPOLATOR},
    {&Settings::shearRelaxMode, "default", &tc::parameters::shear_relaxation_mode, tc::TAU_PI_SHEAR_DEFAULT},
    {&Settings::shearRelaxMode, "minVal", &tc::parameters::shear_relaxation_mode, tc::TAU_PI_SHEAR_MINVAL},
    {&Settings::shearRelaxMode, "Gubser", &tc::parameters::shear_relaxation_mode, tc::TAU_PI_SHEAR_GUBSER},
    {&Settings::zetaMode, "default", &tc::parameters::zeta_mode, tc::ZETA_DEFAULT},
    {&Settings::zetaMode, "constant", &tc::parameters::zeta_mode, tc::ZETA_CONSTANT},
    {&Settings::zetaMode, "DNMR", &tc::parameters::zeta_mode, tc::ZETA_DNMR},
    {&Settings::zetaMode, "interpolate", &tc::parameters::zeta_mode, tc::ZETA_INTERPOLATE},
    {&Settings::zetaMode, "cs2_dependent", &tc::parameters::zeta_mode, tc::ZETA_CS2_DEPENDENT},
    {&Settings::bulkRelaxMode, "default", &tc::parameters::bulk_relaxation_mode, tc::TAU_PI_BULK_DEFAULT},
    {&Settings::bulkRelaxMode, "DNMR", &tc::parameters::bulk_relaxation_mode, tc::TAU_PI_BULK_DNMR},
    {&Settings::diffusionMode, "constant_over_T2", &tc::parameters::diffusion_mode, tc::KAPPA_DEFAULT},
    {&Settings::diffusionMode, "DNMR", &tc::parameters::diffusion_mode, tc::KAPPA_DNMR},
    {&Settings::diffusionMode, "kappaB_coth", &tc::parameters::diffusion_mode, tc::KAPPA_B_COTH},
    {&Settings::diffusionMode, "constant_nB_over_T", &tc::parameters::diffusion_mode, tc::KAPPA_CONST_NB_OVER_T},
    {&Settings::diffusionMode, "holographic_table", &tc::parameters::diffusion_mode, tc::KAPPA_B_HOLO_TABLE},
    {&Settings::diffusionMode, "something else", &tc::parameters::diffusion_mode, tc::KAPPA_DEFAULT},
    {&Settings::diffusionRelaxMode, "constant_over_T", &tc::parameters::diffusion_relaxation_mode, tc::TAU_Q_DEFAULT},
    {&Settings::diffusionRelaxMode, "cb_over_T", &tc::parameters::diffusion_relaxation_mode, tc::TAU_Q_CB_OVER_T},
    {&Settings::diffusionRelaxMode, "kappa_over_n", &tc::parameters::diffusion_relaxation_mode, tc::TAU_Q_KAPPA_OVER_N},
    {&Settings::diffusionRelaxMode, "cattaneo", &tc::parameters::diffusion_relaxation_mode, tc::TAU_Q_CATTANEO},
    {&Settings::diffusionRelaxMode, "nernst_einstein", &tc::parameters::diffusion_relaxation_mode, tc::TAU_Q_CATTANEO},
    {&Settings::diffusionRelaxMode, "default", &tc::parameters::diffusion_relaxation_mode, tc::TAU_Q_DEFAULT},
    {&Settings::delta_pipi_mode, "israel-stewart", &tc::parameters::delta_pipi_mode, tc::DELTA_S_PIPI_IS},
    {&Settings::delta_pipi_mode, "disabled", &tc::parameters::delta_pipi_mode, tc::DELTA_S_PIPI_DISABLED},
    {&Settings::delta_pipi_mode, "other", &tc::parameters::delta_pipi_mode, tc::DELTA_S_PIPI_DEFAULT},
    {&Settings::tau_pipi_mode, "disabled", &tc::parameters::tau_pipi_mode, tc::TAU_PIPI_DISABLED},
    {&Settings::tau_pipi_mode, "other", &tc::parameters::tau_pipi_mode, tc::TAU_PIPI_DEFAULT},
    {&Settings::lambda_piPi_mode, "disabled", &tc::parameters::lambda_piPi_mode, tc::LAMBDA_S_PIPI_DISABLED},
    {&Settings::lambda_piPi_mode, "other", &tc::parameters::lambda_piPi_mode, tc::LAMBDA_S_PIPI_DEFAULT},
    {&Settings::phi6_mode, "disabled", &tc::parameters::phi6_mode, tc::PHI6_DISABLED},
    {&Settings::phi6_mode, "other", &tc::parameters::phi6_mode, tc::PHI6_DEFAULT},
    {&Settings::phi7_mode, "disabled", &tc::parameters::phi7_mode, tc::PHI7_DISABLED},
    {&Settings::phi7_mode, "other", &tc::parameters::phi7_mode, tc::PHI7_DEFAULT},
    {&Settings::delta_PiPi_mode, "israel-stewart", &tc::parameters::delta_PiPi_mode, tc::DELTA_B_PIPI_IS},
    {&Settings::delta_PiPi_mode, "disabled", &tc::parameters::delta_PiPi_mode, tc::DELTA_B_PIPI_DISABLED},
    {&Settings::delta_PiPi_mode, "other", &tc::parameters::delta_PiPi_mode, tc::DELTA_B_PIPI_DEFAULT},
    {&Settings::lambda_Pipi_mode, "disabled", &tc::parameters::lambda_Pipi_mode, tc::LAMBDA_B_PIPI_DISABLED},
    {&Settings::lambda_Pipi_mode, "other", &tc::parameters::lambda_Pipi_mode, tc::LAMBDA_B_PIPI_DEFAULT},
    {&Settings::phi1_mode, "disabled", &tc::parameters::phi1_mode, tc::PHI1_DISABLED},
    {&Settings::phi1_mode, "other", &tc::parameters::phi1_mode, tc::PHI1_DEFAULT},
    {&Settings::phi3_mode, "disabled", &tc::parameters::phi3_mode, tc::PHI3_DISABLED},
    {&Settings::phi3_mode, "other", &tc::parameters::phi3_mode, tc::PHI3_DEFAULT},
  };
  for (const auto& c : cases)
  {
    auto s = settings();
    (*s).*(c.field) = c.name;
    EXPECT_EQ(params(s).*(c.mode), c.value) << "mode name '" << c.name << "'";
  }
}

TEST(TransportModes, CriticalScalingDiffusionMode)
{
  auto s = settings();
  EXPECT_EQ(params(s).critical_scaling_diffusion_mode, 0);  // gaussian_suppression
  s->critical_scaling_diffusion_mode = "xi_power";
  EXPECT_EQ(params(s).critical_scaling_diffusion_mode, 1);
  // MeV in the settings, natural units in the parameters
  s->critical_point_T = 2.0 * constants::hbarc_MeVfm;
  EXPECT_DOUBLE_EQ(params(s).critical_point_T, 2.0);
}

// ----------------------------------------------------------------------- shear
TEST(Shear, Viscosity)
{
  const State st;  // w/T = 4
  auto s = settings();
  s->etaMode = "default";
  EXPECT_DOUBLE_EQ(tc::eta(st(), params(s)), 0.2 * 4.0);
  s->etaMode = "constant";
  s->constant_eta_over_s = 0.08;
  EXPECT_DOUBLE_EQ(tc::eta(st(), params(s)), 0.08 * 4.0);
  s->etaMode = "interpolate";  // a stub
  EXPECT_EQ(tc::eta(st(), params(s)), 0.0);

  State cold = st;
  cold.v[ti::T] = 0.0;
  s->etaMode = "constant";
  EXPECT_EQ(tc::eta(cold(), params(s)), 0.0);
  s->etaMode = "LinearMus";
  EXPECT_EQ(tc::eta(cold(), params(s)), 0.0);
  s->etaMode = "piecewise_muB";
  EXPECT_EQ(tc::eta(cold(), params(s)), 0.0);
}

TEST(Shear, JakiParametrization)
{
  // eta = s [0.0416762/alpha^1.6 + 0.0388977/T^5.1], alpha = 33/(12 pi) (z-1)/(z ln z),
  // z = (0.66 T/T_c)^2, T_c = 155 MeV
  const State st;
  auto s = settings();
  s->etaMode = "JakiParam";
  const double T = 0.5, Tc = 155.0 / constants::hbarc_MeVfm;
  const double z = std::pow(0.66 * T / Tc, 2);
  const double alpha = 33. / (12. * M_PI) * (z - 1) / (z * std::log(z));
  EXPECT_DOUBLE_EQ(tc::eta(st(), params(s)),
                   4.0 * (0.0416762 / std::pow(alpha, 1.6) + 0.0388977 / std::pow(T, 5.1)));
}

TEST(Shear, LinearInChemicalPotentials)
{
  // (eta/s + 0.0033 (muB + muS + muQ)) w/T
  State st;
  st.v[ti::muB] = 1.0;
  st.v[ti::muS] = 0.5;
  st.v[ti::muQ] = -0.25;
  auto s = settings();
  s->etaMode = "LinearMus";
  s->constant_eta_over_s = 0.1;
  EXPECT_DOUBLE_EQ(tc::eta(st(), params(s)), (0.1 + 0.0033 * 1.25) * st.v[ti::w] / 0.5);
}

TEST(Shear, PiecewiseInBaryonChemicalPotential)
{
  // eta/s linear between the nodes eta_0, eta_2, eta_4 at |muB| = 0, 0.2, 0.4 GeV, flat beyond
  auto s = settings();
  s->etaMode = "piecewise_muB";
  s->shear_eta_0 = 0.1;
  s->shear_eta_2 = 0.2;
  s->shear_eta_4 = 0.4;
  const auto p = params(s);
  auto at = [&](double muB_GeV) {
    State st;
    st.v[ti::muB] = muB_GeV / constants::hbarc_GeVfm;
    return tc::eta(st(), p) / 4.0;  // eta/s, w/T = 4
  };
  EXPECT_DOUBLE_EQ(at(0.0), 0.1);
  EXPECT_DOUBLE_EQ(at(0.1), 0.15);
  EXPECT_DOUBLE_EQ(at(-0.1), 0.15);  // symmetric in muB
  EXPECT_DOUBLE_EQ(at(0.3), 0.3);
  EXPECT_DOUBLE_EQ(at(0.5), 0.4);
  EXPECT_DOUBLE_EQ(at(2.0), 0.4);
}

TEST(Shear, RelaxationTime)
{
  // tau_pi = 5 eta/w: with eta/s = 0.2, 5 (0.2 w/T)/w = 1/T = 2 fm
  const State st;
  auto s = settings();
  for (const char* mode : {"default", "minVal", "Gubser"})
  {
    s->shearRelaxMode = mode;
    EXPECT_DOUBLE_EQ(tc::tau_pi(st(), params(s)), 2.0) << mode;
  }
  // without viscosity the floors show: 0.005 fm (default), 0.001 fm (minVal), none (Gubser)
  s->constant_eta_over_s = 0.0;
  s->shearRelaxMode = "default";
  EXPECT_EQ(tc::tau_pi(st(), params(s)), 0.005);
  s->shearRelaxMode = "minVal";
  EXPECT_EQ(tc::tau_pi(st(), params(s)), 0.001);
  s->shearRelaxMode = "Gubser";
  EXPECT_EQ(tc::tau_pi(st(), params(s)), 0.0);
}

TEST(Shear, SecondOrderCoefficients)
{
  const State st;  // tau_pi = 2 fm, p = 0.5
  auto s = settings();
  const double tau_pi = 2.0;
  EXPECT_DOUBLE_EQ(tc::delta_pipi(st(), params(s)), 4.0 / 3.0 * tau_pi);
  EXPECT_DOUBLE_EQ(tc::tau_pipi(st(), params(s)), 10.0 / 7.0 * tau_pi);
  EXPECT_DOUBLE_EQ(tc::lambda_piPi(st(), params(s)), 6.0 / 5.0 * tau_pi);
  EXPECT_EQ(tc::phi6(st(), params(s)), 0.0);
  EXPECT_DOUBLE_EQ(tc::phi7(st(), params(s)), 9.0 / (70.0 * 0.5));
  s->delta_pipi_mode = "israel-stewart";
  EXPECT_DOUBLE_EQ(tc::delta_pipi(st(), params(s)), tau_pi);
  s->delta_pipi_mode = s->tau_pipi_mode = s->lambda_piPi_mode = "disabled";
  s->phi6_mode = s->phi7_mode = "disabled";
  const auto off = params(s);
  EXPECT_EQ(tc::delta_pipi(st(), off), 0.0);
  EXPECT_EQ(tc::tau_pipi(st(), off), 0.0);
  EXPECT_EQ(tc::lambda_piPi(st(), off), 0.0);
  EXPECT_EQ(tc::phi6(st(), off), 0.0);
  EXPECT_EQ(tc::phi7(st(), off), 0.0);
}

// ------------------------------------------------------------------------ bulk
TEST(Bulk, Viscosity)
{
  const State st;  // w/T = 4, 1/3 - cs2 = 1/12
  auto s = settings();
  s->zetaMode = "default";
  EXPECT_DOUBLE_EQ(tc::zeta(st(), params(s)), 0.005 * 4.0);
  s->zetaMode = "constant";
  EXPECT_DOUBLE_EQ(tc::zeta(st(), params(s)), 0.1 * 4.0);
  s->zetaMode = "DNMR";  // a stub
  EXPECT_EQ(tc::zeta(st(), params(s)), 0.0);
  s->zetaMode = "cs2_dependent";  // A (1/3 - cs2)^p w/T
  EXPECT_DOUBLE_EQ(tc::zeta(st(), params(s)), 2.0 * std::pow(1.0 / 12.0, 2) * 4.0);
}

TEST(Bulk, TanhFactorIsOneAtTheTransition)
{
  // factor = (1 + x^p)/2 + (1 - x^p)/2 tanh((T - 150 MeV)/10 MeV), x = T/150 MeV: 1 at T = 150 MeV
  State st;
  st.v[ti::T] = 150.0 / constants::hbarc_MeVfm;
  auto s = settings();
  s->zetaMode = "cs2_dependent";
  const double without = tc::zeta(st(), params(s));
  s->modulate_zeta_with_tanh = true;
  EXPECT_DOUBLE_EQ(tc::zeta(st(), params(s)), without);
  // below the transition it is the formula (tends to x^p)
  const double h = constants::hbarc_MeVfm;
  st.v[ti::T] = 120.0 / h;
  const double low = tc::zeta(st(), params(s));
  s->modulate_zeta_with_tanh = false;
  const double xp = std::pow(120.0 / 150.0, 2);
  const double factor = 0.5 * (1 + xp) + 0.5 * (1 - xp) * std::tanh((120.0 / h - 150.0 / h) / (10.0 / h));
  EXPECT_DOUBLE_EQ(low, tc::zeta(st(), params(s)) * factor);
  EXPECT_LT(factor, 1.0);
}

TEST(Bulk, RelaxationTime)
{
  State st;
  auto s = settings();
  // constant zeta: tau_Pi = coeff zeta/((1/3 - cs2)^p w) = 5 (0.1 w/T)/((1/12)^2 w) = 5 0.1 144/T
  EXPECT_DOUBLE_EQ(tc::tau_Pi(st(), params(s)), 5.0 * 0.1 * 144.0 / 0.5);
  // conformal: the division is guarded
  st.v[ti::cs2] = 1.0 / 3.0;
  EXPECT_EQ(tc::tau_Pi(st(), params(s)), 1e10);
  // cs2_dependent: (1/3 - cs2)^p cancels, tau_Pi = coeff A/T whatever cs2 is
  s->zetaMode = "cs2_dependent";
  for (double cs2 : {0.05, 0.25, 0.3})
  {
    st.v[ti::cs2] = cs2;
    EXPECT_DOUBLE_EQ(tc::tau_Pi(st(), params(s)), 5.0 * 2.0 / 0.5) << "cs2 " << cs2;
  }
  // floor 0.1 fm
  s->cs2_dependent_zeta_A = 1e-6;
  EXPECT_EQ(tc::tau_Pi(st(), params(s)), 0.1);
}

TEST(Bulk, RelaxationTimeDNMR)
{
  // tau_Pi = zeta/(15 (1/3 - cs2)^2 w), the gap floored at 1e-3, the result at 0.1 fm
  State st;
  auto s = settings();
  s->bulkRelaxMode = "DNMR";
  s->constant_zeta_over_s = 10.0;
  EXPECT_DOUBLE_EQ(tc::tau_Pi(st(), params(s)), 10.0 * 4.0 / (15.0 * std::pow(1.0 / 12.0, 2) * 2.0));
  st.v[ti::cs2] = 0.5;
  EXPECT_DOUBLE_EQ(tc::tau_Pi(st(), params(s)), 10.0 * 4.0 / (15.0 * 1e-6 * 2.0));
  st.v[ti::w] = 0.0;
  EXPECT_EQ(tc::tau_Pi(st(), params(s)), 0.1);
  st = State();
  s->constant_zeta_over_s = 1e-9;
  EXPECT_EQ(tc::tau_Pi(st(), params(s)), 0.1);
}

TEST(Bulk, SecondOrderCoefficients)
{
  const State st;
  auto s = settings();
  const double tau_Pi = tc::tau_Pi(st(), params(s));
  EXPECT_DOUBLE_EQ(tc::delta_PiPi(st(), params(s)), 2.0 / 3.0 * tau_Pi);
  EXPECT_DOUBLE_EQ(tc::lambda_Pipi(st(), params(s)), 8.0 / 5.0 * (1.0 / 12.0) * tau_Pi);
  EXPECT_EQ(tc::phi1(st(), params(s)), 0.0);
  EXPECT_EQ(tc::phi3(st(), params(s)), 0.0);
  s->delta_PiPi_mode = "israel-stewart";
  EXPECT_DOUBLE_EQ(tc::delta_PiPi(st(), params(s)), tau_Pi);
  s->delta_PiPi_mode = s->lambda_Pipi_mode = s->phi1_mode = s->phi3_mode = "disabled";
  const auto off = params(s);
  EXPECT_EQ(tc::delta_PiPi(st(), off), 0.0);
  EXPECT_EQ(tc::lambda_Pipi(st(), off), 0.0);
  EXPECT_EQ(tc::phi1(st(), off), 0.0);
  EXPECT_EQ(tc::phi3(st(), off), 0.0);
}

TEST(Bulk, CriticalScaling)
{
  // local baseline: factor = [1 + (xi/xi_0)^3]/2, xi/xi_0 = sqrt(chiBB/chiBB0), capped
  State st;
  auto s = settings();
  s->critical_scaling_bulk = true;
  s->critical_scaling_local_xi0 = true;
  s->critical_scaling_max = 10.0;
  auto p = params(s);
  EXPECT_EQ(tc::critical_scaling_factor_bulk(st(), p), 1.0);  // chiBB = chiBB0
  st.v[ti::chiBB] = 4.0;                                       // xi/xi_0 = 2
  EXPECT_EQ(tc::critical_scaling_factor_bulk(st(), p), 4.5);
  s->zetaMode = "constant";
  EXPECT_DOUBLE_EQ(tc::zeta(st(), p), 4.5 * 0.1 * 4.0);
  s->critical_scaling_max = 3.0;
  EXPECT_EQ(tc::critical_scaling_factor_bulk(st(), params(s)), 3.0);
  s->critical_scaling_max = 0.0;  // no cap
  st.v[ti::chiBB] = 100.0;        // xi/xi_0 = 10
  EXPECT_EQ(tc::critical_scaling_factor_bulk(st(), params(s)), 500.5);
  // an unstable region (chiBB < 0) gets no enhancement
  st.v[ti::chiBB] = -1.0;
  EXPECT_EQ(tc::critical_scaling_factor_bulk(st(), params(s)), 1.0);
  // global baseline xi_0 = 1.0176198446545879
  s->critical_scaling_local_xi0 = false;
  st.v[ti::chiBB] = 4.0;
  EXPECT_DOUBLE_EQ(tc::critical_scaling_factor_bulk(st(), params(s)),
                   1.0 + std::pow(2.0 / 1.0176198446545879, 3));
  // the cs2_dependent relaxation time carries the factor too
  s->zetaMode = "cs2_dependent";
  s->critical_scaling_local_xi0 = true;
  EXPECT_DOUBLE_EQ(tc::tau_Pi(st(), params(s)), 4.5 * 5.0 * 2.0 / 0.5);
}

// ------------------------------------------------------------------- diffusion
namespace {
std::shared_ptr<Settings> diffusion(const char* mode, const char* relaxation)
{
  auto s = settings();
  s->diffusionMode = mode;
  s->diffusionRelaxMode = relaxation;
  s->constant_kappa_over_T2 = {{{0.3, 0.1, 0.0}, {0.1, 0.2, 0.0}, {0.0, 0.0, 0.4}}};
  return s;
}
}  // namespace

TEST(Diffusion, ConstantOverT2)
{
  // kappa_ij = c_ij T^2
  const State st;
  const auto k = tc::kappa(st(), params(diffusion("constant_over_T2", "constant_over_T")));
  const double c[3][3] = {{0.3, 0.1, 0.0}, {0.1, 0.2, 0.0}, {0.0, 0.0, 0.4}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_DOUBLE_EQ(k(i, j), c[i][j] * 0.25) << i << j;
}

TEST(Diffusion, DensityWeighted)
{
  // kappa_ij = c_ij sqrt|rho_i rho_j|/T
  State st;
  st.v[ti::rhoB] = 0.4;
  st.v[ti::rhoS] = -0.1;
  st.v[ti::rhoQ] = 0.9;
  const auto p = params(diffusion("constant_nB_over_T", "constant_over_T"));
  const auto k = tc::kappa(st(), p);
  EXPECT_DOUBLE_EQ(k(0, 0), 0.3 * 0.4 / 0.5);
  EXPECT_DOUBLE_EQ(k(0, 1), 0.1 * 0.2 / 0.5);
  EXPECT_DOUBLE_EQ(k(2, 2), 0.4 * 0.9 / 0.5);
  st.v[ti::T] = 0.0;
  EXPECT_EQ(tc::kappa(st(), p)(0, 0), 0.0);
}

TEST(Diffusion, DNMRIsAStub)
{
  const auto p = params(diffusion("DNMR", "constant_over_T"));
  EXPECT_EQ(tc::kappa(State()(), p)(0, 0), 0.0);
  EXPECT_EQ(tc::tauq(State()(), p)(0, 0), 0.0);
}

TEST(Diffusion, CothForm)
{
  // kappa_B = (C_B/T) |nB| max[coth|muB/T|/3 - |nB| T/w, 0], (B,B) entry only
  auto s = diffusion("kappaB_coth", "cb_over_T");
  s->C_B = 0.4;
  const auto p = params(s);
  State st;
  st.baryon(0.5, 0.1);  // muB/T = 1, w = 2.05
  const double expected = 0.4 / 0.5 * 0.1 * (1.0 / 3.0 / std::tanh(1.0) - 0.1 * 0.5 / 2.05);
  const auto k = tc::kappa(st(), p);
  EXPECT_DOUBLE_EQ(k(0, 0), expected);
  EXPECT_EQ(k(1, 1), 0.0);
  st.baryon(-0.5, -0.1);  // symmetric
  EXPECT_DOUBLE_EQ(tc::kappa(st(), p)(0, 0), expected);
  st.baryon(0.004, 0.1);  // |muB/T| <= 0.01: off
  EXPECT_EQ(tc::kappa(st(), p)(0, 0), 0.0);
  st.baryon(0.5, 10.0);  // dense core: bracket < 0, kappa_B = 0
  EXPECT_EQ(tc::kappa(st(), p)(0, 0), 0.0);
}

TEST(Diffusion, RelaxationTimes)
{
  State st;  // T = 0.5, chiBB = 1
  // constant_over_T: 0.2/T on the diagonal, unless the causality floor D_B = kappa_BB/(T chiBB)
  // is larger. kappa_BB = 0.3 T^2, D_B = 0.3 T = 0.15 < 0.4: no floor
  auto s = diffusion("constant_over_T2", "constant_over_T");
  auto q = tc::tauq(st(), params(s));
  for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(q(i, i), 0.4);
  EXPECT_EQ(q(0, 1), 0.0);
  st.v[ti::chiBB] = 0.25;  // D_B = 0.6 > 0.4: floor
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.3 * 0.25 / (0.5 * 0.25));
  s->tau_q_causality_floor = false;
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.4);
  // constant_nB_over_T uses 0.2/T too
  s = diffusion("constant_nB_over_T", "constant_over_T");
  EXPECT_DOUBLE_EQ(tc::tauq(State()(), params(s))(2, 2), 0.4);
}

TEST(Diffusion, CbOverT)
{
  // tau_q = C_B/T for the coth and holographic modes; 0.2/T with constant_over_T
  auto s = diffusion("kappaB_coth", "cb_over_T");
  s->C_B = 0.4;
  State st;
  st.baryon(0.5, 0.1);
  st.v[ti::chiBB] = 100.0;  // D_B tiny: no floor
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(1, 1), 0.4 / 0.5);
  s->diffusionRelaxMode = "constant_over_T";
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.2 / 0.5);
}

TEST(Diffusion, Cattaneo)
{
  // tau_q = D_B/v0^2, D_B = kappa_BB/(T chiBB), v0^2 = cs2 (or the configured value),
  // clamped to [min, max]/T, then raised to D_B
  auto s = diffusion("constant_over_T2", "cattaneo");
  State st;  // kappa_BB = 0.075, D_B = 0.15, cs2 = 1/4: tau_q = 0.6, inside [0.2, 20]
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.15 / 0.25);
  s->diffusion_wave_speed_sq = 0.5;
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(2, 2), 0.15 / 0.5);
  s->tau_q_max_over_T = 0.1;  // max 0.2 fm, but the floor D_B = 0.15 still holds below it
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.2);
  s->tau_q_max_over_T = 0.05;  // max 0.1 fm < D_B: the causality floor wins
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.15);
  // no susceptibility: tau_q_max
  s = diffusion("constant_over_T2", "cattaneo");
  st.v[ti::chiBB] = 0.0;
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 10.0 / 0.5);
  st.v[ti::T] = 0.0;
  EXPECT_EQ(tc::tauq(st(), params(s))(0, 0), 0.0);
}

TEST(Diffusion, KappaOverN)
{
  // tau_q = b_q kappa_BB/|nB|, clamped to [min, max]/T, then raised to D_B
  auto s = diffusion("constant_over_T2", "kappa_over_n");
  s->b_q = 2.0;
  State st;
  st.v[ti::rhoB] = 0.5;  // tau_q = 2 0.075/0.5 = 0.3 > D_B = 0.15
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.3);
  st.v[ti::rhoB] = 0.0;  // no density: tau_q_max
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(1, 1), 20.0);
  st.v[ti::rhoB] = 100.0;  // below the minimum 0.1/T = 0.2, then the floor 0.15 < 0.2
  EXPECT_DOUBLE_EQ(tc::tauq(st(), params(s))(0, 0), 0.2);
  st.v[ti::T] = 0.0;
  EXPECT_EQ(tc::tauq(st(), params(s))(0, 0), 0.0);
}

TEST(Diffusion, CriticalFactors)
{
  // Only for particles on a table EoS, and only when switched on.
  auto s = diffusion("constant_over_T2", "constant_over_T");
  s->critical_scaling_diffusion = true;
  State st;
  const double kappa_off = 0.3 * 0.25;
  EXPECT_DOUBLE_EQ(tc::kappa(st(), params(s))(0, 0), kappa_off);  // not a table EoS
  st.v[ti::eos_type_is_table] = 1.0;
  // Gaussian suppression 1 - exp(-...) is exactly 0 at the critical point
  s->critical_point_T = 0.5 * constants::hbarc_MeVfm;
  s->critical_gaussian_width_T = 10.0;
  s->critical_gaussian_width_muB = 10.0;
  auto p = params(s);
  p.critical_point_T = 0.5;  // exactly the state's T
  EXPECT_EQ(tc::kappa(st(), p)(0, 0), 0.0);
  // xi_power: g(q) = min{[1 + (xi/xi_0)^q]/2, max}, xi/xi_0 = sqrt(chiBB/chiBB0)
  s->critical_scaling_diffusion_mode = "xi_power";
  s->critical_scaling_kappa_power = 1.0;
  s->critical_scaling_tauq_power = 2.0;
  s->tau_q_causality_floor = false;
  p = params(s);
  EXPECT_DOUBLE_EQ(tc::kappa(st(), p)(0, 0), kappa_off);  // chiBB = chiBB0: g = 1
  st.v[ti::chiBB] = 4.0;                                     // xi/xi_0 = 2
  EXPECT_DOUBLE_EQ(tc::kappa(st(), p)(0, 0), 1.5 * kappa_off);
  EXPECT_DOUBLE_EQ(tc::tauq(st(), p)(0, 0), 2.5 * 0.4);
  // cattaneo already carries g(p_kappa) and chiBB/chiBB0 through D_B: they are divided out,
  // so tau_q = tau_q,nc g(p_tauq), with tau_q,nc the value at chiBB = chiBB0
  s->diffusionRelaxMode = "cattaneo";
  p = params(s);
  State nc = st;
  nc.v[ti::chiBB] = 1.0;
  EXPECT_DOUBLE_EQ(tc::tauq(st(), p)(0, 0), 2.5 * tc::tauq(nc(), p)(0, 0));
  // capped
  s->critical_scaling_max = 1.2;
  EXPECT_DOUBLE_EQ(tc::kappa(st(), params(s))(0, 0), 1.2 * kappa_off);
}

// ------------------------------------------------------ holographic sigma_B table
namespace {
/// T, muB, sigma_B in MeV on a 2 x 2 grid: T = 150, 250; muB = 0, 200.
void write_table(const std::string& path, const std::string& extra = "")
{
  std::ofstream f(path);
  f << "# T muB sigma_B\n150 0 10\n150 200 20\n250 0 30\n250 200 40\n" << extra;
}
}  // namespace

TEST(HolographicTable, NodesAndInterpolation)
{
  write_table("sigmaB.dat");
  auto s = diffusion("holographic_table", "constant_over_T");
  s->holographic_sigmaB_table_path = "sigmaB.dat";
  s->holographic_diffusion_scale = 2.0;
  auto p = params(s);
  tc::load_holographic_sigmaB_table(p, s);
  ASSERT_EQ(p.holo_sigmaB_num_temperatures, 2);
  ASSERT_EQ(p.holo_sigmaB_num_potentials, 2);
  const double h = constants::hbarc_MeVfm;
  auto kappa_at = [&](double T_MeV, double muB_MeV) {
    State st;
    st.v[ti::T] = T_MeV / h;
    st.v[ti::muB] = muB_MeV / h;
    return tc::kappa(st(), p)(0, 0);
  };
  // kappa_B = scale T sigma_B, sigma_B converted to fm^-1
  EXPECT_DOUBLE_EQ(kappa_at(150, 0), 2.0 * (150 / h) * (10 / h));
  EXPECT_DOUBLE_EQ(kappa_at(250, 200), 2.0 * (250 / h) * (40 / h));
  EXPECT_DOUBLE_EQ(kappa_at(200, 100), 2.0 * (200 / h) * (25 / h));  // bilinear: the mean
  EXPECT_DOUBLE_EQ(kappa_at(200, -100), kappa_at(200, 100));       // symmetric in muB
  EXPECT_DOUBLE_EQ(kappa_at(400, 600), 2.0 * (400 / h) * (40 / h));  // the edge value outside
  EXPECT_EQ(kappa_at(0, 0), 0.0);
  // the other diagonal entries stay 0; tau_q is 0.2/T, or C_B/T with cb_over_T
  State st;
  EXPECT_EQ(tc::kappa(st(), p)(1, 1), 0.0);
  EXPECT_DOUBLE_EQ(tc::tauq(st(), p)(0, 0), 0.4);
  p.diffusion_relaxation_mode = tc::TAU_Q_CB_OVER_T;
  p.C_B = 3.0;
  EXPECT_DOUBLE_EQ(tc::tauq(st(), p)(0, 0), 3.0 / 0.5);
}

TEST(HolographicTable, NoTableNoDiffusion)
{
  // without a loaded table the mode gives kappa_B = 0 (and the loader ignores other modes)
  auto s = diffusion("constant_over_T2", "constant_over_T");
  s->holographic_sigmaB_table_path = "does_not_exist.dat";
  auto p = params(s);
  tc::load_holographic_sigmaB_table(p, s);  // not the holographic mode: nothing is read
  p.diffusion_mode = tc::KAPPA_B_HOLO_TABLE;
  EXPECT_EQ(tc::kappa(State()(), p)(0, 0), 0.0);
}

TEST(HolographicTableDeathTest, BadTablesStopTheRun)
{
  auto s = diffusion("holographic_table", "constant_over_T");
  auto load = [&](const std::string& path) {
    s->holographic_sigmaB_table_path = path;
    auto p = params(s);
    tc::load_holographic_sigmaB_table(p, s);
  };
  EXPECT_EXIT(load("does_not_exist.dat"), testing::ExitedWithCode(EXIT_FAILURE), "could not open");
  std::ofstream("empty.dat") << "# nothing\n";
  EXPECT_EXIT(load("empty.dat"), testing::ExitedWithCode(EXIT_FAILURE), "contains no");
  write_table("ragged.dat", "350 0 50\n");
  EXPECT_EXIT(load("ragged.dat"), testing::ExitedWithCode(EXIT_FAILURE), "rectangular");
  std::ofstream("uneven.dat") << "100 0 1\n100 10 1\n200 0 1\n200 10 1\n400 0 1\n400 10 1\n";
  EXPECT_EXIT(load("uneven.dat"), testing::ExitedWithCode(EXIT_FAILURE), "non-uniform");
}

int main(int argc, char* argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
