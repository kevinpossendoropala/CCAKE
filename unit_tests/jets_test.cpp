// BBMG jets (include/jets.h) and the jet files of Output.
//
// A jet moves at the speed of light in its direction and loses energy at the rate
// dE/dtau = kappa(T) tau^z rho^c Gamma_flow (BBMG), which never exceeds what it has
// left. The medium here is a static lattice of SPH particles at rest with one
// temperature T and pressure p, so the interpolated T and rho = p/T are those values,
// the flow factor is 1, and each step's motion and loss follow from the formulas.
// 3+1D and 2+1D (typed tests, CTest names ending in <In3D> and <In2D>).
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "constants.h"
#include "eom_default.h"
#include "jets.h"
#include "output.h"
#include "settings.h"
#include "sph_workstation.h"
#include "system_state.h"

namespace cc = ccake;

struct In3D { static constexpr unsigned int D = 3; };
struct In2D { static constexpr unsigned int D = 2; };
using Dimensions = ::testing::Types<In3D, In2D>;

namespace {

constexpr double TAU = 1.0, DT = 0.1, PRESSURE = 2.0;
constexpr double HOT = 1.0;    // fm^-1, 197 MeV: above the 150 MeV decoherence temperature
constexpr double COLD = 0.5;   // fm^-1, 99 MeV: below it

std::shared_ptr<Settings> jet_settings(){
  auto s = std::make_shared<Settings>();
  s->hT = 0.5;
  s->hEta = 0.5;
  s->dt = DT;
  s->results_directory = ".";
  s->jets_input_mode = "hardcoded";
  s->jets_pT = 100.0;
  s->jets_phi = 0.3;
  s->jets_rapidity = 0.2;
  s->jets_x0 = s->jets_y0 = s->jets_eta0 = 0.0;
  s->jets_Energy_scaling = 0;   // a
  s->jets_Length_scaling = 1;   // z: dE/dtau ~ tau^1
  s->jets_Fluctuations = 0;     // c = (2 + z - a)/3 = 1: dE/dtau ~ rho^1
  return s;
}

/// A lattice at rest around the origin (spacing 0.5 fm, also in eta_s in 3+1D), all at
/// temperature T and pressure PRESSURE.
template <unsigned int D>
std::shared_ptr<cc::SystemState<D>> medium(std::shared_ptr<Settings> s, double T){
  auto sys = std::make_shared<cc::SystemState<D>>(s);
  const int n_eta = (D == 3) ? 3 : 1;
  int id = 0;
  for (int i = -2; i <= 2; ++i)
    for (int j = -2; j <= 2; ++j)
      for (int k = 0; k < n_eta; ++k){
        cc::Particle<D> p;
        p.ID = id++;
        p.r(0) = 0.5*i;
        p.r(1) = 0.5*j;
        if (D == 3) p.r(2) = 0.5*(k - 1);
        p.thermo.T = T;
        p.thermo.p = PRESSURE;
        sys->add_particle(p);
      }
  sys->n_particles = id;
  sys->t = TAU;
  sys->allocate_cabana_particles();
  return sys;
}

// The partner is placed at phi + PI, with PI from include/mathdef.h: pi in single
// precision (3.1415927410...), so the pair is back to back to within 1e-7 rad.
const double CODE_PI = PI;

/// kappa(T) of BBMG<D>::get_kappa, T in MeV.
double kappa(double T_MeV){ return 2.5*std::exp(-9.22*T_MeV/1000) + 0.04; }

} // namespace

template <typename T> class Jets : public ::testing::Test {};
TYPED_TEST_SUITE(Jets, Dimensions);

/// One jet and its back-to-back partner at the configured point; the energy is
/// pT cosh(y), y the rapidity (0 in 2+1D), and the medium values are taken there.
TYPED_TEST(Jets, hardcoded_pair_starts_with_its_energy){
  constexpr unsigned int D = TypeParam::D;
  auto s = jet_settings();
  auto sys = medium<D>(s, HOT);
  cc::BBMG<D> bbmg(s, sys);
  bbmg.initial();
  ASSERT_EQ(bbmg.jetInfo_host.size(), 2u);
  const double y = (D == 3) ? 0.2 : 0.0;
  for (int j = 0; j < 2; ++j){
    const auto& jet = bbmg.jetInfo_host[j];
    EXPECT_EQ(jet.PID, j);
    EXPECT_DOUBLE_EQ(jet.phi, 0.3 + j*CODE_PI);
    EXPECT_DOUBLE_EQ(jet.rapidity, y);
    EXPECT_DOUBLE_EQ(jet.E0, 100.0*std::cosh(y));
    EXPECT_DOUBLE_EQ(jet.total_energy, jet.E0);
    EXPECT_DOUBLE_EQ(jet.T0, HOT*constants::hbarc_MeVfm);
    EXPECT_DOUBLE_EQ(jet.rho0, PRESSURE/HOT);
    EXPECT_EQ(jet.tau_stop, -1.0);
  }
}

/// One step through a hot static medium: the jet moves by dt along its direction
/// (dx/dtau = cos(phi)/cosh(y - eta_s), deta_s/dtau = tanh(y - eta_s)/tau) and loses
/// kappa(T) tau^z rho^c dt, which the energy budget accounts for exactly.
TYPED_TEST(Jets, one_step_through_a_hot_medium){
  constexpr unsigned int D = TypeParam::D;
  auto s = jet_settings();
  auto sys = medium<D>(s, HOT);
  cc::BBMG<D> bbmg(s, sys);
  bbmg.initial();
  bbmg.propagate();
  bbmg.copy_device_to_host_BBMG();
  const auto& jet = bbmg.jetInfo_host[0];
  const double dy = (D == 3) ? 0.2 : 0.0;
  EXPECT_DOUBLE_EQ(jet.r[0], DT*std::cos(0.3)/std::cosh(dy));
  EXPECT_DOUBLE_EQ(jet.r[1], DT*std::sin(0.3)/std::cosh(dy));
  if (D == 3) EXPECT_DOUBLE_EQ(jet.r[2], DT*std::tanh(dy)/TAU);
  // a static medium of one temperature
  EXPECT_DOUBLE_EQ(jet.T, HOT*constants::hbarc_MeVfm);
  EXPECT_DOUBLE_EQ(jet.rho, PRESSURE/HOT);
  EXPECT_EQ(jet.vmag, 0.0);
  EXPECT_EQ(jet.gam, 1.0);
  EXPECT_EQ(jet.flow, 1.0);
  // dE = kappa(T) tau^1 rho^1 dt flow
  const double dE = kappa(jet.T)*std::exp(1*std::log(TAU))*std::exp(1*std::log(jet.rho))*DT;
  EXPECT_DOUBLE_EQ(jet.e_loss, dE);
  EXPECT_DOUBLE_EQ(jet.total_energy + jet.e_loss, jet.E0);
  EXPECT_EQ(jet.tau_stop, -1.0);  // still inside the medium
  // the partner goes the other way
  EXPECT_DOUBLE_EQ(bbmg.jetInfo_host[1].r[0], DT*std::cos(0.3 + CODE_PI)/std::cosh(dy));
}

/// A jet with less energy than one step would take is absorbed: its energy goes to
/// exactly 0, all of it into the medium, and it stops there.
TYPED_TEST(Jets, a_soft_jet_is_absorbed){
  constexpr unsigned int D = TypeParam::D;
  auto s = jet_settings();
  s->jets_pT = 1e-9;
  auto sys = medium<D>(s, HOT);
  cc::BBMG<D> bbmg(s, sys);
  bbmg.initial();
  bbmg.propagate();
  bbmg.copy_device_to_host_BBMG();
  const auto first = bbmg.jetInfo_host[0];
  EXPECT_EQ(first.total_energy, 0.0);
  EXPECT_EQ(first.e_loss, first.E0);
  EXPECT_EQ(first.tau_stop, TAU);
  // afterwards it neither moves nor deposits
  bbmg.propagate();
  bbmg.copy_device_to_host_BBMG();
  const auto& later = bbmg.jetInfo_host[0];
  EXPECT_EQ(later.e_loss, 0.0);
  for (unsigned int i = 0; i < D; ++i) EXPECT_EQ(later.r[i], first.r[i]);
}

/// Below the decoherence temperature a jet is outside the medium: it stops where it
/// is, keeps its energy, and records the time.
TYPED_TEST(Jets, a_jet_in_a_cold_medium_stops){
  constexpr unsigned int D = TypeParam::D;
  auto s = jet_settings();
  auto sys = medium<D>(s, COLD);
  cc::BBMG<D> bbmg(s, sys);
  bbmg.initial();
  bbmg.propagate();
  bbmg.copy_device_to_host_BBMG();
  const auto& jet = bbmg.jetInfo_host[0];
  EXPECT_EQ(jet.tau_stop, TAU);
  EXPECT_EQ(jet.total_energy, jet.E0);
  EXPECT_EQ(jet.e_loss, 0.0);
  for (unsigned int i = 0; i < D; ++i) EXPECT_EQ(jet.r[i], 0.0);
}

/// File mode: x y eta pT phi rapidity [pdg] per line, '#' lines skipped. Without
/// automatic partners every line is one jet; with them, a back-to-back pair.
TYPED_TEST(Jets, jets_from_a_file){
  constexpr unsigned int D = TypeParam::D;
  {
    std::ofstream f("jets_input.dat");
    f << "# x y eta pT phi rapidity pdg\n0.25 -0.25 0 50 1.0 0.1 21\n\n-0.25 0.25 0 30 2.0 -0.1\n";
  }
  auto s = jet_settings();
  s->jets_input_mode = "file";
  s->jets_input_file = "jets_input.dat";
  s->jets_auto_partner = false;
  auto sys = medium<D>(s, HOT);
  cc::BBMG<D> single(s, sys);
  single.initial();
  ASSERT_EQ(single.jetInfo_host.size(), 2u);
  const auto& a = single.jetInfo_host[0];
  const auto& b = single.jetInfo_host[1];
  EXPECT_EQ(a.flavor, 21);
  EXPECT_EQ(b.flavor, 0);
  EXPECT_EQ(a.PID, 0);
  EXPECT_EQ(b.PID, 1);
  EXPECT_DOUBLE_EQ(a.r[0], 0.25);
  EXPECT_DOUBLE_EQ(b.phi, 2.0);
  EXPECT_DOUBLE_EQ(a.E0, 50.0*std::cosh((D == 3) ? 0.1 : 0.0));
  EXPECT_DOUBLE_EQ(b.E0, 30.0*std::cosh((D == 3) ? -0.1 : 0.0));

  s->jets_auto_partner = true;
  cc::BBMG<D> paired(s, sys);
  paired.initial();
  ASSERT_EQ(paired.jetInfo_host.size(), 4u);
  EXPECT_DOUBLE_EQ(paired.jetInfo_host[1].phi, 1.0 + CODE_PI);
  EXPECT_EQ(paired.jetInfo_host[3].PID, 3);
}

TYPED_TEST(Jets, bad_input_stops_the_run){
  constexpr unsigned int D = TypeParam::D;
  auto s = jet_settings();
  auto sys = medium<D>(s, HOT);
  s->jets_input_mode = "somewhere";
  EXPECT_DEATH({ cc::BBMG<D> b(s, sys); b.initial(); }, "unknown jets_input_mode");
  s->jets_input_mode = "file";
  s->jets_input_file = "no_such_file.dat";
  EXPECT_DEATH({ cc::BBMG<D> b(s, sys); b.initial(); }, "cannot open");
}

/// jet_final.dat: one row per parton, the last column its status: 0 still inside the
/// medium, 1 exited (stopped at T <= 150 MeV), 2 absorbed.
TYPED_TEST(Jets, final_file_gives_each_jet_its_status){
  constexpr unsigned int D = TypeParam::D;
  auto s = jet_settings();
  auto sys = medium<D>(s, HOT);
  auto ws = std::make_shared<cc::SPHWorkstation<D, cc::EoM_default>>(s, sys);
  ws->initialize_jets_bbmg();
  ws->bbmg.jetInfo_host.push_back(ws->bbmg.jetInfo_host[0]);
  ws->bbmg.jetInfo_host[1].tau_stop = 0.8;       // exited
  ws->bbmg.jetInfo_host[2].total_energy = 0.0;   // absorbed
  cc::Output<D, cc::EoM_default> out(s, sys, ws);
  out.print_jet_final_to_txt("jet_final.dat");

  std::ifstream in("jet_final.dat");
  std::vector<std::vector<std::string>> rows;
  for (std::string line; std::getline(in, line); ){
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    std::vector<std::string> tokens;
    for (std::string t; row >> t; ) tokens.push_back(t);
    rows.push_back(tokens);
  }
  ASSERT_EQ(rows.size(), 3u);
  for (const auto& r : rows) ASSERT_EQ(r.size(), 16u);
  EXPECT_EQ(rows[0][15], "0");
  EXPECT_EQ(rows[1][15], "1");
  EXPECT_EQ(rows[2][15], "2");
  // E0, printed with 12 significant digits
  const double E0 = 100.0*std::cosh((D == 3) ? 0.2 : 0.0);
  EXPECT_NEAR(std::stod(rows[0][2]), E0, 1e-11*E0);
}

/// With output.jet_evolution the snapshot step also writes jet_state_<n>.dat, one row per jet.
TYPED_TEST(Jets, jet_state_files_follow_the_snapshots){
  constexpr unsigned int D = TypeParam::D;
  auto s = jet_settings();
  s->txt_evolution = false;
  s->hdf_evolution = false;
  s->jet_evolution = true;
  auto sys = medium<D>(s, HOT);
  auto ws = std::make_shared<cc::SPHWorkstation<D, cc::EoM_default>>(s, sys);
  ws->initialize_jets_bbmg();
  cc::Output<D, cc::EoM_default> out(s, sys, ws);
  out.print_system_state();
  std::ifstream in("jet_state_0.dat");
  ASSERT_TRUE(in.is_open());
  int rows = 0;
  for (std::string line; std::getline(in, line); ) if (!line.empty()) ++rows;
  EXPECT_EQ(rows, 2);
}

int main(int argc, char* argv[]){
  Kokkos::ScopeGuard guard(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
