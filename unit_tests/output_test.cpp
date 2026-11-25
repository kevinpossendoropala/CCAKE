// Column layout of the system_state_<n>.dat snapshots (Output::print_system_state).
//
// Analysis scripts read these files by column number, so the layout is part of
// the interface. One particle gets a different value in every printed field, on
// the device (where the run computes them); the snapshot is written and read
// back, and each column must hold its field. Pressure, temperature, chemical
// potentials and energy density are printed in GeV units (times hbar c).
#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constants.h"
#include "eom_default.h"
#include "output.h"
#include "settings.h"
#include "sph_workstation.h"
#include "system_state.h"

namespace cc = ccake;

namespace {

const double hbarc = constants::hbarc_GeVfm;

// The value written into each field, and the column it must appear in.
struct Column { int index; double value; };

template <unsigned int D>
std::vector<std::string> write_and_read_snapshot(const double r[3], const double u[3]){
    auto settingsPtr = std::make_shared<Settings>();
    settingsPtr->results_directory = ".";
    settingsPtr->txt_evolution = true;
    settingsPtr->hdf_evolution = false;
    settingsPtr->jet_evolution = false;
    settingsPtr->hT = 1.;
    auto sysPtr = std::make_shared<cc::SystemState<D>>(settingsPtr);
    sysPtr->particles.resize(1);
    sysPtr->particles[0].ID = 0;
    sysPtr->particles[0].thermo.eos_name = "conformal";   // kept on the host only
    sysPtr->n_particles = 1;
    sysPtr->t = 2.;
    sysPtr->allocate_cabana_particles();

    CREATE_VIEW(device_, sysPtr->cabana_particles);
    auto policy = Cabana::SimdPolicy<VECTOR_LENGTH, ExecutionSpace>(0, 1);
    const double r0 = r[0], r1 = r[1], r2 = r[2], u0 = u[0], u1 = u[1], u2 = u[2];
    Cabana::simd_parallel_for(policy, KOKKOS_LAMBDA(const int is, const int ia){
        const double rr[3] = {r0, r1, r2}, uu[3] = {u0, u1, u2};
        for (int i = 0; i < (int)D; ++i){
            device_position.access(is, ia, i) = rr[i];
            device_hydro_vector.access(is, ia, cc::hydro_info::u, i) = uu[i];
        }
        device_thermo.access(is, ia, cc::thermo_info::p)    = 0.11;
        device_thermo.access(is, ia, cc::thermo_info::T)    = 0.12;
        device_thermo.access(is, ia, cc::thermo_info::muB)  = 0.13;
        device_thermo.access(is, ia, cc::thermo_info::muS)  = 0.14;
        device_thermo.access(is, ia, cc::thermo_info::muQ)  = 0.15;
        device_thermo.access(is, ia, cc::thermo_info::e)    = 0.16;
        device_thermo.access(is, ia, cc::thermo_info::rhoB) = 0.17;
        device_thermo.access(is, ia, cc::thermo_info::rhoS) = 0.18;
        device_thermo.access(is, ia, cc::thermo_info::rhoQ) = 0.19;
        device_thermo.access(is, ia, cc::thermo_info::s)    = 0.21;
        device_hydro_scalar.access(is, ia, cc::hydro_info::eta_pi) = 0.22;
        device_hydro_scalar.access(is, ia, cc::hydro_info::zeta_Pi) = 0.23;
        device_hydro_scalar.access(is, ia, cc::hydro_info::tau_Pi) = 0.24;
        device_hydro_scalar.access(is, ia, cc::hydro_info::tau_pi) = 0.25;
        device_hydro_scalar.access(is, ia, cc::hydro_info::theta) = 0.26;
        device_hydro_scalar.access(is, ia, cc::hydro_info::inverse_reynolds_shear) = 0.27;
        device_hydro_scalar.access(is, ia, cc::hydro_info::inverse_reynolds_bulk) = 0.28;
        device_hydro_scalar.access(is, ia, cc::hydro_info::shear_knudsen) = 0.29;
        device_hydro_scalar.access(is, ia, cc::hydro_info::bulk_knudsen) = 0.31;
        const int shv_index[9][2] = {{0,0},{1,1},{2,2},{0,1},{0,2},{1,2},{1,3},{2,3},{3,3}};
        for (int k = 0; k < 9; ++k)
            device_hydro_spacetime_matrix.access(is, ia, cc::hydro_info::shv,
                                                 shv_index[k][0], shv_index[k][1]) = 0.32 + 0.01*k;
        device_hydro_scalar.access(is, ia, cc::hydro_info::gamma) = 1.44;
        device_freeze.access(is, ia) = 2;
        device_hydro_scalar.access(is, ia, cc::hydro_info::causality) = 1.;
        device_hydro_scalar.access(is, ia, cc::hydro_info::knudsen_diffusion_B) = 0.45;
        device_hydro_scalar.access(is, ia, cc::hydro_info::knudsen_diffusion_S) = 0.46;
        device_hydro_scalar.access(is, ia, cc::hydro_info::knudsen_diffusion_Q) = 0.47;
        device_hydro_scalar.access(is, ia, cc::hydro_info::inverse_reynolds_diffusion_B) = 0.48;
        device_hydro_scalar.access(is, ia, cc::hydro_info::inverse_reynolds_diffusion_S) = 0.49;
        device_hydro_scalar.access(is, ia, cc::hydro_info::inverse_reynolds_diffusion_Q) = 0.51;
    }, "fill_fields");
    Kokkos::fence();

    auto wsPtr = std::make_shared<cc::SPHWorkstation<D, cc::EoM_default>>(settingsPtr, sysPtr);
    cc::Output<D, cc::EoM_default> output(settingsPtr, sysPtr, wsPtr);
    output.print_system_state();

    std::ifstream in("system_state_0.dat");
    std::string first, second;
    std::getline(in, first);
    std::getline(in, second);
    EXPECT_EQ(std::stod(first), 2.) << "the first line is the time";
    std::istringstream row(second);
    std::vector<std::string> tokens;
    for (std::string token; row >> token; ) tokens.push_back(token);
    return tokens;
}

void expect_columns(const std::vector<std::string>& tokens, const std::vector<Column>& columns){
    for (const Column& c : columns){
        ASSERT_LT(c.index, (int)tokens.size());
        const double value = std::stod(tokens[c.index]);
        EXPECT_NEAR(value, c.value, 1.e-6*std::max(1., std::fabs(c.value)))
            << "column " << c.index << " holds " << tokens[c.index];
    }
}

// The columns that do not depend on the dimension.
const std::vector<Column> common_columns = {
    {0, 0.}, {1, 2.},
    {5, 0.11*hbarc}, {6, 0.12*hbarc}, {7, 0.13*hbarc}, {8, 0.14*hbarc}, {9, 0.15*hbarc},
    {10, 0.16*hbarc}, {11, 0.17}, {12, 0.18}, {13, 0.19}, {14, 0.21},
    {15, 0.22}, {16, 0.23}, {17, 0.24}, {18, 0.25}, {19, 0.26},
    {20, 0.27}, {21, 0.28}, {22, 0.29}, {23, 0.31},
    {24, 0.32}, {25, 0.33}, {26, 0.34}, {27, 0.35}, {28, 0.36}, {29, 0.37},
    {30, 0.38}, {31, 0.39}, {32, 0.40},
    {36, 1.44}, {37, 2.}, {39, 1.},
};

// Diffusion Knudsen and inverse Reynolds numbers.
const std::vector<Column> diffusion_columns = {
    {40, 0.45}, {41, 0.46}, {42, 0.47}, {43, 0.48}, {44, 0.49}, {45, 0.51},
};

} // namespace

/// 2+1D: x, y in columns 2 and 3, u^x, u^y in 33 and 34; 46 columns in all.
TEST(OutputLayout, system_state_2d){
    const double r[3] = {0.25, -0.5, 0.}, u[3] = {0.42, 0.43, 0.};
    const std::vector<std::string> tokens = write_and_read_snapshot<2>(r, u);
    EXPECT_EQ(tokens.size(), 46u);
    expect_columns(tokens, common_columns);
    expect_columns(tokens, {{2, 0.25}, {3, -0.5}, {4, 0.}, {33, 0.42}, {34, 0.43}, {35, 0.}});
    ASSERT_GT(tokens.size(), 38u);
    EXPECT_EQ(tokens[38], "conformal") << "column 38 is the name of the equation of state";
}

/// 3+1D: eta_s in column 4 and u^eta in 35.
TEST(OutputLayout, system_state_3d){
    const double r[3] = {0.25, -0.5, 0.7}, u[3] = {0.42, 0.43, 0.05};
    const std::vector<std::string> tokens = write_and_read_snapshot<3>(r, u);
    EXPECT_EQ(tokens.size(), 46u);
    expect_columns(tokens, common_columns);
    expect_columns(tokens, {{2, 0.25}, {3, -0.5}, {4, 0.7}, {33, 0.42}, {34, 0.43}, {35, 0.05}});
}

/// 1+1D: the only coordinate, eta_s, is written in column 2 (not 4), and u^eta
/// in column 33 (not 35). Scripts that read 1+1D and 3+1D snapshots alike must
/// take this into account.
TEST(OutputLayout, system_state_1d){
    const double r[3] = {0.7, 0., 0.}, u[3] = {0.05, 0., 0.};
    const std::vector<std::string> tokens = write_and_read_snapshot<1>(r, u);
    EXPECT_EQ(tokens.size(), 46u);
    expect_columns(tokens, common_columns);
    expect_columns(tokens, {{2, 0.7}, {3, 0.}, {4, 0.}, {33, 0.05}, {34, 0.}, {35, 0.}});
}

/// The diffusion Knudsen and inverse Reynolds numbers (columns 40-45) are
/// computed on the device and must reach the snapshot. (Until 2026-11 they
/// were never copied to the host, and every snapshot printed 0 there.)
TEST(OutputLayout, diffusion_numbers_reach_the_snapshot){
    const double r[3] = {0.25, -0.5, 0.}, u[3] = {0.42, 0.43, 0.};
    const std::vector<std::string> tokens = write_and_read_snapshot<2>(r, u);
    expect_columns(tokens, diffusion_columns);
}

//==============================================================================
// freeze_out.dat (Output::print_freeze_out, flush_freeze_out)
//==============================================================================
// The freeze-out surface is read by column number too: by the particlization code
// and by tests/physics/physics_tools.py. Two particles get freeze-out results on the
// device, every field a different value; only the first is marked for printing.

namespace {

// Field values: 1.01, 1.02, ... in the order the writer prints them in 3+1D.
template <unsigned int D>
void fill_results(cc::FreezeOut<D>& fo){
    FRZ_RESULTS_VIEW(res_, fo.results)
    auto policy = Cabana::SimdPolicy<VECTOR_LENGTH, ExecutionSpace>(0, fo.results.size());
    Cabana::simd_parallel_for(policy, KOKKOS_LAMBDA(const int is, const int ia){
        res_divEener.access(is, ia) = 1.01;
        for (int i = 0; i < 3; ++i){
            res_divE.access(is, ia, i) = 1.02 + 0.01*i;
            res_uout.access(is, ia, i) = 1.06 + 0.01*i;
            res_rsub.access(is, ia, i) = 1.19 + 0.01*i;
        }
        res_gsub.access(is, ia) = 1.05;
        res_swsub.access(is, ia) = 1.09;
        res_bulksub.access(is, ia) = 1.10;
        res_shearsub.access(is, ia, 0, 0) = 1.11;
        res_shearsub.access(is, ia, 1, 1) = 1.12;
        res_shearsub.access(is, ia, 2, 2) = 1.13;
        res_shear33sub.access(is, ia) = 1.14;
        res_shearsub.access(is, ia, 1, 2) = 1.15;
        res_shearsub.access(is, ia, 1, 3) = 1.16;
        res_shearsub.access(is, ia, 2, 3) = 1.17;
        res_tlist.access(is, ia) = 1.18;
        res_sFO.access(is, ia) = 1.22;
        res_Efluc.access(is, ia) = 1.23;
        res_Tfluc.access(is, ia) = 1.24;
        res_muBfluc.access(is, ia) = 1.25;
        res_muSfluc.access(is, ia) = 1.26;
        res_muQfluc.access(is, ia) = 1.27;
        res_wfzfluc.access(is, ia) = 1.28;
        res_cs2fzfluc.access(is, ia) = 1.29;
        res_rhoBfluc.access(is, ia) = 1.30;
        res_rhoSfluc.access(is, ia) = 1.31;
        res_rhoQfluc.access(is, ia) = 1.32;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 4; ++b) res_diffout.access(is, ia, a, b) = 1.33 + 0.01*(4*a + b);
        res_eos_type.access(is, ia) = 1.;
        res_print.access(is, ia) = (is*VECTOR_LENGTH + ia == 0);
    }, "fill_results");
    Kokkos::fence();
}

/// Writes the surface of two particles, one of them marked, and returns the lines.
template <unsigned int D>
std::vector<std::string> write_and_read_surface(){
    auto settingsPtr = std::make_shared<Settings>();
    settingsPtr->results_directory = ".";
    settingsPtr->hT = 1.;
    auto sysPtr = std::make_shared<cc::SystemState<D>>(settingsPtr);
    sysPtr->particles.resize(2);
    sysPtr->particles[0].ID = 0;
    sysPtr->particles[1].ID = 1;
    sysPtr->n_particles = 2;
    sysPtr->t = 2.;
    sysPtr->allocate_cabana_particles();
    auto fo = std::make_shared<cc::FreezeOut<D>>(settingsPtr, sysPtr);
    fill_results(*fo);

    auto wsPtr = std::make_shared<cc::SPHWorkstation<D, cc::EoM_default>>(settingsPtr, sysPtr);
    cc::Output<D, cc::EoM_default> output(settingsPtr, sysPtr, wsPtr);
    output.print_freeze_out(fo, 1.5, -0.25, 0.);   // B, Q, S at the start of the run
    output.flush_freeze_out();

    std::ifstream in("freeze_out.dat");
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line); ) lines.push_back(line);
    return lines;
}

std::vector<std::string> tokens_of(const std::string& line){
    std::istringstream row(line);
    std::vector<std::string> tokens;
    for (std::string token; row >> token; ) tokens.push_back(token);
    return tokens;
}

// The columns after the positions are the same in both layouts, shifted.
std::vector<Column> thermal_columns(int first){
    std::vector<Column> c;
    const double values[] = {1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.30, 1.31, 1.32};
    for (int k = 0; k < 11; ++k) c.push_back({first + k, values[k]});
    for (int k = 0; k < 12; ++k) c.push_back({first + 11 + k, 1.33 + 0.01*k});
    c.push_back({first + 23, 1.});
    return c;
}

} // namespace

/// 3+1D, 45 columns: s T^3 in 21, e 22, T 23, mu_B 24, n_B 29 (tests/physics/physics_tools.py).
/// The header holds B, S, Q at the start of the run; only marked particles are written.
TEST(OutputLayout, freeze_out_3d){
    const auto lines = write_and_read_surface<3>();
    ASSERT_EQ(lines.size(), 2u) << "the header and one element";
    EXPECT_EQ(lines[0], "# 1.5 0 -0.25");
    const auto tokens = tokens_of(lines[1]);
    ASSERT_EQ(tokens.size(), 45u);
    expect_columns(tokens, {
        {0, 1.01}, {1, 1.02}, {2, 1.03}, {3, 1.04},   // d_tau e, d_i e
        {4, 1.05},                                    // gamma
        {5, 1.06}, {6, 1.07}, {7, 1.08},              // u^x, u^y, u^eta
        {8, 1.09},                                    // V = u.dSigma
        {9, 1.10},                                    // bulk
        {10, 1.11}, {11, 1.12}, {12, 1.13}, {13, 1.14}, {14, 1.15}, {15, 1.16}, {16, 1.17},
        {17, 1.18},                                   // tau
        {18, 1.19}, {19, 1.20}, {20, 1.21},           // x, y, eta_s
    });
    expect_columns(tokens, thermal_columns(21));
}

/// 2+1D, 40 columns: V in 6, tau 13, s T^3 in 16, e 17, T 18, mu_B 19, n_B 24.
TEST(OutputLayout, freeze_out_2d){
    const auto lines = write_and_read_surface<2>();
    ASSERT_EQ(lines.size(), 2u) << "the header and one element";
    EXPECT_EQ(lines[0], "# 1.5 0 -0.25");
    const auto tokens = tokens_of(lines[1]);
    ASSERT_EQ(tokens.size(), 40u);
    expect_columns(tokens, {
        {0, 1.01}, {1, 1.02}, {2, 1.03},              // d_tau e, d_x e, d_y e
        {3, 1.05},                                    // gamma
        {4, 1.06}, {5, 1.07},                         // u^x, u^y
        {6, 1.09},                                    // V
        {7, 1.10},                                    // bulk
        {8, 1.11}, {9, 1.12}, {10, 1.13}, {11, 1.14}, {12, 1.15},
        {13, 1.18},                                   // tau
        {14, 1.19}, {15, 1.20},                       // x, y
    });
    expect_columns(tokens, thermal_columns(16));
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
