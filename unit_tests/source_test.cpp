// Source terms (src/source.cpp): what a source releases must reach the fluid.
//
// Every case runs in 3+1D and in 2+1D (CTest names end in <In3D> and <In2D>),
// in Milne coordinates, on a lattice of SPH particles at rest with the grid,
// smoothing lengths and smearing radii of the production configurations:
//   3+1D  as Au+Au 7.7 GeV: dx = dy = 0.2 fm, d(eta_s) = 0.25, h_T = h_eta = 0.3,
//         smearing radii 1.4 fm and 1.0 in eta_s, tau = 1.8 fm/c;
//   2+1D  as Pb+Pb 5.02 TeV: dx = dy = 0.1 fm, h_T = 0.3, smearing radius 0.5 fm,
//         tau = 0.6 fm/c.
// Every particle has SPH masses 1, and its reference density sigma_lab comes
// from the code's own smoothing, as in a run: about one over the cell volume
// inside the lattice and less near its edges.
//
// A source is an AMPT parton. Its budget is its Milne energy p^tau at formation,
// released over four steps, with the Milne velocity p^i/p^tau at the current tau.
// In a step, particle a receives j_a * (tau m_a / sigma_lab_a) * dt of each
// current j (the deposit carries the 1/tau Jacobian). With kernel_renormalize the
// deposit is divided by the discrete kernel sum, so the sums over particles
// equal what the source released, to round-off. Without it they fall short by
// that sum, which is reported, not judged.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constants.h"
#include "eom_default.h"
#include "eos.h"
#include "settings.h"
#include "source.h"
#include "sph_workstation.h"
#include "system_state.h"

namespace cc = ccake;

// The dimensions, 3+1D first. CTest names end in <In3D> or <In2D>.
struct In3D { static constexpr unsigned int D = 3; };
struct In2D { static constexpr unsigned int D = 2; };
using Dimensions = ::testing::Types<In3D, In2D>;
template <typename T> class SourceTest : public ::testing::Test {};
TYPED_TEST_SUITE(SourceTest, Dimensions);

namespace {

constexpr double TOL = 1.e-12;
constexpr double dt = 0.05;
constexpr double normalization = 1.3;
constexpr int deposit_steps = 4;

// Grid and smearing of the production configuration in each dimension.
template <unsigned int D> struct Setup;

template <> struct Setup<3> {   // Au+Au 7.7 GeV, 3+1D
    static constexpr double spacing[3]    = {0.2, 0.2, 0.25};
    static constexpr double half_width[3] = {3.2, 3.2, 2.25};   // x, y in fm; eta_s
    static constexpr double h_T = 0.3, h_eta = 0.3;
    static constexpr double smearing_radius = 1.4, smearing_radius_eta = 1.0;
    static constexpr double tau = 1.8, tau_form = 1.75;
    static constexpr double eta_s = 0.3;   // sources off midrapidity, so eta counts
};

template <> struct Setup<2> {   // Pb+Pb 5.02 TeV, 2+1D (eta_s = 0)
    static constexpr double spacing[2]    = {0.1, 0.1};
    static constexpr double half_width[2] = {1.2, 1.2};
    static constexpr double h_T = 0.3, h_eta = 0.3;
    static constexpr double smearing_radius = 0.5, smearing_radius_eta = 0.5;
    static constexpr double tau = 0.6, tau_form = 0.5;
    static constexpr double eta_s = 0.0;
};

// An AMPT parton: PDG code, momentum and mass in GeV, position (x, y) in fm and
// space-time rapidity eta_s, formed at the proper time of the setup.
struct Parton { int pid; double px, py, pz, mass, x, y, eta_s; };

template <unsigned int D>
std::shared_ptr<Settings> source_settings(const std::string& file, bool renormalize){
    using S = Setup<D>;
    auto settingsPtr = std::make_shared<Settings>();
    settingsPtr->coordinate_system = "hyperbolic";
    settingsPtr->dt = dt;
    settingsPtr->t0 = 0.;
    settingsPtr->results_directory = ".";
    settingsPtr->source_type = "ic";
    settingsPtr->source_model = "AMPT";
    settingsPtr->source_input_file = file;
    settingsPtr->source_normalization = normalization;
    settingsPtr->source_test_particles = 1.;
    settingsPtr->baryon_source = true;
    settingsPtr->strangeness_source = true;
    settingsPtr->electric_source = true;
    settingsPtr->smearing_radius = S::smearing_radius;
    settingsPtr->smearing_radius_eta = S::smearing_radius_eta;
    settingsPtr->source_deposit_steps = deposit_steps;
    settingsPtr->source_propagate = false;
    settingsPtr->source_deposit_at_rapidity = false;
    settingsPtr->source_kernel_renormalize = renormalize;
    settingsPtr->hT = S::h_T;
    settingsPtr->hEta = S::h_eta;
    settingsPtr->xmin = -S::half_width[0] - 1.;   // lower corner of the neighbour-search grid
    settingsPtr->ymin = -S::half_width[1] - 1.;
    settingsPtr->etamin = (D == 3) ? -S::half_width[D-1] - 1. : -1.;
    return settingsPtr;
}

// AMPT parton file: a header line, then pid px py pz mass x y z t (+ 3 more
// columns), with z = tau sinh(eta_s), t = tau cosh(eta_s) at formation.
template <unsigned int D>
void write_partons(const std::string& file, const std::vector<Parton>& partons){
    const double tau = Setup<D>::tau_form;
    std::ofstream out(file);
    out.precision(17);
    out << "# test partons\n";
    for (const Parton& p : partons)
        out << p.pid << " " << p.px << " " << p.py << " " << p.pz << " " << p.mass << " "
            << p.x << " " << p.y << " " << tau*std::sinh(p.eta_s) << " "
            << tau*std::cosh(p.eta_s) << " 0 0 0\n";
}

// The lattice of particles at rest, with sigma_lab from the code's smoothing.
template <unsigned int D>
std::shared_ptr<cc::SystemState<D>> lattice(std::shared_ptr<Settings> settingsPtr){
    using S = Setup<D>;
    auto sysPtr = std::make_shared<cc::SystemState<D>>(settingsPtr);
    int n[D], total = 1;
    for (unsigned d = 0; d < D; ++d){
        n[d] = (int)std::lround(2.*S::half_width[d]/S::spacing[d]) + 1;
        total *= n[d];
    }
    sysPtr->particles.resize(total);
    for (int index = 0; index < total; ++index){
        cc::Particle<D>& p = sysPtr->particles[index];
        int rest = index;
        for (int d = D - 1; d >= 0; --d){
            p.r(d) = -S::half_width[d] + (rest % n[d])*S::spacing[d];
            rest /= n[d];
        }
        p.ID = index;
        p.sph_mass.s = p.sph_mass.rhoB = p.sph_mass.rhoS = p.sph_mass.rhoQ = 1.;
        p.hydro.gamma = 1.;
        p.Freeze = 0;
    }
    sysPtr->n_particles = total;
    sysPtr->t = S::tau;
    sysPtr->allocate_cabana_particles();
    // sigma_lab = sum_b m_b W(r_a - r_b), by the code's own smoothing
    cc::SPHWorkstation<D, cc::EoM_default> workstation(settingsPtr, sysPtr);
    sysPtr->reset_neighbour_list();
    workstation.smooth_all_particle_fields(S::tau*S::tau);
    return sysPtr;
}

// Particle nearest to a point of the lattice (for reporting).
template <unsigned int D>
const cc::Particle<D>& nearest(const cc::SystemState<D>& system, const double point[D]){
    std::size_t best = 0;
    double best_distance = 1.e300;
    for (std::size_t i = 0; i < system.particles.size(); ++i){
        double distance = 0.;
        for (unsigned d = 0; d < D; ++d){
            const double x = system.particles[i].r(d) - point[d];
            distance += x*x;
        }
        if (distance < best_distance){ best_distance = distance; best = i; }
    }
    return system.particles[best];
}

// Energy, momentum (Milne components) and charges that reached the fluid in the
// last deposit.
struct Received { double E = 0., px = 0., py = 0., peta = 0., B = 0., S = 0., Q = 0.; };

template <unsigned int D>
Received received(std::shared_ptr<cc::SystemState<D>> sysPtr){
    sysPtr->copy_device_to_host();
    const double tau = sysPtr->t;
    Received r;
    for (const auto& p : sysPtr->particles){
        const double volume_dt = tau*p.sph_mass.s/p.hydro.sigma_lab*dt;
        r.E  += p.hydro.j0_ext*volume_dt;
        r.px += p.hydro.j_ext(0)*volume_dt;
        r.py += p.hydro.j_ext(1)*volume_dt;
        if (D == 3) r.peta += p.hydro.j_ext(D-1)*volume_dt;
        r.B  += p.hydro.rho_B_ext*volume_dt;
        r.S  += p.hydro.rho_S_ext*volume_dt;
        r.Q  += p.hydro.rho_Q_ext*volume_dt;
    }
    return r;
}

// What a parton should release: its Milne energy p^tau at formation (fm^-1) and
// its Milne velocity p^i/p^tau at proper time tau (eta component in fm^-1).
struct Release { double E0, vx, vy, veta; };

Release release(const Parton& p, double tau){
    const double E = std::sqrt(p.px*p.px + p.py*p.py + p.pz*p.pz + p.mass*p.mass);  // GeV
    const double ch = std::cosh(p.eta_s), sh = std::sinh(p.eta_s);
    const double p_tau = E*ch - p.pz*sh;
    const double p_eta = (p.pz*ch - E*sh)/tau;
    return {p_tau/constants::hbarc_GeVfm, p.px/p_tau, p.py/p_tau, p_eta/p_tau};
}

::testing::AssertionResult close(double value, double expected){
    if (std::fabs(value - expected) <= TOL*(1. + std::fabs(expected)))
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << value << " instead of " << expected;
}


// An up quark inside the fluid, off midrapidity in 3+1D.
template <unsigned int D>
Parton up_quark(){ return {2, 0.3, -0.2, 0.4, 0.006, 0.1, -0.2, Setup<D>::eta_s}; }

} // namespace

/// An up quark inside the fluid: each of the four steps delivers a quarter of its
/// Milne energy (times the normalization) with the momentum of a free particle,
/// and the charges arrive once, in the step that empties the budget.
TYPED_TEST(SourceTest, renormalized_deposit_delivers_exactly_what_the_source_releases){
    constexpr unsigned int D = TypeParam::D;
    const Parton u = up_quark<D>();
    write_partons<D>("partons.dat", {u});
    auto settingsPtr = source_settings<D>("partons.dat", true);
    auto sysPtr = lattice<D>(settingsPtr);
    cc::Source<D> source(settingsPtr, sysPtr);

    const Release x = release(u, sysPtr->t);
    const double chunk = normalization*x.E0/deposit_steps;
    double total = 0.;
    for (int step = 1; step <= deposit_steps; ++step){
        source.advance_sources();
        source.deposit_sources();
        const Received r = received(sysPtr);
        EXPECT_TRUE(close(r.E, chunk)) << "energy in step " << step;
        EXPECT_TRUE(close(r.px, chunk*x.vx)) << "x momentum in step " << step;
        EXPECT_TRUE(close(r.py, chunk*x.vy)) << "y momentum in step " << step;
        if (D == 3) EXPECT_TRUE(close(r.peta, chunk*x.veta)) << "eta momentum in step " << step;
        const bool last = (step == deposit_steps);
        EXPECT_TRUE(close(r.B, last ? 1./3. : 0.)) << "baryon number in step " << step;
        EXPECT_TRUE(close(r.Q, last ? 2./3. : 0.)) << "electric charge in step " << step;
        EXPECT_TRUE(close(r.S, 0.)) << "strangeness in step " << step;
        total += r.E;
    }
    EXPECT_TRUE(close(total, normalization*x.E0));

    // the budget is spent: nothing more arrives
    source.advance_sources();
    source.deposit_sources();
    const Received after = received(sysPtr);
    EXPECT_EQ(after.E, 0.);
    EXPECT_EQ(after.B, 0.);
}

/// An up quark, a strange quark and an anti-up quark at different places: the
/// fluid receives the sum of their energies and of their charges
/// (B = 1/3 + 1/3 - 1/3, S = -1, Q = 2/3 - 1/3 - 2/3).
TYPED_TEST(SourceTest, charges_of_several_partons_add_up){
    constexpr unsigned int D = TypeParam::D;
    const double eta = Setup<D>::eta_s;
    const std::vector<Parton> partons = {{ 2, 0.3, -0.2, 0.4, 0.006,  0.1, -0.2,  eta},
                                         { 3, 0.0,  0.4, 0.1, 0.199, -0.3,  0.2, -eta},
                                         {-2, 0.1,  0.1, 0.0, 0.006,  0.4,  0.3,  0.}};
    write_partons<D>("partons.dat", partons);
    auto settingsPtr = source_settings<D>("partons.dat", true);
    auto sysPtr = lattice<D>(settingsPtr);
    cc::Source<D> source(settingsPtr, sysPtr);

    double E = 0.;
    for (const Parton& p : partons) E += release(p, sysPtr->t).E0;
    Received sum;
    for (int step = 1; step <= deposit_steps; ++step){
        source.advance_sources();
        source.deposit_sources();
        const Received r = received(sysPtr);
        sum.E += r.E; sum.B += r.B; sum.S += r.S; sum.Q += r.Q;
    }
    EXPECT_TRUE(close(sum.E, normalization*E));
    EXPECT_TRUE(close(sum.B, 1./3.));
    EXPECT_TRUE(close(sum.S, -1.));
    EXPECT_TRUE(close(sum.Q, -1./3.));
}

/// A source whose kernel does not reach the fluid keeps its budget and waits.
TYPED_TEST(SourceTest, source_outside_the_fluid_waits){
    constexpr unsigned int D = TypeParam::D;
    using S = Setup<D>;
    const double far = S::half_width[0] + 2.*S::smearing_radius + 1.;   // beyond the kernel's reach
    const Parton outside{2, 0.3, -0.2, 0.0, 0.006, far, far, 0.};
    write_partons<D>("partons.dat", {outside});
    auto settingsPtr = source_settings<D>("partons.dat", true);
    auto sysPtr = lattice<D>(settingsPtr);
    cc::Source<D> source(settingsPtr, sysPtr);
    for (int step = 1; step <= deposit_steps + 2; ++step){
        source.advance_sources();
        source.deposit_sources();
        const Received r = received(sysPtr);
        EXPECT_EQ(r.E, 0.) << "step " << step;
        EXPECT_EQ(r.B, 0.) << "step " << step;
    }
}

/// Without kernel_renormalize the deposit is not divided by the discrete kernel
/// sum. The fraction that reaches the fluid is printed for a source inside the
/// lattice and one at its corner; it is not judged.
TYPED_TEST(SourceTest, unrenormalized_deposit_fraction_is_reported){
    constexpr unsigned int D = TypeParam::D;
    using S = Setup<D>;
    const double eta_corner = (D == 3) ? S::half_width[D-1] : 0.;
    for (const Parton& p : {up_quark<D>(),
                            Parton{2, 0.3, -0.2, 0.4, 0.006, S::half_width[0], S::half_width[1], eta_corner}}){
        write_partons<D>("partons.dat", {p});
        auto settingsPtr = source_settings<D>("partons.dat", false);
        auto sysPtr = lattice<D>(settingsPtr);
        cc::Source<D> source(settingsPtr, sysPtr);
        double delivered = 0., baryon = 0.;
        for (int step = 1; step <= deposit_steps; ++step){
            source.advance_sources();
            source.deposit_sources();
            const Received r = received(sysPtr);
            delivered += r.E;
            baryon += r.B;
        }
        const double fraction = delivered/(normalization*release(p, sysPtr->t).E0);
        const bool corner = (p.x == S::half_width[0]);
        const std::string where = corner ? "corner" : "inside";
        double point[D];
        point[0] = p.x; point[1] = p.y;
        if (D == 3) point[D-1] = p.eta_s;
        double cell = 1.;
        for (unsigned d = 0; d < D; ++d) cell *= S::spacing[d];
        const double sigma_cell = nearest<D>(*sysPtr, point).hydro.sigma_lab*cell;
        std::printf("[ MEASURED ] %uD, unrenormalized source %s the lattice: %.6f of the energy and "
                    "%.6f of the baryon number reach the fluid (sigma_lab x cell volume = %.4f there)\n",
                    D, corner ? "at the corner of" : "inside", fraction, baryon/(1./3.), sigma_cell);
        this->RecordProperty("delivered_fraction_" + where, std::to_string(fraction));
        EXPECT_GT(fraction, 0.);
        EXPECT_TRUE(std::isfinite(fraction));
    }
}

/// Without kernel_renormalize the charges also arrive with the last of the
/// deposit_steps chunks: the budget, reduced by E0/deposit_steps each step, counts
/// as spent even when round-off leaves ~1e-16 of it.
TYPED_TEST(SourceTest, unrenormalized_charges_arrive_with_the_last_chunk){
    constexpr unsigned int D = TypeParam::D;
    using S = Setup<D>;
    const double eta_corner = (D == 3) ? S::half_width[D-1] : 0.;
    for (const Parton& p : {up_quark<D>(),
                            Parton{2, 0.3, -0.2, 0.4, 0.006, S::half_width[0], S::half_width[1], eta_corner}}){
        write_partons<D>("partons.dat", {p});
        auto settingsPtr = source_settings<D>("partons.dat", false);
        auto sysPtr = lattice<D>(settingsPtr);
        cc::Source<D> source(settingsPtr, sysPtr);
        const std::string where = (p.x == S::half_width[0]) ? " (corner)" : " (inside)";
        for (int step = 1; step <= deposit_steps + 1; ++step){
            source.advance_sources();
            source.deposit_sources();
            const Received r = received(sysPtr);
            if (step < deposit_steps) EXPECT_EQ(r.B, 0.) << "step " << step << where;
            if (step == deposit_steps) EXPECT_GT(r.B, 0.) << "step " << step << where;
            if (step > deposit_steps){
                EXPECT_EQ(r.E, 0.) << "step " << step << where;
                EXPECT_EQ(r.B, 0.) << "step " << step << where;
            }
        }
    }
}

/// From the deposit to the equations of motion. The source terms enter the time
/// derivatives through j0_ext, j_ext and rho_ext. For a fluid at rest (uniform,
/// no viscosity or diffusion) in a charged state, each particle gains per unit time
///   energy   T dS/dt + sum_a mu_a dN_a/dt,   S = m_s (s/sigma), N_a = m_a (n_a/sigma)
///   momentum (m_s/sigma) w du^i/dt          (Milne components)
///   charge   dN_a/dt
/// with sigma = sigma_lab/(gamma tau), and, summed over the particles, these must
/// be what the source releases in the step. The entropy equation has to take the
/// chemical work mu_a dN_a out of the deposited energy (first law) for the energy
/// to come out right.
TYPED_TEST(SourceTest, deposit_reaches_the_equations_of_motion){
    constexpr unsigned int D = TypeParam::D;
    const Parton u = up_quark<D>();
    write_partons<D>("partons.dat", {u});
    auto settingsPtr = source_settings<D>("partons.dat", true);
    settingsPtr->using_shear = false;
    settingsPtr->using_diffusion = false;
    auto sysPtr = lattice<D>(settingsPtr);

    // a charged state from the conformal-diagonal equation of state (fm^-1)
    auto eos_settings = std::make_shared<Settings>();
    eos_settings->eos_type = "conformal_diagonal";
    cc::EquationOfState eos;
    eos.set_SettingsPtr(eos_settings);
    eos.init();
    eos.tbqs(1.2, 0.3, -0.05, 0.1, "conformal_diagonal");   // T, mu_B, mu_Q, mu_S
    sysPtr->copy_device_to_host();
    for (auto& p : sysPtr->particles){
        eos.set_thermo(p.thermo);
        p.hydro.tau_Pi = 1.;   // divides the (vanishing) bulk terms
    }
    sysPtr->copy_host_to_device();
    ASSERT_NE(sysPtr->particles[0].thermo.muB, 0.);

    cc::Source<D> source(settingsPtr, sysPtr);
    const Release x = release(u, sysPtr->t);
    const double chunk = normalization*x.E0/deposit_steps;
    for (int step = 1; step <= deposit_steps; ++step){
        source.advance_sources();
        cc::EoM_default<D>::update_velocity(sysPtr);
        cc::EoM_default<D>::reset_pi_tensor(sysPtr);
        source.deposit_sources();
        cc::EoM_default<D>::evaluate_time_derivatives(sysPtr, settingsPtr);
        sysPtr->copy_device_to_host();

        Received gained;
        for (const auto& p : sysPtr->particles){
            const double dS = p.sph_mass.s*p.d_dt_extensive.s*dt;
            const double dB = p.sph_mass.rhoB*p.d_dt_extensive.rhoB*dt;
            const double dS_charge = p.sph_mass.rhoS*p.d_dt_extensive.rhoS*dt;
            const double dQ = p.sph_mass.rhoQ*p.d_dt_extensive.rhoQ*dt;
            const double volume = p.sph_mass.s/p.hydro.sigma;
            gained.E  += p.thermo.T*dS + p.thermo.muB*dB + p.thermo.muS*dS_charge + p.thermo.muQ*dQ;
            gained.px += volume*p.thermo.w*p.hydro.du_dt(0)*dt;
            gained.py += volume*p.thermo.w*p.hydro.du_dt(1)*dt;
            if (D == 3) gained.peta += volume*p.thermo.w*p.hydro.du_dt(D-1)*dt;
            gained.B  += dB;
            gained.S  += dS_charge;
            gained.Q  += dQ;
        }
        EXPECT_TRUE(close(gained.E, chunk)) << "energy in step " << step;
        EXPECT_TRUE(close(gained.px, chunk*x.vx)) << "x momentum in step " << step;
        EXPECT_TRUE(close(gained.py, chunk*x.vy)) << "y momentum in step " << step;
        if (D == 3) EXPECT_TRUE(close(gained.peta, chunk*x.veta)) << "eta momentum in step " << step;
        const bool last = (step == deposit_steps);
        EXPECT_TRUE(close(gained.B, last ? 1./3. : 0.)) << "baryon number in step " << step;
        EXPECT_TRUE(close(gained.Q, last ? 2./3. : 0.)) << "electric charge in step " << step;
        EXPECT_TRUE(close(gained.S, 0.)) << "strangeness in step " << step;
    }
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
