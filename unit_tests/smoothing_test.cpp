#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <Kokkos_Core.hpp>
//#include <gmock/gmock.h> //To mock settings pointers

#include "kernel_reference.h"
#include "sph_workstation.h"
#include "system_state.h"
#include "settings.h"
#include "particle.h"
#define TOL 1.E-14

//class MockSettings : Settings{
//    MOCK_METHOD(void, check_consistency, (), (override));
//};

TEST(SmoothingTest, Smooth2D){

    // (Kokkos is started once, in main() at the end of this file)
    //Create mocked settings
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings> ();
    settingsPtr->xmin = -4;
    settingsPtr->ymin = -6;
    settingsPtr->etamin = -2;
    settingsPtr->hT = 1;
    std::shared_ptr<ccake::SystemState<2>> sysPtr = std::make_shared<ccake::SystemState<2>> (settingsPtr);
    ccake::SPHWorkstation<2,ccake::EoM_default> ws(settingsPtr, sysPtr);

    //ccake::SystemState<2> sys(settingsPtr);
    // (the smoothing length is read from the settings; SystemState has no hT of its own)

    //Place particles in positions {0,0}, {.5,0}, {.8,.6} and {0,1.5}
    std::array<ccake::Particle<2>,4> particles_array_in;
    particles_array_in[0].r = {0,0};
    particles_array_in[1].r = {0,.5};
    particles_array_in[2].r = {.8,.6};
    particles_array_in[3].r = {1.5,0};
    for (auto p : particles_array_in) sysPtr->add_particle(p);
    sysPtr->n_particles = 4;
    // hydro.sigma is now the rest-frame density sigma_lab/(gamma*tau). With the
    // fluid at rest (gamma = 1) at tau = 1 it equals the smoothed density that
    // this test has always checked.
    sysPtr->t = 1.;
    for (int i=0; i<4; ++i){
        sysPtr->particles[i].hydro.gamma = 1.;
        sysPtr->particles[i].ID = i;
        sysPtr->particles[i].sph_mass.s = 1;
        sysPtr->particles[i].sph_mass.rhoB = 2;
        sysPtr->particles[i].sph_mass.rhoQ = 3;
        sysPtr->particles[i].sph_mass.rhoS = 4;

        sysPtr->particles[i].extensive.s = 1;
        sysPtr->particles[i].extensive.rhoB = 2;
        sysPtr->particles[i].extensive.rhoQ = 3;
        sysPtr->particles[i].extensive.rhoS = 4;
    }
    
    sysPtr->allocate_cabana_particles();
    sysPtr->reset_neighbour_list();    
    ws.smooth_all_particle_fields(1.);
    sysPtr->copy_device_to_host();

    EXPECT_LT( fabs(sysPtr->particles[0].smoothed.s - 20./7./M_PI), TOL) << "Test failed! Reporting values used: " << endl
    << "Particle 1: -------" << endl
    << "r: " << sysPtr->particles[0].r << endl
    << "sph_mass: " << sysPtr->particles[0].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[0].extensive.s << endl
    << "smoothed: " << sysPtr->particles[0].smoothed.s << endl
    << "Particle 2: -------" << endl
    << "r: " << sysPtr->particles[1].r << endl
    << "sph_mass: " << sysPtr->particles[1].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[1].extensive.s << endl
    << "smoothed: " << sysPtr->particles[1].smoothed.s << endl
    << "Particle 3: -------" << endl
    << "r: " << sysPtr->particles[2].r << endl
    << "sph_mass: " << sysPtr->particles[2].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[2].extensive.s << endl
    << "smoothed: " << sysPtr->particles[2].smoothed.s << endl
    << "Particle 4: -------" << endl
    << "r: " << sysPtr->particles[3].r << endl
    << "sph_mass: " << sysPtr->particles[3].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[3].extensive.s << endl
    << "smoothed: " << sysPtr->particles[3].smoothed.s << endl;

    EXPECT_LT( fabs(sysPtr->particles[0].smoothed.rhoB - 4*20./7./M_PI), TOL) << "Test failed! Reporting values used: " << endl
    << "Particle 1: -------" << endl
    << "r: " << sysPtr->particles[0].r << endl
    << "sph_mass: " << sysPtr->particles[0].sph_mass.rhoB << endl
    << "extensive: " << sysPtr->particles[0].extensive.rhoB << endl
    << "smoothed: " << sysPtr->particles[0].smoothed.rhoB << endl
    << "Particle 2: -------" << endl
    << "r: " << sysPtr->particles[1].r << endl
    << "sph_mass: " << sysPtr->particles[1].sph_mass.rhoB << endl
    << "extensive: " << sysPtr->particles[1].extensive.rhoB << endl
    << "smoothed: " << sysPtr->particles[1].smoothed.rhoB << endl
    << "Particle 3: -------" << endl
    << "r: " << sysPtr->particles[2].r << endl
    << "sph_mass: " << sysPtr->particles[2].sph_mass.rhoB << endl
    << "extensive: " << sysPtr->particles[2].extensive.rhoB << endl
    << "smoothed: " << sysPtr->particles[2].smoothed.rhoB << endl
    << "Particle 4: -------" << endl
    << "r: " << sysPtr->particles[3].r << endl
    << "sph_mass: " << sysPtr->particles[3].sph_mass.rhoB << endl
    << "extensive: " << sysPtr->particles[3].extensive.rhoB << endl
    << "smoothed: " << sysPtr->particles[3].smoothed.rhoB << endl;

    EXPECT_LT( fabs(sysPtr->particles[0].smoothed.rhoQ - 9*20./7./M_PI), TOL) << "Test failed! Reporting values used: " << endl
    << "Particle 1: -------" << endl
    << "r: " << sysPtr->particles[0].r << endl
    << "sph_mass: " << sysPtr->particles[0].sph_mass.rhoQ << endl
    << "extensive: " << sysPtr->particles[0].extensive.rhoQ << endl
    << "smoothed: " << sysPtr->particles[0].smoothed.rhoQ << endl
    << "Particle 2: -------" << endl
    << "r: " << sysPtr->particles[1].r << endl
    << "sph_mass: " << sysPtr->particles[1].sph_mass.rhoQ << endl
    << "extensive: " << sysPtr->particles[1].extensive.rhoQ << endl
    << "smoothed: " << sysPtr->particles[1].smoothed.rhoQ << endl
    << "Particle 3: -------" << endl
    << "r: " << sysPtr->particles[2].r << endl
    << "sph_mass: " << sysPtr->particles[2].sph_mass.rhoQ << endl
    << "extensive: " << sysPtr->particles[2].extensive.rhoQ << endl
    << "smoothed: " << sysPtr->particles[2].smoothed.rhoQ << endl
    << "Particle 4: -------" << endl
    << "r: " << sysPtr->particles[3].r << endl
    << "sph_mass: " << sysPtr->particles[3].sph_mass.rhoQ << endl
    << "extensive: " << sysPtr->particles[3].extensive.rhoQ << endl
    << "smoothed: " << sysPtr->particles[3].smoothed.rhoQ << endl;

    EXPECT_LT( fabs(sysPtr->particles[0].smoothed.rhoS - 16*20./7./M_PI), TOL) << "Test failed! Reporting values used: " << endl
    << "Particle 1: -------" << endl
    << "r: " << sysPtr->particles[0].r << endl
    << "sph_mass: " << sysPtr->particles[0].sph_mass.rhoS << endl
    << "extensive: " << sysPtr->particles[0].extensive.rhoS << endl
    << "smoothed: " << sysPtr->particles[0].smoothed.rhoS << endl
    << "Particle 2: -------" << endl
    << "r: " << sysPtr->particles[1].r << endl
    << "sph_mass: " << sysPtr->particles[1].sph_mass.rhoS << endl
    << "extensive: " << sysPtr->particles[1].extensive.rhoS << endl
    << "smoothed: " << sysPtr->particles[1].smoothed.rhoS << endl
    << "Particle 3: -------" << endl
    << "r: " << sysPtr->particles[2].r << endl
    << "sph_mass: " << sysPtr->particles[2].sph_mass.rhoS << endl
    << "extensive: " << sysPtr->particles[2].extensive.rhoS << endl
    << "smoothed: " << sysPtr->particles[2].smoothed.rhoS << endl
    << "Particle 4: -------" << endl
    << "r: " << sysPtr->particles[3].r << endl
    << "sph_mass: " << sysPtr->particles[3].sph_mass.rhoS << endl
    << "extensive: " << sysPtr->particles[3].extensive.rhoS << endl
    << "smoothed: " << sysPtr->particles[3].smoothed.rhoS << endl;

    EXPECT_LT( fabs(sysPtr->particles[0].hydro.sigma - 20./7./M_PI), TOL) << "Test failed! Reporting values used: " << endl
    << "Particle 1: -------" << endl
    << "r: " << sysPtr->particles[0].r << endl
    << "sph_mass: " << sysPtr->particles[0].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[0].extensive.s << endl
    << "smoothed: " << sysPtr->particles[0].smoothed.s << endl
    << "sigma: " << sysPtr->particles[0].hydro.sigma << endl
    << "Particle 2: -------" << endl
    << "r: " << sysPtr->particles[1].r << endl
    << "sph_mass: " << sysPtr->particles[1].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[1].extensive.s << endl
    << "smoothed: " << sysPtr->particles[1].smoothed.s << endl
    << "sigma: " << sysPtr->particles[1].hydro.sigma << endl
    << "Particle 3: -------" << endl
    << "r: " << sysPtr->particles[2].r << endl
    << "sph_mass: " << sysPtr->particles[2].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[2].extensive.s << endl
    << "smoothed: " << sysPtr->particles[2].smoothed.s << endl
    << "sigma: " << sysPtr->particles[2].hydro.sigma << endl
    << "Particle 4: -------" << endl
    << "r: " << sysPtr->particles[3].r << endl
    << "sph_mass: " << sysPtr->particles[3].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[3].extensive.s << endl
    << "smoothed: " << sysPtr->particles[3].smoothed.s << endl
    << "sigma: " << sysPtr->particles[3].hydro.sigma << endl;

}

/// The smoothed (lab-frame) densities do not depend on the flow, but the
/// rest-frame densities the equation of state receives are divided by gamma*tau,
/// and so is sigma. Same four particles as above, now with gamma = 2 at tau = 3.
/// (gamma is set directly; the smoothing does not recompute it from u.)
TEST(SmoothingTest, RestFrameDensitiesDivideByGammaTau){
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings> ();
    settingsPtr->xmin = -4;
    settingsPtr->ymin = -6;
    settingsPtr->etamin = -2;
    settingsPtr->hT = 1;
    std::shared_ptr<ccake::SystemState<2>> sysPtr = std::make_shared<ccake::SystemState<2>> (settingsPtr);
    ccake::SPHWorkstation<2,ccake::EoM_default> ws(settingsPtr, sysPtr);

    std::array<ccake::Particle<2>,4> particles_array_in;
    particles_array_in[0].r = {0,0};
    particles_array_in[1].r = {0,.5};
    particles_array_in[2].r = {.8,.6};
    particles_array_in[3].r = {1.5,0};
    for (auto p : particles_array_in) sysPtr->add_particle(p);
    sysPtr->n_particles = 4;
    // gamma = 2: u = (sqrt(3), 0), from which the code computes gamma, as in a run
    const double gamma = 2., tau = 3.;
    sysPtr->t = tau;
    for (int i=0; i<4; ++i){
        sysPtr->particles[i].hydro.u = {std::sqrt(3.), 0.};
        sysPtr->particles[i].ID = i;
        sysPtr->particles[i].sph_mass.s = 1;
        sysPtr->particles[i].sph_mass.rhoB = 2;
        sysPtr->particles[i].sph_mass.rhoQ = 3;
        sysPtr->particles[i].sph_mass.rhoS = 4;
        sysPtr->particles[i].extensive.s = 1;
        sysPtr->particles[i].extensive.rhoB = 2;
        sysPtr->particles[i].extensive.rhoQ = 3;
        sysPtr->particles[i].extensive.rhoS = 4;
    }
    sysPtr->allocate_cabana_particles();
    ws.calculate_gamma_and_velocities();
    sysPtr->reset_neighbour_list();
    ws.smooth_all_particle_fields(tau*tau);
    sysPtr->copy_device_to_host();
    const ccake::Particle<2>& p = sysPtr->particles[0];
    ASSERT_LT(fabs(p.hydro.gamma - gamma), TOL);

    // sum of the kernel over particle 0 and its three neighbours
    const double kernel_sum = 20./7./M_PI;
    EXPECT_LT(fabs(p.smoothed.s    -    kernel_sum), TOL);
    EXPECT_LT(fabs(p.smoothed.rhoB -  4*kernel_sum), TOL);
    EXPECT_LT(fabs(p.smoothed.rhoQ -  9*kernel_sum), TOL);
    EXPECT_LT(fabs(p.smoothed.rhoS - 16*kernel_sum), TOL);
    EXPECT_LT(fabs(p.hydro.sigma   -    kernel_sum/(gamma*tau)), TOL);
    EXPECT_LT(fabs(p.thermo.s      -    kernel_sum/(gamma*tau)), TOL);
    EXPECT_LT(fabs(p.thermo.rhoB   -  4*kernel_sum/(gamma*tau)), TOL);
    EXPECT_LT(fabs(p.thermo.rhoQ   -  9*kernel_sum/(gamma*tau)), TOL);
    EXPECT_LT(fabs(p.thermo.rhoS   - 16*kernel_sum/(gamma*tau)), TOL);
}


// The dimensions, 3+1D first. CTest names end in <In3D> or <In2D>.
struct In3D { static constexpr unsigned int D = 3; };
struct In2D { static constexpr unsigned int D = 2; };
using Dimensions = ::testing::Types<In3D, In2D>;

namespace {

// Six particles in Milne coordinates (x, y, eta_s), some within reach of each
// other (smoothing length 1, support 2) and one alone; in 2+1D the first two
// coordinates. Each moves with its own u (u^eta in fm^-1).
constexpr int N_PARTICLES = 6;
const double POSITION[N_PARTICLES][3] = {{0., 0., 0.}, {0., .5, .3}, {.8, .6, -.2},
                                         {1.5, 0., .4}, {-.7, 1.1, .9}, {2.5, 2.5, 2.}};
const double U[N_PARTICLES][3] = {{0.3, -0.2, 0.1}, {-0.5, 0.2, 0.}, {0.1, 0.4, -0.3},
                                  {0.7, 0., 0.2}, {0., -0.6, 0.05}, {0.2, 0.2, 0.}};

template <unsigned int D>
double distance(int a, int b){
    double d2 = 0.;
    for (unsigned i = 0; i < D; ++i){
        const double x = POSITION[a][i] - POSITION[b][i];
        d2 += x*x;
    }
    return std::sqrt(d2);
}

// Lorentz factor of u in Milne coordinates: 1 + u^x^2 + u^y^2 + tau^2 u^eta^2.
template <unsigned int D>
double lorentz_factor(int a, double tau){
    double g2 = 1. + U[a][0]*U[a][0] + U[a][1]*U[a][1];
    if (D == 3) g2 += tau*tau*U[a][2]*U[a][2];
    return std::sqrt(g2);
}

template <unsigned int D>
std::shared_ptr<ccake::SystemState<D>> moving_particles(std::shared_ptr<Settings> settingsPtr, double tau){
    auto sysPtr = std::make_shared<ccake::SystemState<D>>(settingsPtr);
    sysPtr->particles.resize(N_PARTICLES);
    for (int a = 0; a < N_PARTICLES; ++a){
        ccake::Particle<D>& p = sysPtr->particles[a];
        p.ID = a;
        for (unsigned i = 0; i < D; ++i){
            p.r(i) = POSITION[a][i];
            p.hydro.u(i) = U[a][i];
        }
        p.sph_mass.s = 1.; p.sph_mass.rhoB = 2.; p.sph_mass.rhoQ = 3.; p.sph_mass.rhoS = 4.;
        p.extensive.s = 1. + 0.1*a;
        p.extensive.rhoB = 2. - 0.2*a;
        p.extensive.rhoQ = 0.5 + 0.3*a;
        p.extensive.rhoS = -1. + 0.25*a;
    }
    sysPtr->n_particles = N_PARTICLES;
    sysPtr->t = tau;
    return sysPtr;
}

std::shared_ptr<Settings> smoothing_settings(){
    auto settingsPtr = std::make_shared<Settings>();
    settingsPtr->xmin = -4;
    settingsPtr->ymin = -6;
    settingsPtr->etamin = -4;
    settingsPtr->hT = 1;
    settingsPtr->hEta = 1;
    return settingsPtr;
}

} // namespace

template <typename T> class SmoothingDimensions : public ::testing::Test {};
TYPED_TEST_SUITE(SmoothingDimensions, Dimensions);

/// Every particle's sums, in 3+1D and 2+1D, against the cubic spline of
/// kernel_reference.h: sigma_lab = sum_b m_b W(r_ab), the smoothed densities
/// sum_b m_b N_b W(r_ab) (b = a included), and the rest-frame densities
/// sigma = sigma_lab/(gamma tau), s = smoothed s/(gamma tau), ..., with gamma
/// computed by the code from u, as in a run.
TYPED_TEST(SmoothingDimensions, densities_are_kernel_sums_divided_by_gamma_tau){
    constexpr unsigned int D = TypeParam::D;
    const double tau = 1.5, h = 1.;
    auto settingsPtr = smoothing_settings();
    auto sysPtr = moving_particles<D>(settingsPtr, tau);
    ccake::SPHWorkstation<D, ccake::EoM_default> ws(settingsPtr, sysPtr);
    sysPtr->allocate_cabana_particles();
    ws.calculate_gamma_and_velocities();
    sysPtr->reset_neighbour_list();
    ws.smooth_all_particle_fields(tau*tau);
    sysPtr->copy_device_to_host();

    for (int a = 0; a < N_PARTICLES; ++a){
        double sigma_lab = 0., s = 0., rhoB = 0., rhoQ = 0., rhoS = 0.;
        for (int b = 0; b < N_PARTICLES; ++b){
            const double w = kernel_reference::W(D, distance<D>(a, b), h);
            const ccake::Particle<D>& q = sysPtr->particles[b];
            sigma_lab += q.sph_mass.s*w;
            s    += q.sph_mass.s*q.extensive.s*w;
            rhoB += q.sph_mass.rhoB*q.extensive.rhoB*w;
            rhoQ += q.sph_mass.rhoQ*q.extensive.rhoQ*w;
            rhoS += q.sph_mass.rhoS*q.extensive.rhoS*w;
        }
        const ccake::Particle<D>& p = sysPtr->particles[a];
        const double gamma_tau = lorentz_factor<D>(a, tau)*tau;
        const std::string at = " of particle " + std::to_string(a);
        EXPECT_LT(fabs(p.hydro.gamma - lorentz_factor<D>(a, tau)), TOL) << "gamma" << at;
        EXPECT_LT(fabs(p.hydro.sigma_lab - sigma_lab), TOL) << "sigma_lab" << at;
        EXPECT_LT(fabs(p.smoothed.s - s), TOL) << "smoothed s" << at;
        EXPECT_LT(fabs(p.smoothed.rhoB - rhoB), TOL) << "smoothed rhoB" << at;
        EXPECT_LT(fabs(p.smoothed.rhoQ - rhoQ), TOL) << "smoothed rhoQ" << at;
        EXPECT_LT(fabs(p.smoothed.rhoS - rhoS), TOL) << "smoothed rhoS" << at;
        EXPECT_LT(fabs(p.hydro.sigma - sigma_lab/gamma_tau), TOL) << "sigma" << at;
        EXPECT_LT(fabs(p.thermo.s - s/gamma_tau), TOL) << "s" << at;
        EXPECT_LT(fabs(p.thermo.rhoB - rhoB/gamma_tau), TOL) << "rhoB" << at;
        EXPECT_LT(fabs(p.thermo.rhoQ - rhoQ/gamma_tau), TOL) << "rhoQ" << at;
        EXPECT_LT(fabs(p.thermo.rhoS - rhoS/gamma_tau), TOL) << "rhoS" << at;
    }
    // particle 0 has neighbours; the lone particle sees only itself
    EXPECT_GT(sysPtr->particles[0].hydro.sigma_lab, 1.5*kernel_reference::W(D, 0., h));
    EXPECT_LT(fabs(sysPtr->particles[5].hydro.sigma_lab - kernel_reference::W(D, 0., h)), TOL);
}


/// The factorized 3+1D kernel, W = W_2D(r_perp; h_T) W_1D(|d eta_s|; h_eta), with
/// h_eta != h_T: every particle's sums against the product of the 2D and 1D
/// cubic splines of kernel_reference.h.
TEST(SmoothingFactorized3D, densities_are_products_of_transverse_and_longitudinal_kernels){
    const double tau = 1.5, h_T = 1., h_eta = 0.6;
    auto settingsPtr = smoothing_settings();
    settingsPtr->factorized_kernel = true;
    settingsPtr->hEta = h_eta;
    auto sysPtr = moving_particles<3>(settingsPtr, tau);
    ccake::SPHWorkstation<3, ccake::EoM_default> ws(settingsPtr, sysPtr);
    sysPtr->allocate_cabana_particles();
    ws.calculate_gamma_and_velocities();
    sysPtr->reset_neighbour_list();
    ws.smooth_all_particle_fields(tau*tau);
    sysPtr->copy_device_to_host();

    for (int a = 0; a < N_PARTICLES; ++a){
        double sigma_lab = 0., s = 0., rhoB = 0.;
        for (int b = 0; b < N_PARTICLES; ++b){
            const double dx = POSITION[a][0] - POSITION[b][0], dy = POSITION[a][1] - POSITION[b][1];
            const double w = kernel_reference::W(2, std::sqrt(dx*dx + dy*dy), h_T)
                           * kernel_reference::W(1, std::fabs(POSITION[a][2] - POSITION[b][2]), h_eta);
            const ccake::Particle<3>& q = sysPtr->particles[b];
            sigma_lab += q.sph_mass.s*w;
            s    += q.sph_mass.s*q.extensive.s*w;
            rhoB += q.sph_mass.rhoB*q.extensive.rhoB*w;
        }
        const ccake::Particle<3>& p = sysPtr->particles[a];
        const double gamma_tau = lorentz_factor<3>(a, tau)*tau;
        EXPECT_LT(fabs(p.hydro.sigma_lab - sigma_lab), TOL) << "sigma_lab of particle " << a;
        EXPECT_LT(fabs(p.smoothed.s - s), TOL) << "smoothed s of particle " << a;
        EXPECT_LT(fabs(p.smoothed.rhoB - rhoB), TOL) << "smoothed rhoB of particle " << a;
        EXPECT_LT(fabs(p.hydro.sigma - sigma_lab/gamma_tau), TOL) << "sigma of particle " << a;
    }
    EXPECT_GT(sysPtr->particles[0].hydro.sigma_lab,
              1.5*kernel_reference::W(2, 0., h_T)*kernel_reference::W(1, 0., h_eta));
}

// Particles at the given (x, y, eta_s), at rest, masses 1; returns sigma_lab
// and the neighbour count (btrack) of each, with the factorized kernel.
struct Neighbours { std::vector<double> sigma_lab; std::vector<int> count; };

Neighbours factorized_sums(const std::vector<std::array<double,3>>& positions,
                           double h_T, double h_eta){
    auto settingsPtr = smoothing_settings();
    settingsPtr->factorized_kernel = true;
    settingsPtr->hT = h_T;
    settingsPtr->hEta = h_eta;
    auto sysPtr = std::make_shared<ccake::SystemState<3>>(settingsPtr);
    sysPtr->particles.resize(positions.size());
    for (std::size_t a = 0; a < positions.size(); ++a){
        ccake::Particle<3>& p = sysPtr->particles[a];
        p.ID = (int) a;
        p.r = {positions[a][0], positions[a][1], positions[a][2]};
        p.hydro.gamma = 1.;
        p.sph_mass.s = 1.;
        p.extensive.s = 1.;
    }
    sysPtr->n_particles = (int) positions.size();
    sysPtr->t = 1.;
    ccake::SPHWorkstation<3, ccake::EoM_default> ws(settingsPtr, sysPtr);
    sysPtr->allocate_cabana_particles();
    sysPtr->reset_neighbour_list();
    ws.smooth_all_particle_fields(1.);
    sysPtr->copy_device_to_host();
    Neighbours result;
    for (const auto& p : sysPtr->particles){
        result.sigma_lab.push_back(p.hydro.sigma_lab);
        result.count.push_back(p.btrack);
    }
    return result;
}

double product_kernel(const std::array<double,3>& a, const std::array<double,3>& b,
                      double h_T, double h_eta){
    const double dx = a[0] - b[0], dy = a[1] - b[1];
    return kernel_reference::W(2, std::sqrt(dx*dx + dy*dy), h_T)
         * kernel_reference::W(1, std::fabs(a[2] - b[2]), h_eta);
}

/// The factorized kernel reaches a cylinder, r_perp < 2 h_T and |d eta_s| < 2 h_eta.
/// Neighbours in its corners, outside the sphere of radius 2 h_T, count too: here
/// dx = d eta_s = 1.5 with h_T = h_eta = 1 (distance 2.12). (Until 2026-11 the
/// neighbour list kept only the sphere.)
TEST(SmoothingFactorized3D, neighbours_in_the_corners_of_the_cylinder_count){
    const std::vector<std::array<double,3>> positions = {{0., 0., 0.}, {1.5, 0., 1.5}};
    const Neighbours n = factorized_sums(positions, 1., 1.);
    const double expected = product_kernel(positions[0], positions[0], 1., 1.)
                          + product_kernel(positions[0], positions[1], 1., 1.);
    EXPECT_LT(fabs(n.sigma_lab[0] - expected), TOL)
        << "sigma_lab is " << n.sigma_lab[0] << ", the kernel sum " << expected;
    EXPECT_EQ(n.count[0], 1);
}

/// With h_eta > h_T the cylinder reaches 2 h_eta along eta_s, beyond 2 h_T: a pair
/// at d eta_s = 2.5 with h_T = 1, h_eta = 1.5 is in.
TEST(SmoothingFactorized3D, longitudinal_reach_is_two_h_eta){
    const std::vector<std::array<double,3>> positions = {{0., 0., 0.}, {0., 0., 2.5}};
    const Neighbours n = factorized_sums(positions, 1., 1.5);
    const double expected = product_kernel(positions[0], positions[0], 1., 1.5)
                          + product_kernel(positions[0], positions[1], 1., 1.5);
    EXPECT_GT(product_kernel(positions[0], positions[1], 1., 1.5), 0.);
    EXPECT_LT(fabs(n.sigma_lab[0] - expected), TOL)
        << "sigma_lab is " << n.sigma_lab[0] << ", the kernel sum " << expected;
    EXPECT_EQ(n.count[0], 1);
}

/// The neighbour list holds exactly the pairs inside the cylinder: the neighbour
/// count (btrack, used by the freeze-out) of every particle of a scattered group
/// equals the number of others with r_perp < 2 h_T and |d eta_s| < 2 h_eta, and
/// sigma_lab equals the kernel sum. Two anisotropies, h_eta < h_T and h_eta > h_T.
TEST(SmoothingFactorized3D, neighbours_are_exactly_the_pairs_in_the_cylinder){
    std::vector<std::array<double,3>> positions;
    for (int i = 0; i < 40; ++i)   // a fixed scatter, no randomness
        positions.push_back({std::fmod(0.37*i, 3.1) - 1.5, std::fmod(0.61*i, 2.9) - 1.4,
                             std::fmod(0.23*i, 2.3) - 1.1});
    for (const std::array<double,2> h : {std::array<double,2>{0.8, 0.45}, std::array<double,2>{0.5, 0.9}}){
        const double h_T = h[0], h_eta = h[1];
        const Neighbours n = factorized_sums(positions, h_T, h_eta);
        for (std::size_t a = 0; a < positions.size(); ++a){
            int inside = 0;
            double sigma = 0.;
            for (std::size_t b = 0; b < positions.size(); ++b){
                sigma += product_kernel(positions[a], positions[b], h_T, h_eta);
                if (b == a) continue;
                const double dx = positions[a][0] - positions[b][0], dy = positions[a][1] - positions[b][1];
                if (dx*dx + dy*dy < 4.*h_T*h_T && std::fabs(positions[a][2] - positions[b][2]) < 2.*h_eta)
                    ++inside;
            }
            EXPECT_EQ(n.count[a], inside) << "particle " << a << ", h_T = " << h_T << ", h_eta = " << h_eta;
            EXPECT_LT(fabs(n.sigma_lab[a] - sigma), TOL) << "particle " << a << ", h_T = " << h_T;
        }
    }
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
