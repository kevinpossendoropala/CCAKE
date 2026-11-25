#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>
//#include <gmock/gmock.h> //To mock settings pointers

#include "kernel_reference.h"
#include "sph_workstation.h"
#include "system_state.h"
#include "settings.h"
#include "particle.h"
#define TOL 1.E-14


TEST(GradSmoothingTest, Smooth2D){

    // (Kokkos is started once, in main() at the end of this file)
    //Create mocked settings
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings> ();
    settingsPtr->xmin = -4;
    settingsPtr->ymin = -6;
    settingsPtr->etamin = -2;
    settingsPtr->hT = 1;
    std::shared_ptr<ccake::SystemState<2>> sysPtr = std::make_shared<ccake::SystemState<2>> (settingsPtr);
    ccake::SPHWorkstation<2,ccake::EoM_default> ws(settingsPtr, sysPtr);
    const double target_val_x = 45./8./M_PI;
    const double target_val_y = 0;

    //ccake::SystemState<2> sys(settingsPtr);
    // (the smoothing length is read from the settings; SystemState has no hT of its own)

    //Place particles in positions {0,0}, {.5,0}, {.8,.6} and {0,1.5}
    const int n_sph_particles = 5;
    std::array<ccake::Particle<2>,n_sph_particles> particles_array_in;
    particles_array_in[0].r = {0,0}; //Should not contribute
    particles_array_in[1].r = {.5,0};
    particles_array_in[2].r = {1.0,0};
    particles_array_in[3].r = {1.5,0};
    particles_array_in[4].r = {-1.5,1.5}; //Should not contrinute
    //particles_array_in[4].r = {0,0};
    //particles_array_in[5].r = {-.5,0};
    //particles_array_in[6].r = {-1.0,0};
    //particles_array_in[7].r = {-1.5,0};
    for (auto p : particles_array_in) sysPtr->add_particle(p);
    sysPtr->n_particles = n_sph_particles;
    for (int i=0; i<n_sph_particles; ++i){
        sysPtr->particles[i].ID = i;
        sysPtr->particles[i].sph_mass.s = 1;
        sysPtr->particles[i].sph_mass.rhoB = 2;
        sysPtr->particles[i].sph_mass.rhoQ = 3;
        sysPtr->particles[i].sph_mass.rhoS = 4;
        // The densities are set by hand (this checks the gradient sum for given
        // densities): sigma_lab = 1, and sigma = sigma_lab/(gamma t) with the
        // fluid at rest (gamma = 1) at t = 1. The gradients read sigma_lab.
        sysPtr->particles[i].hydro.gamma = 1;
        sysPtr->particles[i].hydro.sigma = 1;
        sysPtr->particles[i].hydro.sigma_lab = 1;
        sysPtr->particles[i].thermo.p = 1;


        sysPtr->particles[i].extensive.s = 1;
        sysPtr->particles[i].extensive.rhoB = 2;
        sysPtr->particles[i].extensive.rhoQ = 3;
        sysPtr->particles[i].extensive.rhoS = 4;
    }
    sysPtr->particles[1].thermo.p = .5;
    sysPtr->particles[2].thermo.p = 1.5;
    sysPtr->particles[3].thermo.p = 2.5;

    sysPtr->allocate_cabana_particles();
    sysPtr->reset_neighbour_list();    
    ws.smooth_all_particle_gradients(1.);
    sysPtr->copy_device_to_host();
    // all contributing neighbours lie on the x axis, so there is no y component
    EXPECT_LT( fabs(sysPtr->particles[0].hydro.gradP(1) - target_val_y), TOL)
        << "gradP(1) is " << sysPtr->particles[0].hydro.gradP(1);
    EXPECT_LT( fabs(sysPtr->particles[0].hydro.gradP(0) - target_val_x), TOL) << "Test failed! Reporting values used: " << endl
    << "Particle 1: -------" << endl
    << "r: " << sysPtr->particles[0].r << endl
    << "sph_mass: " << sysPtr->particles[0].sph_mass.s << endl
    << "extensive: " << sysPtr->particles[0].extensive.s << endl
    << "smoothed: " << sysPtr->particles[0].smoothed.s << endl
    << "sigma^*: " << sysPtr->particles[0].hydro.sigma << endl
    << "p: " << sysPtr->particles[0].thermo.p << endl
    << "grad_p: " << sysPtr->particles[0].hydro.gradP << endl
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

template <typename T> class GradientDimensions : public ::testing::Test {};
TYPED_TEST_SUITE(GradientDimensions, Dimensions);

/// Pressure and velocity gradients of every particle, in 3+1D and 2+1D, against
/// the SPH sums with the cubic spline of kernel_reference.h,
///   grad p_a      = sum_b m_b sigma_a (p_a/sigma_a^2 + p_b/sigma_b^2) grad_a W(r_ab),
///   d_i v^j (a)   = sum_b (m_b/sigma_a) (v_b^j - v_a^j) d_i W(r_ab),
/// with grad_a W = W'(r) (r_a - r_b)/r and sigma = sigma_lab = sum_b m_b W(r_ab).
/// As in a run, the code computes gamma and v from u and sigma_lab by its own
/// smoothing before the gradients.
TYPED_TEST(GradientDimensions, pressure_and_velocity_gradients_are_sph_sums){
    constexpr unsigned int D = TypeParam::D;
    const double tau = 1.5, h = 1.;
    auto settingsPtr = smoothing_settings();
    auto sysPtr = moving_particles<D>(settingsPtr, tau);
    double pressure[N_PARTICLES];
    for (int a = 0; a < N_PARTICLES; ++a){
        pressure[a] = 1. + 0.3*POSITION[a][0] - 0.2*POSITION[a][1] + (D == 3 ? 0.5*POSITION[a][2] : 0.);
        sysPtr->particles[a].thermo.p = pressure[a];
    }
    ccake::SPHWorkstation<D, ccake::EoM_default> ws(settingsPtr, sysPtr);
    sysPtr->allocate_cabana_particles();
    ws.calculate_gamma_and_velocities();
    sysPtr->reset_neighbour_list();
    ws.smooth_all_particle_fields(tau*tau);
    ws.smooth_all_particle_gradients(tau*tau);
    sysPtr->copy_device_to_host();

    double sigma[N_PARTICLES], v[N_PARTICLES][3];
    for (int a = 0; a < N_PARTICLES; ++a){
        sigma[a] = 0.;
        for (int b = 0; b < N_PARTICLES; ++b) sigma[a] += kernel_reference::W(D, distance<D>(a, b), h);
        for (unsigned i = 0; i < D; ++i) v[a][i] = U[a][i]/lorentz_factor<D>(a, tau);
    }
    for (int a = 0; a < N_PARTICLES; ++a){
        double gradP[3] = {0., 0., 0.}, gradV[3][3] = {{0.}};
        for (int b = 0; b < N_PARTICLES; ++b){
            const double r = distance<D>(a, b);
            if (b == a || r >= 2.*h) continue;
            const double dw = kernel_reference::dW(D, r, h);
            for (unsigned i = 0; i < D; ++i){
                const double grad_w = dw*(POSITION[a][i] - POSITION[b][i])/r;
                gradP[i] += sigma[a]*(pressure[a]/(sigma[a]*sigma[a])
                                      + pressure[b]/(sigma[b]*sigma[b]))*grad_w;
                for (unsigned j = 0; j < D; ++j)
                    gradV[i][j] += (1./sigma[a])*(v[b][j] - v[a][j])*grad_w;
            }
        }
        const ccake::Particle<D>& p = sysPtr->particles[a];
        if (a == 0){   // has neighbours: the sums are not trivially zero
            EXPECT_GT(std::fabs(gradP[0]), 0.1);
            EXPECT_GT(std::fabs(gradV[0][1]), 0.01);
        }
        ASSERT_LT(fabs(p.hydro.sigma_lab - sigma[a]), TOL) << "sigma_lab of particle " << a;
        for (unsigned i = 0; i < D; ++i){
            EXPECT_LT(fabs(p.hydro.gradP(i) - gradP[i]), TOL)
                << "d_" << i << " p of particle " << a << ": " << p.hydro.gradP(i) << " vs " << gradP[i];
            for (unsigned j = 0; j < D; ++j)
                EXPECT_LT(fabs(p.hydro.gradV(i,j) - gradV[i][j]), TOL)
                    << "d_" << i << " v^" << j << " of particle " << a;
        }
    }
}


/// The factorized 3+1D kernel, W = W_2D(r_perp; h_T) W_1D(|d eta_s|; h_eta), with
/// h_eta != h_T: the pressure gradient of every particle against the SPH sum with
/// the gradient of the product,
///   d_x W = W_2D'(r_perp) (dx/r_perp) W_1D,  d_y likewise,
///   d_eta W = W_2D W_1D'(|d eta|) sign(d eta).
TEST(GradientFactorized3D, pressure_gradient_uses_the_product_kernel){
    const double tau = 1.5, h_T = 1., h_eta = 0.6;
    auto settingsPtr = smoothing_settings();
    settingsPtr->factorized_kernel = true;
    settingsPtr->hEta = h_eta;
    auto sysPtr = moving_particles<3>(settingsPtr, tau);
    double pressure[N_PARTICLES];
    for (int a = 0; a < N_PARTICLES; ++a){
        pressure[a] = 1. + 0.3*POSITION[a][0] - 0.2*POSITION[a][1] + 0.5*POSITION[a][2];
        sysPtr->particles[a].thermo.p = pressure[a];
    }
    ccake::SPHWorkstation<3, ccake::EoM_default> ws(settingsPtr, sysPtr);
    sysPtr->allocate_cabana_particles();
    ws.calculate_gamma_and_velocities();
    sysPtr->reset_neighbour_list();
    ws.smooth_all_particle_fields(tau*tau);
    ws.smooth_all_particle_gradients(tau*tau);
    sysPtr->copy_device_to_host();

    auto kernel = [&](int a, int b, double grad[3]){
        const double dx = POSITION[a][0] - POSITION[b][0], dy = POSITION[a][1] - POSITION[b][1];
        const double de = POSITION[a][2] - POSITION[b][2];
        const double r_perp = std::sqrt(dx*dx + dy*dy);
        const double w2 = kernel_reference::W(2, r_perp, h_T), w1 = kernel_reference::W(1, std::fabs(de), h_eta);
        const double dw2 = r_perp > 0. ? kernel_reference::dW(2, r_perp, h_T)/r_perp : 0.;
        grad[0] = dw2*dx*w1;
        grad[1] = dw2*dy*w1;
        grad[2] = w2*kernel_reference::dW(1, std::fabs(de), h_eta)*(de > 0. ? 1. : (de < 0. ? -1. : 0.));
        return w2*w1;
    };
    double sigma[N_PARTICLES];
    for (int a = 0; a < N_PARTICLES; ++a){
        sigma[a] = 0.;
        double unused[3];
        for (int b = 0; b < N_PARTICLES; ++b) sigma[a] += kernel(a, b, unused);
    }
    for (int a = 0; a < N_PARTICLES; ++a){
        double gradP[3] = {0., 0., 0.};
        for (int b = 0; b < N_PARTICLES; ++b){
            if (b == a) continue;
            double grad_w[3];
            kernel(a, b, grad_w);
            for (int i = 0; i < 3; ++i)
                gradP[i] += sigma[a]*(pressure[a]/(sigma[a]*sigma[a])
                                      + pressure[b]/(sigma[b]*sigma[b]))*grad_w[i];
        }
        const ccake::Particle<3>& p = sysPtr->particles[a];
        ASSERT_LT(fabs(p.hydro.sigma_lab - sigma[a]), TOL) << "sigma_lab of particle " << a;
        for (int i = 0; i < 3; ++i)
            EXPECT_LT(fabs(p.hydro.gradP(i) - gradP[i]), TOL)
                << "d_" << i << " p of particle " << a << ": " << p.hydro.gradP(i) << " vs " << gradP[i];
        if (a == 0) EXPECT_GT(std::fabs(gradP[2]), 0.01);   // the eta direction really counts
    }
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
