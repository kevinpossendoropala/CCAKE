#include <gtest/gtest.h>
#include <functional>
#include <memory>
//#include <gmock/gmock.h> //To mock settings pointers

#include "sph_workstation.h"
#include "system_state.h"
#include "settings.h"
#include "particle.h"
#include "evolver.h"
#define TOL 5.E-13

namespace cc = ccake;

std::tuple<std::shared_ptr<cc::SystemState<2>>,std::shared_ptr<cc::Evolver<2>>> prepare_system(){
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings> ();
    settingsPtr->xmin = -4;
    settingsPtr->ymin = -6;
    settingsPtr->etamin = -2;
    settingsPtr->hT = 1;
    settingsPtr->using_shear = true;
    settingsPtr->t0 = 2;

    std::shared_ptr<cc::SystemState<2>> sysPtr = std::make_shared<cc::SystemState<2>>(settingsPtr);
    sysPtr->n_particles=1;
    sysPtr->t=2.;
    cc::Particle<2> p;
    sysPtr->particles.push_back(p);
    sysPtr->initialize();

    std::shared_ptr<cc::Evolver<2>> evo = std::make_shared<cc::Evolver<2>>(settingsPtr,sysPtr); 

    return std::make_tuple(sysPtr,evo);
}

 
TEST(evolver, test_rk2){

    std::tuple<std::shared_ptr<cc::SystemState<2>>,std::shared_ptr<cc::Evolver<2>>> sys_evo_tuple;
    sys_evo_tuple = prepare_system();
    std::shared_ptr<cc::SystemState<2>> systemPtr = std::get<0>(sys_evo_tuple);
    std::shared_ptr<cc::Evolver<2>> evoPtr = std::get<1>(sys_evo_tuple);

    cc::Particle<2>* p = &systemPtr->particles[0];
    
    //Initial positions
    p->r(0) = -1.5;
    p->r(1) = -2.5;

    //Initial velocity
    p->hydro.u(0) = 3.;
    p->hydro.u(1) = 4.;
    p->hydro.gamma = sqrt(26.);
    p->hydro.v(0) = 3./sqrt(26.);
    p->hydro.v(1) = 4./sqrt(26.);

    //Initial entropy
    p->extensive.s = 2.;
    
    // The evolver now integrates the "extensive" bulk pressure and shear tensor:
    // hydro.extensive_bulk replaces hydro.Bulk, and the independent shear
    // components live in the 2x3 matrix hydro.extensive_shv, whose entries
    // (0,0), (0,1), (1,1) replace shv(1,1), shv(1,2), shv(2,2). The components
    // that follow from them (shv(0,i), shv33) are no longer evolved, so they
    // are not set here. Values and expected results are those of the original test.

    //Initial Bulk
    p->hydro.extensive_bulk = 6.;

    //Initial viscous shear tensor
    p->hydro.extensive_shv(0,0) = 3.;
    p->hydro.extensive_shv(0,1) = 4.;
    p->hydro.extensive_shv(1,1) = 5.;

    //Setup derivatives
    p->hydro.du_dt(0) = 9./2.;
    p->hydro.du_dt(1) = 7./2.;

    p->d_dt_extensive.s = 5./2.;
    p->hydro.d_extensive_bulk_dt = 3./2.;

    p->hydro.d_extensive_shv_dt(0,0) = M_PI;
    p->hydro.d_extensive_shv_dt(0,1) = 2*M_PI;
    p->hydro.d_extensive_shv_dt(1,1) = 3*M_PI;

    p->ID = 0;
    
    double dt = .5;
    
    double const expected_r0 = -1.5 + dt*3./sqrt(26.);
    double const expected_r1 = -2.5 + dt*4./sqrt(26.);

    double const expected_u0 = 3 + dt*9./2.;
    double const expected_u1 = 4 + dt*7./2.;

    double const expected_extensive_s = 2. + dt*5./2.;
    double const expected_Bulk = 6. + dt*3./2.;
    double const expected_shv11 = 3. + dt*M_PI;
    double const expected_shv12 = 4. + dt*2*M_PI;
    double const expected_shv22 = 5. + dt*3.*M_PI;

    auto evaluate_time_derivatives = [](){};

    systemPtr->allocate_cabana_particles();
    //systemPtr->copy_host_to_device();
    // advance_timestep_rk2 is private now; execute_timestep with order 2 calls it.
    evoPtr->allocate_cache();
    evoPtr->allocate_k_values();
    evoPtr->execute_timestep(dt, 2, evaluate_time_derivatives);
    systemPtr->copy_device_to_host();
    p = &systemPtr->particles[0];

    //Asserts that we have the correct shear tensor
    ASSERT_DOUBLE_EQ(p->r(0), expected_r0);
    ASSERT_DOUBLE_EQ(p->r(1), expected_r1);
    ASSERT_DOUBLE_EQ(p->hydro.u(0), expected_u0);
    ASSERT_DOUBLE_EQ(p->hydro.u(1), expected_u1);
    ASSERT_DOUBLE_EQ(p->extensive.s, expected_extensive_s);
    ASSERT_DOUBLE_EQ(p->hydro.extensive_bulk, expected_Bulk);

    ASSERT_DOUBLE_EQ(p->hydro.extensive_shv(0,0), expected_shv11);
    ASSERT_DOUBLE_EQ(p->hydro.extensive_shv(0,1), expected_shv12);
    ASSERT_DOUBLE_EQ(p->hydro.extensive_shv(1,1), expected_shv22);
    // one full step also advances the clock
    ASSERT_DOUBLE_EQ(systemPtr->t, 2. + dt);
}


namespace {

// The derivatives as functions of the current state, written on the device the
// way evaluate_time_derivatives would: du/dt = -u, dr/dt = v = u, and
// d(extensive s)/dt = -(extensive s), d(extensive Pi)/dt = -(extensive Pi).
std::function<void()> exponential_decay(std::shared_ptr<cc::SystemState<2>> systemPtr){
    return [systemPtr](){
        CREATE_VIEW(device_, systemPtr->cabana_particles);
        auto policy = Cabana::SimdPolicy<VECTOR_LENGTH, ExecutionSpace>(0, systemPtr->cabana_particles.size());
        Cabana::simd_parallel_for(policy, KOKKOS_LAMBDA(const int is, const int ia){
            for (int i = 0; i < 2; ++i){
                const double u = device_hydro_vector.access(is, ia, cc::hydro_info::u, i);
                device_hydro_vector.access(is, ia, cc::hydro_info::du_dt, i) = -u;
                device_hydro_vector.access(is, ia, cc::hydro_info::v, i) = u;
            }
            device_d_dt_extensive.access(is, ia, cc::densities_info::s)
                = -device_extensive.access(is, ia, cc::densities_info::s);
            device_hydro_scalar.access(is, ia, cc::hydro_info::d_extensive_bulk_dt)
                = -device_hydro_scalar.access(is, ia, cc::hydro_info::extensive_bulk);
        }, "decay");
        Kokkos::fence();
    };
}

// Constant derivatives: du/dt = (1.5, -0.5), v = 0, d(extensive s)/dt = 2.
std::function<void()> constant_rates(std::shared_ptr<cc::SystemState<2>> systemPtr){
    return [systemPtr](){
        CREATE_VIEW(device_, systemPtr->cabana_particles);
        auto policy = Cabana::SimdPolicy<VECTOR_LENGTH, ExecutionSpace>(0, systemPtr->cabana_particles.size());
        Cabana::simd_parallel_for(policy, KOKKOS_LAMBDA(const int is, const int ia){
            device_hydro_vector.access(is, ia, cc::hydro_info::du_dt, 0) = 1.5;
            device_hydro_vector.access(is, ia, cc::hydro_info::du_dt, 1) = -0.5;
            device_hydro_vector.access(is, ia, cc::hydro_info::v, 0) = 0.;
            device_hydro_vector.access(is, ia, cc::hydro_info::v, 1) = 0.;
            device_d_dt_extensive.access(is, ia, cc::densities_info::s) = 2.;
        }, "constant");
        Kokkos::fence();
    };
}

// One particle at r = (-1.5, -2.5) with u = (3, 4), extensive s = 2 and
// extensive Pi = 6, at t = 2. Advances one step of the given order.
std::shared_ptr<cc::SystemState<2>> one_step(double dt, int order,
        std::function<std::function<void()>(std::shared_ptr<cc::SystemState<2>>)> derivatives){
    auto system_and_evolver = prepare_system();
    std::shared_ptr<cc::SystemState<2>> systemPtr = std::get<0>(system_and_evolver);
    std::shared_ptr<cc::Evolver<2>> evoPtr = std::get<1>(system_and_evolver);
    cc::Particle<2>& p = systemPtr->particles[0];
    p.ID = 0;
    p.r = {-1.5, -2.5};
    p.hydro.u = {3., 4.};
    p.extensive.s = 2.;
    p.hydro.extensive_bulk = 6.;
    systemPtr->allocate_cabana_particles();
    evoPtr->allocate_cache();
    evoPtr->allocate_k_values();
    evoPtr->execute_timestep(dt, order, derivatives(systemPtr));
    systemPtr->copy_device_to_host();
    return systemPtr;
}

// y(dt)/y(0) of y' = -y for a correct method of order 2 or 4 with as many stages:
// for a linear equation every such method gives the Taylor polynomial of exp(-dt).
double growth(double dt, int order){
    double g = 1. - dt + dt*dt/2.;
    if (order == 4) g += -dt*dt*dt/6. + dt*dt*dt*dt/24.;
    return g;
}

} // namespace

/// Second-order Runge-Kutta on du/dt = -u, dr/dt = u, ds/dt = -s, dPi/dt = -Pi.
TEST(evolver, rk2_linear_decay_is_second_order_exact){
    const double dt = 0.1;
    auto systemPtr = one_step(dt, 2, exponential_decay);
    const cc::Particle<2>& p = systemPtr->particles[0];
    const double g = growth(dt, 2);
    EXPECT_DOUBLE_EQ(p.hydro.u(0), 3.*g);
    EXPECT_DOUBLE_EQ(p.hydro.u(1), 4.*g);
    // r = r0 + u0*(1 - g), the integral of u to the same order
    EXPECT_DOUBLE_EQ(p.r(0), -1.5 + 3.*(1. - g));
    EXPECT_DOUBLE_EQ(p.r(1), -2.5 + 4.*(1. - g));
    EXPECT_DOUBLE_EQ(p.extensive.s, 2.*g);
    EXPECT_DOUBLE_EQ(p.hydro.extensive_bulk, 6.*g);
    EXPECT_DOUBLE_EQ(systemPtr->t, 2. + dt);
}

/// A constant derivative is integrated exactly by any Runge-Kutta method.
TEST(evolver, rk2_constant_derivative_is_exact){
    const double dt = 0.1;
    auto systemPtr = one_step(dt, 2, constant_rates);
    const cc::Particle<2>& p = systemPtr->particles[0];
    EXPECT_DOUBLE_EQ(p.hydro.u(0), 3. + 1.5*dt);
    EXPECT_DOUBLE_EQ(p.hydro.u(1), 4. - 0.5*dt);
    EXPECT_DOUBLE_EQ(p.extensive.s, 2. + 2.*dt);
    EXPECT_DOUBLE_EQ(p.r(0), -1.5);
}

/// Fourth-order Runge-Kutta: the weights 1/6, 2/6, 2/6, 1/6 add up to one, so a
/// constant derivative is integrated exactly. (Until 2026-11 the stages were
/// summed with weights adding up to 7/6.)
TEST(evolver, rk4_constant_derivative_is_exact){
    const double dt = 0.1;
    auto systemPtr = one_step(dt, 4, constant_rates);
    const cc::Particle<2>& p = systemPtr->particles[0];
    EXPECT_DOUBLE_EQ(p.hydro.u(0), 3. + 1.5*dt);
    EXPECT_DOUBLE_EQ(p.hydro.u(1), 4. - 0.5*dt);
    EXPECT_DOUBLE_EQ(p.extensive.s, 2. + 2.*dt);
    EXPECT_DOUBLE_EQ(systemPtr->t, 2. + dt);
}

/// Fourth-order Runge-Kutta on du/dt = -u, ds/dt = -s, dPi/dt = -Pi: each stage
/// evaluated at the right point gives the Taylor polynomial of exp(-dt) to
/// fourth order.
TEST(evolver, rk4_linear_decay_is_fourth_order_exact){
    const double dt = 0.1;
    auto systemPtr = one_step(dt, 4, exponential_decay);
    const cc::Particle<2>& p = systemPtr->particles[0];
    const double g = growth(dt, 4);
    EXPECT_DOUBLE_EQ(p.hydro.u(0), 3.*g);
    EXPECT_DOUBLE_EQ(p.hydro.u(1), 4.*g);
    EXPECT_DOUBLE_EQ(p.extensive.s, 2.*g);
    EXPECT_DOUBLE_EQ(p.hydro.extensive_bulk, 6.*g);
}

/// The steps advance the extensive bulk only. The bulk pressure itself is
/// Pi = extensive_bulk*sigma, recomputed at the next derivative evaluation, so
/// neither order may write the extensive value into it. (RK4 used to.)
TEST(evolver, rk2_and_rk4_leave_the_bulk_pressure_alone){
    const double dt = 0.1;
    auto rk2 = one_step(dt, 2, constant_rates);
    auto rk4 = one_step(dt, 4, constant_rates);
    EXPECT_DOUBLE_EQ(rk2->particles[0].hydro.extensive_bulk, 6.);
    EXPECT_DOUBLE_EQ(rk4->particles[0].hydro.extensive_bulk, 6.);
    EXPECT_DOUBLE_EQ(rk2->particles[0].hydro.bulk, 0.);
    EXPECT_DOUBLE_EQ(rk4->particles[0].hydro.bulk, 0.);
}

/// A frozen particle (Freeze >= 4) whose entropy would go negative is kept at
/// 1e-3 instead of stopping the run.
TEST(evolver, frozen_particle_negative_entropy_is_floored){
    auto system_and_evolver = prepare_system();
    std::shared_ptr<cc::SystemState<2>> systemPtr = std::get<0>(system_and_evolver);
    std::shared_ptr<cc::Evolver<2>> evoPtr = std::get<1>(system_and_evolver);
    cc::Particle<2>& p = systemPtr->particles[0];
    p.ID = 0;
    p.Freeze = 4;
    p.extensive.s = 0.1;
    systemPtr->allocate_cabana_particles();
    evoPtr->allocate_cache();
    evoPtr->allocate_k_values();
    auto drain = [systemPtr](){
        CREATE_VIEW(device_, systemPtr->cabana_particles);
        auto policy = Cabana::SimdPolicy<VECTOR_LENGTH, ExecutionSpace>(0, systemPtr->cabana_particles.size());
        Cabana::simd_parallel_for(policy, KOKKOS_LAMBDA(const int is, const int ia){
            device_d_dt_extensive.access(is, ia, cc::densities_info::s) = -10.;
        }, "drain");
        Kokkos::fence();
    };
    evoPtr->execute_timestep(0.5, 2, drain);
    systemPtr->copy_device_to_host();
    EXPECT_DOUBLE_EQ(systemPtr->particles[0].extensive.s, 1.e-3);
}

/// The same for a particle that is still fluid stops the run.
TEST(evolver, fluid_particle_negative_entropy_stops_the_run){
    auto system_and_evolver = prepare_system();
    std::shared_ptr<cc::SystemState<2>> systemPtr = std::get<0>(system_and_evolver);
    std::shared_ptr<cc::Evolver<2>> evoPtr = std::get<1>(system_and_evolver);
    cc::Particle<2>& p = systemPtr->particles[0];
    p.ID = 0;
    p.Freeze = 0;
    p.extensive.s = 0.1;
    systemPtr->allocate_cabana_particles();
    systemPtr->reset_neighbour_list();      // the error message lists the neighbours
    evoPtr->allocate_cache();
    evoPtr->allocate_k_values();
    auto drain = [systemPtr](){
        CREATE_VIEW(device_, systemPtr->cabana_particles);
        auto policy = Cabana::SimdPolicy<VECTOR_LENGTH, ExecutionSpace>(0, systemPtr->cabana_particles.size());
        Cabana::simd_parallel_for(policy, KOKKOS_LAMBDA(const int is, const int ia){
            device_d_dt_extensive.access(is, ia, cc::densities_info::s) = -10.;
        }, "drain");
        Kokkos::fence();
    };
    EXPECT_EXIT(evoPtr->execute_timestep(0.5, 2, drain), testing::ExitedWithCode(EXIT_FAILURE), "");
}

// Define the main function
int main(int argc, char* argv[]) {
 
    // Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);

    // Run the tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}