// Copying a particle.
#include <gtest/gtest.h>

#include <string>

#include "particle.h"
#include "system_state.h"

/// A copy keeps every field
TEST(Particle, copy_keeps_its_state){
    ccake::Particle<2> p;
    p.ID = 7;
    p.Freeze = 4;
    p.btrack = 3;
    p.r = {0.5, -1.0};
    p.thermo.T = 0.8;
    p.thermo.s = 3.0;
    p.thermo.rhoB = 0.4;
    p.thermo.e = 5.0;
    p.thermo.muB = 0.2;
    p.thermo.eos_name = "table";
    p.sph_mass.s = 2.0;
    p.extensive.s = 1.5;
    p.smoothed.s = 2.25;
    p.hydro.gamma = 1.25;
    p.hydro.sigma = 0.75;
    p.hydro.ID = 7;
    p.freezein_jB[2] = 0.125;

    const ccake::Particle<2> copy(p);

    EXPECT_EQ(copy.ID, 7);
    EXPECT_EQ(copy.Freeze, 4);
    EXPECT_EQ(copy.btrack, 3);
    EXPECT_EQ(copy.r(0), 0.5);
    EXPECT_EQ(copy.r(1), -1.0);
    EXPECT_EQ(copy.thermo.T, 0.8);
    EXPECT_EQ(copy.thermo.s, 3.0);
    EXPECT_EQ(copy.thermo.rhoB, 0.4);
    EXPECT_EQ(copy.thermo.e, 5.0);
    EXPECT_EQ(copy.thermo.muB, 0.2);
    EXPECT_EQ(copy.thermo.eos_name, std::string("table"));
    EXPECT_EQ(copy.sph_mass.s, 2.0);
    EXPECT_EQ(copy.extensive.s, 1.5);
    EXPECT_EQ(copy.smoothed.s, 2.25);
    EXPECT_EQ(copy.hydro.gamma, 1.25);
    EXPECT_EQ(copy.hydro.sigma, 0.75);
    EXPECT_EQ(copy.hydro.ID, 7);
    EXPECT_EQ(copy.freezein_jB[2], 0.125);
}

/// The fields the initial-condition readers set before add_particle().
TEST(Particle, copy_keeps_position_flow_and_input){
    ccake::Particle<2> p;
    p.r = {0.5, -1.0};
    p.input.e = 4.0;
    p.input.s = 2.5;
    p.input.rhoB = 0.3;
    p.hydro.u = {0.1, 0.2};
    p.hydro.bulk = -0.05;
    p.hydro.shv(1,2) = 0.01;
    p.thermo.T = 0.8;
    p.thermo.p = 1.2;

    const ccake::Particle<2> copy(p);

    EXPECT_EQ(copy.r(0), 0.5);
    EXPECT_EQ(copy.r(1), -1.0);
    EXPECT_EQ(copy.input.e, 4.0);
    EXPECT_EQ(copy.input.s, 2.5);
    EXPECT_EQ(copy.input.rhoB, 0.3);
    EXPECT_EQ(copy.hydro.u(0), 0.1);
    EXPECT_EQ(copy.hydro.u(1), 0.2);
    EXPECT_EQ(copy.hydro.bulk, -0.05);
    EXPECT_EQ(copy.hydro.shv(1,2), 0.01);
    EXPECT_EQ(copy.thermo.T, 0.8);
    EXPECT_EQ(copy.thermo.p, 1.2);
}

/// add_particle() takes its argument by value and the particles vector copies
/// or moves every element each time it grows: nothing set before may be lost.
TEST(Particle, add_particle_keeps_every_particle_as_the_vector_grows){
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings>();
    ccake::SystemState<2> system(settingsPtr);
    const int n = 1000;
    for (int i = 0; i < n; ++i){
        ccake::Particle<2> p;
        p.ID = i;
        p.thermo.s = 0.5*i;
        p.sph_mass.s = 1.0 + i;
        system.add_particle(p);
    }
    ASSERT_EQ((int) system.particles.size(), n);
    for (int i = 0; i < n; ++i){
        EXPECT_EQ(system.particles[i].ID, i);
        EXPECT_EQ(system.particles[i].thermo.s, 0.5*i);
        EXPECT_EQ(system.particles[i].sph_mass.s, 1.0 + i);
    }
}

int main(int argc, char* argv[]) {
    // SystemState holds Cabana containers, which need Kokkos.
    Kokkos::ScopeGuard guard(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
