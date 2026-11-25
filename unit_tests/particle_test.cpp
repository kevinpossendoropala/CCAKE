// Copying a particle.
#include <gtest/gtest.h>

#include <sstream>
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

/// The getters return the thermodynamic fields they are named after.
TEST(Particle, getters_return_their_fields){
    ccake::Particle<3> p;
    double value = 1.;
    auto next = [&value](){ value += 0.5; return value; };
    auto& th = p.thermo;
    th.T = next(); th.muB = next(); th.muS = next(); th.muQ = next();
    th.p = next(); th.s = next(); th.e = next(); th.rhoB = next(); th.rhoS = next();
    th.rhoQ = next(); th.w = next(); th.cs2 = next(); th.dwds = next(); th.dwdB = next();
    th.dwdS = next(); th.dwdQ = next();
    th.dalpha_Bds = next(); th.dalpha_Qds = next(); th.dalpha_Sds = next();
    th.dalpha_BdB = next(); th.dalpha_QdB = next(); th.dalpha_SdB = next();
    th.dalpha_BdS = next(); th.dalpha_QdS = next(); th.dalpha_SdS = next();
    th.dalpha_BdQ = next(); th.dalpha_QdQ = next(); th.dalpha_SdQ = next();
    th.chiBB = next(); th.chiBB0 = next();
    th.eos_name = "tanh_conformal";

    EXPECT_EQ(p.T(), th.T);       EXPECT_EQ(p.muB(), th.muB);   EXPECT_EQ(p.muS(), th.muS);
    EXPECT_EQ(p.muQ(), th.muQ);   EXPECT_EQ(p.p(), th.p);       EXPECT_EQ(p.s(), th.s);
    EXPECT_EQ(p.e(), th.e);       EXPECT_EQ(p.rhoB(), th.rhoB); EXPECT_EQ(p.rhoS(), th.rhoS);
    EXPECT_EQ(p.rhoQ(), th.rhoQ); EXPECT_EQ(p.w(), th.w);       EXPECT_EQ(p.cs2(), th.cs2);
    EXPECT_EQ(p.dwds(), th.dwds); EXPECT_EQ(p.dwdB(), th.dwdB); EXPECT_EQ(p.dwdS(), th.dwdS);
    EXPECT_EQ(p.dwdQ(), th.dwdQ);
    EXPECT_EQ(p.dalpha_Bds(), th.dalpha_Bds); EXPECT_EQ(p.dalpha_Qds(), th.dalpha_Qds);
    EXPECT_EQ(p.dalpha_Sds(), th.dalpha_Sds); EXPECT_EQ(p.dalpha_BdB(), th.dalpha_BdB);
    EXPECT_EQ(p.dalpha_QdB(), th.dalpha_QdB); EXPECT_EQ(p.dalpha_SdB(), th.dalpha_SdB);
    EXPECT_EQ(p.dalpha_BdS(), th.dalpha_BdS); EXPECT_EQ(p.dalpha_QdS(), th.dalpha_QdS);
    EXPECT_EQ(p.dalpha_SdS(), th.dalpha_SdS); EXPECT_EQ(p.dalpha_BdQ(), th.dalpha_BdQ);
    EXPECT_EQ(p.dalpha_QdQ(), th.dalpha_QdQ); EXPECT_EQ(p.dalpha_SdQ(), th.dalpha_SdQ);
    EXPECT_EQ(p.chiBB(), th.chiBB); EXPECT_EQ(p.chiBB0(), th.chiBB0);
    EXPECT_EQ(p.get_current_eos_name(), "tanh_conformal");
}

/// The printout names each field next to its value.
TEST(Particle, printout_lists_the_fields){
    ccake::Particle<2> p;
    p.ID = 42;
    p.thermo.T = 0.75;
    p.input.e = 12.5;
    p.hydro.sigma = 3.25;
    p.hydro.tau_pi = 0.375;
    std::ostringstream out;
    out << p;
    const std::string text = out.str();
    EXPECT_NE(text.find("ID.............: 42\n"), std::string::npos);
    EXPECT_NE(text.find("thermo.T.......: 0.75\n"), std::string::npos);
    EXPECT_NE(text.find("input.e........: 12.5\n"), std::string::npos);
    EXPECT_NE(text.find("hydro.sigma...: 3.25\n"), std::string::npos);
    EXPECT_NE(text.find("hydro.tau_pi..: 0.375\n"), std::string::npos);
}

int main(int argc, char* argv[]) {
    // SystemState holds Cabana containers, which need Kokkos.
    Kokkos::ScopeGuard guard(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
