#include <gtest/gtest.h>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <Cabana_Core.hpp>
//#include <gmock/gmock.h> //To mock settings pointers

#include "system_state.h"
#include "settings.h"
#include "particle.h"
#include "eos_interpolator.h"
#include "hydrodynamic_info.h"
#include "thermodynamic_info.h"
#include "densities.h"

#define TOL 5.E-13

namespace cc = ccake;

std::shared_ptr<cc::SystemState<2>> prepare_system(){
    int const num_particles = 10000;
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings> ();
    settingsPtr->xmin = -4;
    settingsPtr->ymin = -6;
    settingsPtr->etamin = -2;
    settingsPtr->hT = 1;
    settingsPtr->using_shear = true;
    settingsPtr->t0 = 2;

    std::shared_ptr<cc::SystemState<2>> sysPtr = std::make_shared<cc::SystemState<2>>(settingsPtr);
    sysPtr->n_particles=num_particles;
    sysPtr->t=2.;
    for (int i=0; i<num_particles; i++){
        cc::Particle<2> p;
        sysPtr->particles.push_back(p);
        // Only particle 0 is given a state; the test checks particle 0 only.
        ccake::Particle<2>* pPtr = &sysPtr->particles[0];
        pPtr->thermo.s = 1.1;
        pPtr->thermo.rhoB = 1.2;
        pPtr->thermo.rhoQ = 1.3;
        pPtr->thermo.rhoS = 1.4;
    }

    return sysPtr;
}

// The fake equation of state of this test: every quantity is a multiple of
// s + rhoB + rhoQ + rhoS. A function that is linear in its four arguments is
// reproduced exactly by the interpolator, whatever the grid.
std::array<double,11> fake_eos(double s, double B, double Q, double S){
    double e = s+B+Q+S;
    /// T muB muQ muS e p cs2 dwds dwdB dwdQ dwdS
    return {e, e*2, e*3, e*4, e*5, e*6, e*7, e*8, e*9, e*10, e*11};
}

// Writes the fake equation of state in the format EoS_Interpolator reads: one
// HDF5 group "table_<s>_<B>_<S>_<Q>" per combination of decades (10^s .. 10^(s+1)
// in entropy density, and so on), with the grid limits as attributes and one
// 4D dataset per quantity, axes ordered (s, rhoB, rhoS, rhoQ). Two points per
// axis are enough here because the function is linear.
// (This replaces the text table the test used to write; the interpolator no
// longer reads that format.)
// With N_written > N the datasets hold more points per axis than the min, max
// and step attributes describe, as utilities/generate_eos_table.cpp used to
// write them.
using FakeEoS = std::function<std::array<double,11>(double, double, double, double)>;

void generate_fake_eos(const std::string& filename = "eos_fake.h5",
                       FakeEoS eos_function = fake_eos, const int N = 2,
                       int N_written = 0){
    const char* names[11] = {"T", "muB", "muQ", "muS", "e", "p", "cs2",
                             "dwds", "dwdB", "dwdQ", "dwdS"};
    if (N_written < N) N_written = N;
    const int M = N_written;
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::DataSpace scalar(H5S_SCALAR);
    const hsize_t dims[4] = {(hsize_t) M, (hsize_t) M, (hsize_t) M, (hsize_t) M};
    H5::DataSpace grid(4, dims);

    auto write_double = [&](H5::Group& group, const std::string& name, double value){
        group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar)
             .write(H5::PredType::NATIVE_DOUBLE, &value);
    };
    auto write_int = [&](H5::Group& group, const std::string& name, int value){
        group.createAttribute(name, H5::PredType::NATIVE_INT, scalar)
             .write(H5::PredType::NATIVE_INT, &value);
    };

    for (int es=-3; es<2; ++es)
    for (int eB=-3; eB<1; ++eB)
    for (int eS=-3; eS<1; ++eS)
    for (int eQ=-3; eQ<1; ++eQ){
        const int exponents[4] = {es, eB, eS, eQ};
        const char* axes[4] = {"s", "rhoB", "rhoS", "rhoQ"};
        double min[4], step[4];
        std::string table_name = "table";
        for (int i=0; i<4; ++i){
            table_name += "_" + std::to_string(exponents[i]);
            min[i] = std::pow(10., exponents[i]);
            step[i] = (std::pow(10., exponents[i]+1) - min[i])/(N-1);
        }
        H5::Group group = file.createGroup(table_name);
        for (int i=0; i<4; ++i){
            const std::string axis = axes[i];
            write_double(group, axis + "_min", min[i]);
            write_double(group, axis + "_max", min[i] + (N-1)*step[i]);
            write_double(group, "d" + axis, step[i]);
            write_int(group, axis + "_Axis", i);
        }
        std::vector<std::vector<double>> values(11, std::vector<double>(M*M*M*M));
        for (int is=0; is<M; ++is)
        for (int iB=0; iB<M; ++iB)
        for (int iS=0; iS<M; ++iS)
        for (int iQ=0; iQ<M; ++iQ){
            const double s = min[0] + is*step[0];
            const double B = min[1] + iB*step[1];
            const double S = min[2] + iS*step[2];
            const double Q = min[3] + iQ*step[3];
            const std::array<double,11> eos = eos_function(s, B, Q, S);
            const int index = ((is*M + iB)*M + iS)*M + iQ;
            for (int ivar=0; ivar<11; ++ivar) values[ivar][index] = eos[ivar];
        }
        for (int ivar=0; ivar<11; ++ivar)
            group.createDataSet(names[ivar], H5::PredType::NATIVE_DOUBLE, grid)
                 .write(values[ivar].data(), H5::PredType::NATIVE_DOUBLE);
    }
}

TEST(eos_interpolator, linear_table){

    generate_fake_eos();
    std::shared_ptr<cc::SystemState<2>> systemPtr = prepare_system();
    cc::EoS_Interpolator eos("eos_fake.h5");

    double const expectedT = 5.;
    double const expectedmuB = 2*expectedT;
    double const expectedmuQ = 3*expectedT;
    double const expectedmuS = 4*expectedT;
    double const expectede = 5*expectedT;
    double const expectedp = 6*expectedT;
    double const expectedcs2 = 7*expectedT;
    double const expecteddwds = 8*expectedT;
    double const expecteddwdB = 9*expectedT;
    double const expecteddwdQ = 10*expectedT;
    double const expecteddwdS = 11*expectedT;


    systemPtr->allocate_cabana_particles();
    eos.fill_thermodynamics(systemPtr->cabana_particles,2);
    auto particles = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), systemPtr->cabana_particles);
    CREATE_VIEW(host_,particles)

    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::T), expectedT);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::muB), expectedmuB);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::muQ), expectedmuQ);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::muS), expectedmuS);

    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::e), expectede);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::p), expectedp);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::cs2), expectedcs2);

    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::dwds), expecteddwds);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::dwdB), expecteddwdB);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::dwdQ), expecteddwdQ);
    ASSERT_DOUBLE_EQ(host_thermo(0,cc::thermo_info::dwdS), expecteddwdS);


}


// A fake equation of state that tells the four axes apart (different
// coefficients) and has cross terms. Multilinear interpolation reproduces any
// function that is linear in each variable separately, so the expected values
// are exact; a swapped axis, a wrong cell or wrong weights would show.
double multilinear(double s, double B, double S, double Q){
    return 0.5 + s + 2.*B + 3.*S + 5.*Q + 0.25*s*B + 0.125*S*Q;
}
std::array<double,11> asymmetric_eos(double s, double B, double Q, double S){
    const double f = multilinear(s, B, S, Q);
    return {f, 2*f, 3*f, 4*f, 5*f, 6*f, 7*f, 8*f, 9*f, 10*f, 11*f};
}

struct Query { double s, B, S, Q; };

// One particle per query, then fill_thermodynamics.
std::vector<cc::thermodynamic_info> interpolate(cc::EoS_Interpolator& eos,
                                                const std::vector<Query>& queries){
    std::shared_ptr<Settings> settingsPtr = std::make_shared<Settings>();
    std::shared_ptr<cc::SystemState<2>> sysPtr = std::make_shared<cc::SystemState<2>>(settingsPtr);
    sysPtr->t = 2.;
    sysPtr->particles.resize(queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i){
        cc::Particle<2>& p = sysPtr->particles[i];
        p.ID = (int)i;
        p.thermo.s = queries[i].s;
        p.thermo.rhoB = queries[i].B;
        p.thermo.rhoS = queries[i].S;
        p.thermo.rhoQ = queries[i].Q;
    }
    sysPtr->n_particles = (int)queries.size();
    sysPtr->allocate_cabana_particles();
    eos.fill_thermodynamics(sysPtr->cabana_particles, 2.);
    sysPtr->copy_device_to_host();
    std::vector<cc::thermodynamic_info> result;
    for (const auto& p : sysPtr->particles) result.push_back(p.thermo);
    return result;
}

void expect_close(double value, double expected, const std::string& what){
    EXPECT_LE(std::fabs(value - expected), TOL*std::fabs(expected))
        << what << " is " << value << ", expected " << expected;
}

/// Points in different decades of every variable (so in different table tiles),
/// inside cells rather than on grid points.
TEST(eos_interpolator, multilinear_table_across_decades){
    generate_fake_eos("eos_asymmetric.h5", asymmetric_eos, 3);
    cc::EoS_Interpolator eos("eos_asymmetric.h5");
    const std::vector<Query> queries = {
        {1.1,  1.2,  1.4,   1.3},
        {0.05, 3.5,  0.002, 0.7},
        {45.,  0.33, 0.055, 0.0042},
        {7.3,  0.06, 0.6,   2.2},
    };
    const std::vector<cc::thermodynamic_info> out = interpolate(eos, queries);
    for (std::size_t i = 0; i < queries.size(); ++i){
        const Query& q = queries[i];
        const double f = multilinear(q.s, q.B, q.S, q.Q);
        const std::string at = " at point " + std::to_string(i);
        expect_close(out[i].T,    f,     "T"   + at);
        expect_close(out[i].muB,  2.*f,  "muB" + at);
        expect_close(out[i].muQ,  3.*f,  "muQ" + at);
        expect_close(out[i].muS,  4.*f,  "muS" + at);
        expect_close(out[i].e,    5.*f,  "e"   + at);
        expect_close(out[i].p,    6.*f,  "p"   + at);
        expect_close(out[i].cs2,  7.*f,  "cs2" + at);
        expect_close(out[i].dwds, 8.*f,  "dwds" + at);
        expect_close(out[i].dwdB, 9.*f,  "dwdB" + at);
        expect_close(out[i].dwdQ, 10.*f, "dwdQ" + at);
        expect_close(out[i].dwdS, 11.*f, "dwdS" + at);
        expect_close(out[i].w,    11.*f, "w = e + p" + at);
    }
}

/// Densities below the smallest tabulated value (1e-3) are looked up at 1e-3.
TEST(eos_interpolator, small_densities_use_the_table_floor){
    generate_fake_eos("eos_asymmetric.h5", asymmetric_eos, 3);
    cc::EoS_Interpolator eos("eos_asymmetric.h5");
    const std::vector<cc::thermodynamic_info> out =
        interpolate(eos, {{2.0, 1e-5, 0.0, 4e-4}, {2.0, 1e-3, 1e-3, 1e-3}});
    expect_close(out[0].T, out[1].T, "T below the floor");
    expect_close(out[0].e, out[1].e, "e below the floor");
    expect_close(out[1].T, multilinear(2.0, 1e-3, 1e-3, 1e-3), "T at the floor");
}

/// The table stores positive densities only and is read at |rho|. Under
/// rho_a -> -rho_a, T, e, p, cs2 and dw/ds are unchanged, while mu_a and dw/drho_a
/// change sign; the other chemical potentials do not. (Until 2026-11 the signs
/// were not restored, and rhoB < 0 gave muB > 0.)
TEST(eos_interpolator, negative_density_flips_its_chemical_potential){
    generate_fake_eos("eos_asymmetric.h5", asymmetric_eos, 3);
    cc::EoS_Interpolator eos("eos_asymmetric.h5");
    const std::vector<cc::thermodynamic_info> out = interpolate(eos, {
        {1.1,  1.2,  1.4,  1.3},
        {1.1, -1.2,  1.4,  1.3},   // B flipped
        {1.1,  1.2, -1.4,  1.3},   // S flipped
        {1.1, -1.2, -1.4, -1.3},   // all flipped
    });
    const cc::thermodynamic_info& ref = out[0];
    const double sign[4][3] = {{1, 1, 1}, {-1, 1, 1}, {1, -1, 1}, {-1, -1, -1}};
    for (int i = 1; i < 4; ++i){
        const std::string at = " at point " + std::to_string(i);
        expect_close(out[i].T,    ref.T,    "T" + at);
        expect_close(out[i].e,    ref.e,    "e" + at);
        expect_close(out[i].p,    ref.p,    "p" + at);
        expect_close(out[i].cs2,  ref.cs2,  "cs2" + at);
        expect_close(out[i].dwds, ref.dwds, "dw/ds" + at);
        expect_close(out[i].muB,  sign[i][0]*ref.muB,  "muB" + at);
        expect_close(out[i].dwdB, sign[i][0]*ref.dwdB, "dw/drhoB" + at);
        expect_close(out[i].muS,  sign[i][1]*ref.muS,  "muS" + at);
        expect_close(out[i].dwdS, sign[i][1]*ref.dwdS, "dw/drhoS" + at);
        expect_close(out[i].muQ,  sign[i][2]*ref.muQ,  "muQ" + at);
        expect_close(out[i].dwdQ, sign[i][2]*ref.dwdQ, "dw/drhoQ" + at);
    }
}

/// Queries on the last grid point of a tile, and beyond the end of the table.
/// The top tiles are s in [10, 100] and rho in [1, 10]; larger values are
/// looked up in them. The interpolation uses the last cell of the grid, so the
/// last grid point is reproduced and values beyond it are extrapolated
/// linearly; for this multilinear EoS both are exact. (Until 2026-11 such
/// queries took the cell that starts at the last grid point, and read one
/// point past the end of the grid: past the allocation along s, a point in
/// the next row of the table along the other axes.)
TEST(eos_interpolator, last_grid_point_and_beyond){
    generate_fake_eos("eos_asymmetric.h5", asymmetric_eos, 3);
    cc::EoS_Interpolator eos("eos_asymmetric.h5");
    const std::vector<Query> queries = {
        {100., 1.2, 1.4, 1.3},    // last grid point in s
        {150., 1.2, 1.4, 1.3},    // beyond it
        {1.1,  10., 1.4, 1.3},    // last grid point in rhoB
        {1.1,  14., 1.4, 1.3},
        {1.1,  1.2, 10., 1.3},    // rhoS
        {1.1,  1.2, 12., 1.3},
        {1.1,  1.2, 1.4, 10.},    // rhoQ
        {1.1,  1.2, 1.4, 13.},
        {100., 10., 10., 10.},    // the last point of the whole table
    };
    const std::vector<cc::thermodynamic_info> out = interpolate(eos, queries);
    for (std::size_t i = 0; i < queries.size(); ++i){
        const Query& q = queries[i];
        const double f = multilinear(q.s, q.B, q.S, q.Q);
        const std::string at = " at point " + std::to_string(i);
        expect_close(out[i].T,    f,     "T"    + at);
        expect_close(out[i].muB,  2.*f,  "muB"  + at);
        expect_close(out[i].e,    5.*f,  "e"    + at);
        expect_close(out[i].dwdQ, 10.*f, "dwdQ" + at);
    }
}

/// A table whose datasets hold more points than the min, max and step
/// attributes describe cannot be read; the loader says so and stops. (Until
/// 2026-11 it read the larger datasets past the end of its buffers, and
/// utilities/generate_eos_table.cpp wrote such tables: 20 points with a step of
/// a twentieth of the maximum, which the attributes describe as 19 points.)
TEST(eos_interpolatorDeathTest, datasets_must_match_the_grid_attributes){
    generate_fake_eos("eos_wrong_shape.h5", asymmetric_eos, 3, 4);
    EXPECT_DEATH( { cc::EoS_Interpolator eos("eos_wrong_shape.h5"); },
                  "describe 3 x 3 x 3 x 3 points. The table cannot be read" );
}

// The output with every run of whitespace replaced by one space (report()
// wraps long messages).
std::string squeeze(const std::string& text){
    std::string out;
    for (char c : text){
        const bool space = std::isspace(static_cast<unsigned char>(c));
        if (space && (out.empty() || out.back() == ' ')) continue;
        out += space ? ' ' : c;
    }
    return out;
}

/// Queries above the end of the table are reported, once per run and variable,
/// with the advice to regenerate the table over a larger range.
TEST(eos_interpolator, query_above_the_table_is_reported_once){
    generate_fake_eos("eos_asymmetric.h5", asymmetric_eos, 3);
    {
        cc::EoS_Interpolator eos("eos_asymmetric.h5");
        testing::internal::CaptureStdout();
        interpolate(eos, {{150., 1.2, 1.4, 1.3}, {120., 1.2, 1.4, 1.3}, {1.1, 1.2, 1.4, 1.3}});
        const std::string first = squeeze(testing::internal::GetCapturedStdout());
        EXPECT_NE(first.find("WARNING: The pre-inverted EoS table ends at s = 100 fm^-3, but at t = 2 fm/c "
                             "2 of 3 particles are above it (largest s = 150 fm^-3)."), std::string::npos)
            << first;
        EXPECT_NE(first.find("Regenerate the table over a larger range."), std::string::npos) << first;
        EXPECT_EQ(first.find("rho_B ="), std::string::npos) << "rho_B is inside the table: " << first;

        testing::internal::CaptureStdout();
        interpolate(eos, {{150., 1.2, 1.4, 1.3}});
        const std::string second = testing::internal::GetCapturedStdout();
        EXPECT_EQ(second.find("WARNING"), std::string::npos) << "reported twice: " << second;
    }
    {
        cc::EoS_Interpolator eos("eos_asymmetric.h5");
        testing::internal::CaptureStdout();
        interpolate(eos, {{1.1, 1.2, 1.4, 1.3}, {100., 10., 10., 10.}});
        const std::string inside = testing::internal::GetCapturedStdout();
        EXPECT_EQ(inside.find("WARNING"), std::string::npos) << "nothing is above the table: " << inside;
    }
}

// Writes a flat (s, nB) table in the layout load_flat_table reads: /dimensions
// = {Ns, NB} and 2D datasets under /data in MeV units, on log-spaced axes
// s = 1 ... 100 fm^-3 and nB = 0.01 ... 10 fm^-3. mu_S and mu_Q are nonzero,
// as in a strangeness-neutral table.
void generate_flat_eos(const std::string& filename){
    const int Ns = 5, NB = 4;
    const double hc = constants::hbarc_MeVfm, hc3 = hc*hc*hc;
    std::vector<std::vector<double>> data(12, std::vector<double>(Ns*NB));
    const char* names[12] = {"s", "nB", "T", "muB", "muS", "muQ", "e", "p", "cs2",
                             "chiTT", "chiTB", "chiBB"};
    for (int is = 0; is < Ns; ++is)
    for (int iB = 0; iB < NB; ++iB){
        const int i = is*NB + iB;
        const double s = std::pow(10., 0.5*is), nB = std::pow(10., -2. + iB);   // fm^-3
        const double values[12] = {s*hc3, nB*hc3, 150. + 10.*is + iB, 50. + 5.*iB, 20., -10.,
                                   1.e9*(1. + is) + 1.e7*iB, 3.e8*(1. + is) + 1.e6*iB, 0.3,
                                   3., 1., 2.};
        for (int k = 0; k < 12; ++k) data[k][i] = values[k];
    }
    H5::H5File file(filename, H5F_ACC_TRUNC);
    const hsize_t two[1] = {2};
    const int dims[2] = {Ns, NB};
    file.createDataSet("dimensions", H5::PredType::NATIVE_INT, H5::DataSpace(1, two))
        .write(dims, H5::PredType::NATIVE_INT);
    H5::Group group = file.createGroup("data");
    const hsize_t grid[2] = {(hsize_t) Ns, (hsize_t) NB};
    for (int k = 0; k < 12; ++k)
        group.createDataSet(names[k], H5::PredType::NATIVE_DOUBLE, H5::DataSpace(2, grid))
             .write(data[k].data(), H5::PredType::NATIVE_DOUBLE);
}

/// Flat (s, nB) table: under nB -> -nB every chemical potential and dw/dnB change
/// sign (mu_S and mu_Q are functions of nB here), on the device and on the host.
TEST(eos_interpolator, flat_table_negative_density_flips_every_chemical_potential){
    generate_flat_eos("eos_flat.h5");
    cc::EoS_Interpolator eos("eos_flat.h5");
    ASSERT_TRUE(eos.is_flat());
    const std::vector<cc::thermodynamic_info> out = interpolate(eos, {{2.0, 0.5, 0., 0.},
                                                                      {2.0, -0.5, 0., 0.}});
    cc::thermodynamic_info host_plus, host_minus;
    ASSERT_TRUE(eos.query_thermo_host(2.0, 0.5, host_plus));
    ASSERT_TRUE(eos.query_thermo_host(2.0, -0.5, host_minus));
    const cc::thermodynamic_info* plus[2]  = {&out[0], &host_plus};
    const cc::thermodynamic_info* minus[2] = {&out[1], &host_minus};
    for (int k = 0; k < 2; ++k){
        const std::string where = k == 0 ? " (device)" : " (host)";
        const cc::thermodynamic_info& a = *plus[k];
        const cc::thermodynamic_info& b = *minus[k];
        ASSERT_GT(a.muS, 0.) << where;
        expect_close(b.T,    a.T,     "T" + where);
        expect_close(b.e,    a.e,     "e" + where);
        expect_close(b.p,    a.p,     "p" + where);
        expect_close(b.dwds, a.dwds,  "dw/ds" + where);
        expect_close(b.muB,  -a.muB,  "muB" + where);
        expect_close(b.muS,  -a.muS,  "muS" + where);
        expect_close(b.muQ,  -a.muQ,  "muQ" + where);
        expect_close(b.dwdB, -a.dwdB, "dw/dnB" + where);
    }
}

/// The host-side lookup of the flat table (used for the initial condition)
/// holds values above the table at the edge, and says so once.
TEST(eos_interpolator, flat_table_host_query_above_the_table_is_reported){
    generate_flat_eos("eos_flat.h5");
    cc::EoS_Interpolator eos("eos_flat.h5");
    cc::thermodynamic_info edge, above, again;
    ASSERT_TRUE(eos.query_thermo_host(100., 0.5, edge));
    testing::internal::CaptureStdout();
    ASSERT_TRUE(eos.query_thermo_host(200., 0.5, above));
    const std::string first = squeeze(testing::internal::GetCapturedStdout());
    testing::internal::CaptureStdout();
    ASSERT_TRUE(eos.query_thermo_host(300., 0.5, again));
    const std::string second = testing::internal::GetCapturedStdout();
    EXPECT_NE(first.find("WARNING: The pre-inverted EoS table ends at s = 100 fm^-3, but a lookup on "
                         "the host (initial condition or freeze-out setup) asked for s = 200 fm^-3. "
                         "It used the value at the edge of the table."), std::string::npos) << first;
    EXPECT_EQ(first.find("NTBL_s"), std::string::npos) << "the tiled-format hint is not for flat tables";
    EXPECT_EQ(second.find("WARNING"), std::string::npos) << "reported twice: " << second;
    expect_close(above.T, edge.T, "T above the table");
}

int main(int argc, char* argv[]) {

    // Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);

    // Run the tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}