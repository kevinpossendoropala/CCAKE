#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <ostream>
#include <unistd.h>
#include <gtest/gtest.h>

#include "input.h"

//TODO: Write death tests for the case of missing fields in the input file

namespace fs = std::filesystem;
#define TOL 1.E-14

//Forward declarations of helper functions
void create_test_input();
std::string sample_yaml();

/// \brief Happy scenartio: Create a sample input and check if the constructor
/// reads it correctly
TEST (InputTest, Constructor){
  create_test_input();
  fs::create_directory("test_result_output");

  std::array<std::string,3> fake_args = {"InputTest.Constructor", "sample_input.yaml", "test_result_output"};
  char** man_args;
  man_args = new char*[3];
  for(int i=0; i < 3; i++) {
    man_args[i] = fake_args[i].data();
  }

  ccake::Input in(fake_args.size(), man_args);

  //Check if the input file was read correctly

  //IC parameters
  EXPECT_EQ( in.settingsPtr->IC_type, "ccake" );
  EXPECT_EQ( in.settingsPtr->IC_file, "initial_conditions/ccake_trento.dat" );
  EXPECT_DOUBLE_EQ( in.settingsPtr->t0, 0.6);
  EXPECT_EQ( in.settingsPtr->dim, 2 );
  EXPECT_TRUE( in.settingsPtr->input_as_entropy );

  //Simulation parameters
  EXPECT_DOUBLE_EQ( in.settingsPtr->dt, 0.03 );
  EXPECT_DOUBLE_EQ( in.settingsPtr->hT, 0.5 );
  EXPECT_DOUBLE_EQ( in.settingsPtr->hEta, 0.5 );
  EXPECT_EQ( in.settingsPtr->kernel_type, "quartic_spline" );
  EXPECT_DOUBLE_EQ( in.settingsPtr->e_cutoff, .1 );
  EXPECT_TRUE( in.settingsPtr->buffer_event );
  EXPECT_TRUE (in.settingsPtr->circular_buffer);
  EXPECT_DOUBLE_EQ( in.settingsPtr->padding_thickness, .2);

  //EOS parameters
  EXPECT_EQ( in.settingsPtr->eos_type, "table" );
  EXPECT_EQ( in.settingsPtr->eos_path, "Houston" );

  //Particlization settings
  EXPECT_TRUE( in.settingsPtr->particlization_enabled);
  // (Settings::Freeze_Out_Type no longer exists, so its check was dropped)
  EXPECT_DOUBLE_EQ( in.settingsPtr->Freeze_Out_Temperature, 140. );

  //Hydro values
  EXPECT_FALSE( in.settingsPtr->baryon_charge_enabled );
  EXPECT_FALSE( in.settingsPtr->strange_charge_enabled );
  EXPECT_FALSE( in.settingsPtr->electric_charge_enabled );

  //Shear parameters
  EXPECT_EQ( in.settingsPtr->etaMode, "constant" );
  EXPECT_DOUBLE_EQ( in.settingsPtr->constant_eta_over_s, 0.2 );
  EXPECT_EQ( in.settingsPtr->shearRelaxMode, "default" );

  //Bulk parameters
  EXPECT_EQ( in.settingsPtr->zetaMode, "cs2_dependent" );
  EXPECT_DOUBLE_EQ( in.settingsPtr->constant_zeta_over_s, 0.0 );
  EXPECT_EQ( in.settingsPtr->bulkRelaxMode, "default" );
  EXPECT_DOUBLE_EQ( in.settingsPtr->cs2_dependent_zeta_A, 1.67552 );
  EXPECT_DOUBLE_EQ( in.settingsPtr->cs2_dependent_zeta_p, 2.0 );
  EXPECT_TRUE( in.settingsPtr->modulate_zeta_with_tanh);







  fs::remove_all("test_result_output");
  //fs::remove("sample_input.yaml");

}

/// \brief This is a scenario where the output directory does not exists. The
/// constructor should create it.
TEST (InputTest, ConstructorMkDir){
  create_test_input();

  std::array<std::string,3> fake_args = {"InputTest.Constructor", "sample_input.yaml", "test_result_output"};
  char** man_args;
  man_args = new char*[3];
  for(int i=0; i < 3; i++) {
    man_args[i] = fake_args[i].data();
  }

  ccake::Input in(fake_args.size(), man_args);


  EXPECT_TRUE( fs::exists("test_result_output"));

  fs::remove_all("test_result_output");
  //fs::remove("sample_input.yaml");

}

/// \brief This is a scenario where the input file does not exists. The program
/// should abort.
TEST (InputDeathTest, Constructor){

  fs::create_directory("test_result_output");

  std::array<std::string,3> fake_args = {"InputTest.Constructor", "invalid.yaml", "test_result_output"};
  char** man_args;
  man_args = new char*[3];
  for(int i=0; i < 3; i++) {
    man_args[i] = fake_args[i].data();
  }
  EXPECT_EXIT( ccake::Input(fake_args.size(), man_args) , testing::ExitedWithCode(EXIT_FAILURE), ".*");

}

/// \brief Helper function that generates a sample input file, as it should be
/// expected by the code
std::string sample_yaml(){
  return R"(---
initial_conditions:
  type: "ccake" # Possible values are: ["ccake", "iccing", "hdf5"]
  file: "initial_conditions/ccake_trento.dat" # Path to the initial conditions file
  t0: 0.6 # tau0 = \sqrt(t^2-z^2) in fm/c. Must match the value for initial condition file.
  dimension: 2 # (1+1)D, (2+1)D or (3+1)D simulation
  input_as_entropy: true # if false, input will be interpreted as energy density
parameters:
  dt: 0.03 # time-step of the simulation in fm/c
  h_T:  0.500000  # Smothing parameter inside kernel, in the transverse direction (in fm)
  h_eta:  0.500000  # Smothing parameter inside kernel in the longitudinal direction
  kernel_type: "quartic_spline" # Kernel type. Possible values are: ["cubic_spline"]
  energy_cutoff: 0.1 # GeV/fm^3 - Particles with less than this energy will be deleted before starting the evolution
  buffer_particles:
    enabled: true # Set to true to include buffer particles. May help stability.
    circular: true # whether to buffer with entire grid or just circular padding.
    padding_thickness: .2 # if circular buffer is true, specifies the fractional amount
                          # of padding to add beyond the point with maximum distance from origin
eos:
  online_inverter_enabled: true # without it a pre-inverted table path is required
  type: "table" # Possible values are: ["table", "conformal"]
  path: "Houston" # If type is "table", the path to the equation of state table.
particlization:
  enabled: true # Set to true to output particlization hypersurface
  type: "fixed_T" # Possible values are: ["fixed_T"]
  T: 140.00000 # Temperature of the particlization hypersurface (in MeV)
hydro:
  baryon_charge_enabled: false # Set to false to disable baryon charge evolution
  strange_charge_enabled: false # Set to false to disable strange charge evolution
  electric_charge_enabled: false # Set to false to disable electric charge evolution
  viscous_parameters:
    shear:
      mode: "constant" # Possible values are: ["constant", "disabled"]
      constant_eta_over_s: 0.20 # Value of the constant shear viscosity
      relaxation_mode: "default" # How shear relaxation coeffiecient is computed. Possible values are: ["default"]
    bulk:
      mode: "cs2_dependent"
      constant_zeta_over_s: 0.0
      cs2_dependent_zeta_A: 1.67552
      cs2_dependent_zeta_p: 2.0
      relaxation_mode: "default"
      modulate_with_tanh: true #If true, will modulate the bulk viscosity
                               #with a tanh function below transition temperature
    diffusion:
      mode: "disabled" # optional block
  source:
    type: "disabled" # optional block
output:
  print_conservation_state: true # optional block
)";
}

void create_test_input(){
  std::ofstream sample_input("sample_input.yaml",std::ios::out);
  sample_input << sample_yaml();
  sample_input.close();
}



//------------------------------------------------------------------------------
// Keys that were removed from CCAKE stop the run with a message, so that an old
// configuration cannot silently run without them.
//------------------------------------------------------------------------------
namespace {

/// The sample configuration with one line replaced (or a line inserted after `after`).
std::string with_change(const std::string& find, const std::string& replace_by){
  std::string yaml = sample_yaml();
  const std::size_t at = yaml.find(find);
  if (at == std::string::npos) throw std::runtime_error("sample has no '" + find + "'");
  yaml.replace(at, find.size(), replace_by);
  return yaml;
}

void read_config(const std::string& yaml){
  std::ofstream("changed_input.yaml") << yaml;
  fs::create_directory("test_result_output");
  std::array<std::string,3> fake_args = {"InputTest", "changed_input.yaml", "test_result_output"};
  char* man_args[3];
  for (int i = 0; i < 3; i++) man_args[i] = fake_args[i].data();
  ccake::Input in(3, man_args);
}

const std::string T_line = "  T: 140.00000 # Temperature of the particlization hypersurface (in MeV)\n";
const std::string shear_mode = R"(mode: "constant" # Possible values are: ["constant", "disabled"])";
const std::string shear_relax = R"(relaxation_mode: "default" # How shear relaxation coeffiecient is computed)";
const std::string bulk_mode = R"(mode: "cs2_dependent")";
const std::string diffusion_mode = R"(      mode: "disabled" # optional block)";
const std::string source_type = R"(    type: "disabled" # optional block)";
const std::string output_line = R"(  print_conservation_state: true # optional block)";

} // namespace

TEST (InputDeathTest, RemovedParticlizationCriterion){
  EXPECT_DEATH( read_config(with_change(T_line, T_line + "  criterion: \"temperature\"\n")),
                "particlization.criterion was removed" );
}

TEST (InputDeathTest, RemovedHolographicShearMode){
  EXPECT_DEATH( read_config(with_change(shear_mode, "mode: \"holographic\"")),
                "shear mode 'holographic' was removed" );
}

TEST (InputDeathTest, RemovedHolographicShearRelaxation){
  EXPECT_DEATH( read_config(with_change(shear_relax, "relaxation_mode: \"holographic\" #")),
                "shear relaxation_mode 'holographic' was removed" );
}

TEST (InputDeathTest, RemovedGaussianBulkMode){
  EXPECT_DEATH( read_config(with_change(bulk_mode, "mode: \"gaussian_muB\"")),
                "bulk mode 'gaussian_muB' was removed" );
}

/// The kappa_B edge tapers and the smoothed coth bracket.
TEST (InputDeathTest, RemovedDiffusionKeys){
  for (const std::string key : {"taper_muB_start", "taper_muB_end", "taper_T_min", "bracket_eps"}){
    SCOPED_TRACE(key);
    EXPECT_DEATH( read_config(with_change(diffusion_mode, diffusion_mode + "\n      " + key + ": 0.5")),
                  "diffusion\\." + key + " was removed" );
  }
}

/// Renamed in 2025-06 (before, an old config ran with the default matrix).
TEST (InputDeathTest, RenamedDiffusionKappaMatrix){
  EXPECT_DEATH( read_config(with_change(diffusion_mode, diffusion_mode
                  + "\n      kappa_matrix: [[1,0,0],[0,1,0],[0,0,1]]")),
                "kappa_matrix was renamed to hydro.viscous_parameters.diffusion.constant_kappa_over_T2" );
}

/// Options of the source class before its rework (2026-10); entropy_ref was
/// silently ignored until 2026-11.
TEST (InputDeathTest, RemovedSourceKeys){
  for (const std::string key : {"propagate_budget", "fluidization_threshold", "fluidization_criterion",
                                "fluidization_delay", "entropy_ref"}){
    SCOPED_TRACE(key);
    EXPECT_DEATH( read_config(with_change(source_type, source_type + "\n    " + key + ": 1")),
                  "hydro\\.source\\." + key + " was removed" );
  }
}

TEST (InputDeathTest, RenamedSourceMaxBoost){
  EXPECT_DEATH( read_config(with_change(source_type, source_type + "\n    max_boost: 2.0")),
                "max_boost was renamed to hydro.source.max_renormalization_factor" );
}

/// Removed with the 3D jets (2026-06); silently ignored until 2026-11.
TEST (InputDeathTest, RemovedJetPhiBins){
  EXPECT_DEATH( read_config(with_change(source_type, source_type + "\n  jets:\n    Phi_bins: 7")),
                "hydro\\.jets\\.Phi_bins was removed" );
}

/// The Theorem-1 causality check.
TEST (InputDeathTest, RemovedOutputKeys){
  for (const std::string key : {"causality_theorem1", "causality_theorem1_angles", "causality_zero_baryon"}){
    SCOPED_TRACE(key);
    EXPECT_DEATH( read_config(with_change(output_line, output_line + "\n  " + key + ": true")),
                  "output\\." + key + " was removed" );
  }
}

//------------------------------------------------------------------------------
// A mode that no part of CCAKE acts on (a misspelling, a wrong case) stops the
// run; every mode the code does act on is accepted.
//------------------------------------------------------------------------------
namespace {

struct ModeSetting {
  const char* key;                    // as printed in the message
  std::string line;                   // the line of the sample that sets it
  std::string set_to;                 // that line with the value replaced by %
  std::vector<std::string> known;     // the values the code acts on
};

const std::vector<ModeSetting> mode_settings = {
  {"initial_conditions.type", R"(type: "ccake")", R"(type: "%")",
   {"ccake", "ICCING", "freezein"}},
  {"hydro.viscous_parameters.shear.mode", shear_mode, R"(mode: "%")",
   {"disabled", "default", "JakiParam", "constant", "LinearMus", "piecewise_muB", "interpolate"}},
  {"hydro.viscous_parameters.shear.relaxation_mode", shear_relax, R"(relaxation_mode: "%" #)",
   {"default", "minVal", "Gubser"}},
  {"hydro.viscous_parameters.bulk.mode", bulk_mode, R"(mode: "%")",
   {"default", "constant", "DNMR", "interpolate", "cs2_dependent"}},
  {"hydro.viscous_parameters.bulk.relaxation_mode", "relaxation_mode: \"default\"\n",
   "relaxation_mode: \"%\"\n", {"default", "DNMR"}},
};

std::string with_mode(const ModeSetting& m, const std::string& value){
  std::string line = m.set_to;
  line.replace(line.find('%'), 1, value);
  return with_change(m.line, line);
}

void read_config_and_exit(const std::string& yaml){
  read_config(yaml);
  std::exit(0);
}

} // namespace

TEST (InputDeathTest, UnknownModesAreRefused){
  for (const ModeSetting& m : mode_settings){
    SCOPED_TRACE(m.key);
    // a wrong case, as in "gubser" for "Gubser"
    std::string wrong_case = m.known.back();
    wrong_case[0] = std::isupper(wrong_case[0]) ? std::tolower(wrong_case[0]) : std::toupper(wrong_case[0]);
    for (const std::string& value : {std::string("no_such_mode"), wrong_case})
      EXPECT_DEATH( read_config(with_mode(m, value)),
                    std::string(m.key) + " '" + value + "' is not known" );
  }
}

TEST (InputDeathTest, KnownModesAreAccepted){
  for (const ModeSetting& m : mode_settings)
    for (const std::string& value : m.known){
      SCOPED_TRACE(std::string(m.key) + " = " + value);
      EXPECT_EXIT( read_config_and_exit(with_mode(m, value)), testing::ExitedWithCode(0), "" );
    }
}

//------------------------------------------------------------------------------
// Settings that cannot work together stop the run at start-up.
//------------------------------------------------------------------------------
namespace {

/// Some of these messages go to standard output, and a death test reads only
/// standard error: point standard output at standard error (this runs only in
/// the dying child process). The messages are wrapped at about 80 characters, so
/// each expected text below is a piece of one printed line.
void read_config_all_to_stderr(const std::string& yaml){
  std::cout.flush();
  dup2(STDERR_FILENO, STDOUT_FILENO);
  read_config(yaml);
}

const std::string baryon_off = "baryon_charge_enabled: false";

} // namespace

TEST (InputDeathTest, CothDiffusionNeedsBaryonOnlyCharges){
  EXPECT_EXIT( read_config_all_to_stderr(with_change(diffusion_mode, "      mode: \"kappaB_coth\"")),
               testing::ExitedWithCode(EXIT_FAILURE),
               "diffusion mode 'kappaB_coth' requires a baryon-only charge" );
}

TEST (InputDeathTest, HolographicDiffusionNeedsBaryonOnlyCharges){
  EXPECT_EXIT( read_config_all_to_stderr(with_change(diffusion_mode, "      mode: \"holographic_table\"")),
               testing::ExitedWithCode(EXIT_FAILURE),
               "diffusion mode 'holographic_table' requires a baryon-only charge" );
}

TEST (InputDeathTest, HolographicDiffusionNeedsItsTable){
  std::string yaml = with_change(diffusion_mode, "      mode: \"holographic_table\"");
  yaml.replace(yaml.find(baryon_off), baryon_off.size(), "baryon_charge_enabled: true");
  EXPECT_EXIT( read_config_all_to_stderr(yaml), testing::ExitedWithCode(EXIT_FAILURE),
               "hydro/viscous_parameters/diffusion/sigmaB_table_path pointing at" );
}

TEST (InputDeathTest, UnknownDiffusionCriticalScalingMode){
  EXPECT_DEATH( read_config(with_change(diffusion_mode, diffusion_mode
                  + "\n      critical_scaling: true\n      critical_scaling_mode: \"xi_squared\"")),
                "critical_scaling_mode 'xi_squared' is not 'gaussian_suppression' or 'xi_power'" );
}

TEST (InputDeathTest, XiPowerDiffusionScalingNeedsLocalXi0){
  EXPECT_DEATH( read_config(with_change(diffusion_mode, diffusion_mode
                  + "\n      critical_scaling: true\n      critical_scaling_mode: \"xi_power\"")),
                "'xi_power' requires bulk critical_scaling_local_xi0: true" );
}

/// The hydro.viscous_parameters.diffusion, hydro.source and output blocks are
/// optional: a configuration without them is read with the defaults. (Until
/// 2026-11 the checks for removed keys indexed into the missing block and
/// yaml-cpp threw "invalid node".)
TEST (InputTest, OptionalBlocksCanBeLeftOut){
  std::string yaml = sample_yaml();
  yaml = yaml.substr(0, yaml.find("    diffusion:"));
  ASSERT_EQ(yaml.find("source:"), std::string::npos);
  ASSERT_EQ(yaml.find("output:"), std::string::npos);
  EXPECT_NO_THROW( read_config(yaml) );
}

int main(int argc, char* argv[]) {
 
    // Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);

    // Run the tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}