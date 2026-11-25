#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "particle.h"

using namespace ccake;


/// @brief Default constructor
template<unsigned int D>
Particle<D>::Particle() { input.s = 0.0; }

// Explicit instantiations. They stay after every member definition above: an explicit
// instantiation only instantiates the members that are already defined (GCC defers
// it to the end of the file, Clang does not).
namespace ccake {
template class Particle<1>;
template class Particle<2>;
template class Particle<3>;
}
