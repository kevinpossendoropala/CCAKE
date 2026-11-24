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

//Template instantiations
template class Particle<1>;
template class Particle<2>;
template class Particle<3>;

/// @brief Default constructor
template<unsigned int D>
Particle<D>::Particle() { input.s = 0.0; }

