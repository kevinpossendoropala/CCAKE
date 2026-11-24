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
template class ccake::Particle<1>;
template class ccake::Particle<2>;
template class ccake::Particle<3>;

/// @brief Default constructor
template<unsigned int D>
Particle<D>::Particle() { input.s = 0.0; }

