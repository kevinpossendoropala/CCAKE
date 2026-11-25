// The SPH kernel is defined, inline, in include/kernel.h. This file only provides
// out-of-line copies for the three dimensions; tools that compile it directly
// (utilities/wvt_ic_generator, utilities/systemstate_interpolator) keep working.
#include "kernel.h"

namespace ccake
{
template class SPHkernel<1>;
template class SPHkernel<2>;
template class SPHkernel<3>;
}
