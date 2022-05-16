#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cmath>


#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace fractions {

  const double one_half = 1. / 2.;
  const double one_third = 1. / 3.;
  const double one_sixth = 1. / 6.;
  const double three_halves = 3. / 2.;

  const double pi = M_PI;
  const double pi_d_12 = M_PI / 12.;
  const double pi_d_6 = M_PI / 6.;
  const double pi_d_4 = M_PI / 4.;
  const double pi_d_3 = M_PI / 3.;
  const double fivepi_d_12 = M_PI * 5. / 12.;
  const double pi_d_2 = M_PI / 2.;
  const double sevenpi_d_12 = M_PI * 7. / 12.;
  const double twopi_d_3 = M_PI * 2. / 3.;
  const double threepi_d_4 = M_PI * 3. / 4.;
  const double fivepi_d_6 = M_PI * 5. / 6.;
  const double elevenpi_d_12 = M_PI * 11. / 12.;
  const double twopi = M_PI * 2.;
  const double fourpi = M_PI * 4.;

}

namespace limb_darkening {

  const int quadratic = 0;
  const int non_linear = 1;

}

namespace intersections {

  const int T_plus = 1;
  const int T_minus = 0;

  const int dT_dtheta_plus = 1;
  const int dT_dtheta_minus = 0;

  const int planet = 0;
  const int star = 1;
  const int beyond = 2;

}

namespace tolerance {

  const double unit_circle_lo = 1. - 1.e-7;
  const double unit_circle_hi = 1. + 1.e-7;

  const double intersect_associate = 1.e-7;

}

namespace convolution {}

namespace legendre {}

#endif
