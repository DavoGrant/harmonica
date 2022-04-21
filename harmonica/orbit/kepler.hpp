#ifndef _EXOPLANET_CONSTANTS_H_
#define _EXOPLANET_CONSTANTS_H_

#include <algorithm>
#include <cmath>

namespace exoplanet {

#ifdef __CUDACC__
#define EXOPLANET_INLINE_OR_DEVICE __host__ __device__

template <class T>
EXOPLANET_INLINE_OR_DEVICE void swap(T& a, T& b) {
  T c(a);
  a = b;
  b = c;
}

#else
#define EXOPLANET_INLINE_OR_DEVICE inline

template <class T>
EXOPLANET_INLINE_OR_DEVICE void swap(T& a, T& b) {
  std::swap(a, b);
}

template <class T>
inline void sincos(const T& x, T* sx, T* cx) {
  *sx = sin(x);
  *cx = cos(x);
}
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

const double one_third = 1. / 3;
const double one_sixth = 1. / 6;

const double if3 = 1. / 6;
const double if5 = 1. / (6. * 20);
const double if7 = 1. / (6. * 20 * 42);
const double if9 = 1. / (6. * 20 * 42 * 72);
const double if11 = 1. / (6. * 20 * 42 * 72 * 110);
const double if13 = 1. / (6. * 20 * 42 * 72 * 110 * 156);
const double if15 = 1. / (6. * 20 * 42 * 72 * 110 * 156 * 210);

const double pi = M_PI;
const double pi_d_12 = M_PI / 12;
const double pi_d_6 = M_PI / 6;
const double pi_d_4 = M_PI / 4;
const double pi_d_3 = M_PI / 3;
const double fivepi_d_12 = M_PI * 5. / 12;
const double pi_d_2 = M_PI / 2;
const double sevenpi_d_12 = M_PI * 7. / 12;
const double twopi_d_3 = M_PI * 2. / 3;
const double threepi_d_4 = M_PI * 3. / 4;
const double fivepi_d_6 = M_PI * 5. / 6;
const double elevenpi_d_12 = M_PI * 11. / 12;
const double twopi = M_PI * 2;
const double fourpi = M_PI * 4;

template <typename T>
EXOPLANET_INLINE_OR_DEVICE int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

}  // namespace exoplanet

#endif

#ifndef _EXOPLANET_KEPLER_H_
#define _EXOPLANET_KEPLER_H_

// C code to compute the eccentric anomaly, its sine and cosine, from
// an input mean anomaly and eccentricity.  Timothy D. Brandt wrote
// the code; the algorithm is based on Raposo-Pulido & Pelaez, 2017,
// MNRAS, 467, 1702.  Computational cost is equivalent to around 3
// trig calls (sine, cosine) in tests as of August 2020.  This can be
// further reduced if using many mean anomalies at fixed eccentricity;
// the code would need some modest refactoring in that case.  Accuracy
// should be within a factor of a few of machine epsilon in E-ecc*sinE
// and in cosE up to at least ecc=0.999999.

#include <cmath>
#include <cstdlib>

//#include "constants.h"

namespace exoplanet {
namespace kepler {

// Evaluate sine with a series expansion.  We can guarantee that the
// argument will be <=pi/4, and this reaches double precision (within
// a few machine epsilon) at a signficantly lower cost than the
// function call to sine that obeys the IEEE standard.
template <typename Scalar>
inline Scalar shortsin(const Scalar &x) {
  Scalar x2 = x * x;
  return x *
         (1 - x2 * (if3 -
                    x2 * (if5 - x2 * (if7 - x2 * (if9 - x2 * (if11 - x2 * (if13 - x2 * if15)))))));
}

// Modulo 2pi: works best when you use an increment so that the
// argument isn't too much larger than 2pi.
template <typename Scalar>
inline Scalar MAmod(const Scalar &M_in) {
  if (M_in < twopi && M_in >= 0) return M_in;

  if (M_in >= twopi) {
    const Scalar M = M_in - twopi;
    if (M > twopi)
      return fmod(M, Scalar(twopi));
    else
      return M;
  } else {
    const Scalar M = M_in + twopi;
    if (M < 0)
      return fmod(M, Scalar(twopi)) + twopi;
    else
      return M;
  }
}

// Use the second-order series expanion in Raposo-Pulido & Pelaez
// (2017) in the singular corner (eccentricity close to 1, mean
// anomaly close to zero).
template <typename Scalar>
inline Scalar EAstart(const Scalar &M, const Scalar &ecc) {
  const Scalar ome = 1. - ecc;
  const Scalar sqrt_ome = sqrt(ome);

  const Scalar chi = M / (sqrt_ome * ome);
  const Scalar Lam = sqrt(8 + 9 * chi * chi);
  const Scalar S = cbrt(Lam + 3 * chi);
  const Scalar sigma = 6 * chi / (2 + S * S + 4. / (S * S));
  const Scalar s2 = sigma * sigma;
  const Scalar s4 = s2 * s2;

  const Scalar denom = 1.0 / (s2 + 2);
  const Scalar E =
      sigma * (1 + s2 * ome * denom *
                       ((s2 + 20) / 60. +
                        s2 * ome * denom * denom * (s2 * s4 + 25 * s4 + 340 * s2 + 840) / 1400));

  return E * sqrt_ome;
}

// Calculate the eccentric anomaly, its sine and cosine, using a
// variant of the algorithm suggested in Raposo-Pulido & Pelaez (2017)
// and used in Brandt et al. (2020).  Use the series expansion above
// to generate an initial guess in the singular corner and use a
// fifth-order polynomial to get the initial guess otherwise.  Use
// series and square root calls to evaluate sine and cosine, and
// update their values using series.  Accurate to better than 1e-15 in
// E-ecc*sin(E)-M at all mean anomalies and at eccentricies up to
// 0.999999.
template <typename Scalar>
inline void calcEA(const Scalar &M, const Scalar &ecc, Scalar *sinE,
                                       Scalar *cosE) {
  const Scalar g2s_e = 0.2588190451025207623489 * ecc;
  const Scalar g3s_e = 0.5 * ecc;
  const Scalar g4s_e = 0.7071067811865475244008 * ecc;
  const Scalar g5s_e = 0.8660254037844386467637 * ecc;
  const Scalar g6s_e = 0.9659258262890682867497 * ecc;

  Scalar bounds[13];
  Scalar EA_tab[9];

  int k;
  Scalar MA, EA, sE, cE, x, y;
  Scalar B0, B1, B2, dx, idx;
  int MAsign = 1;
  Scalar one_over_ecc = 1e17;
  if (ecc > 1e-17) one_over_ecc = 1. / ecc;

  MA = MAmod(M);
  if (MA > pi) {
    MAsign = -1;
    MA = twopi - MA;
  }

  // Series expansion
  if (2 * MA + 1 - ecc < 0.2) {
    EA = EAstart(MA, ecc);
  } else {
    // Polynomial boundaries given in Raposo-Pulido & Pelaez
    bounds[0] = 0;
    bounds[1] = pi_d_12 - g2s_e;
    bounds[2] = pi_d_6 - g3s_e;
    bounds[3] = pi_d_4 - g4s_e;
    bounds[4] = pi_d_3 - g5s_e;
    bounds[5] = fivepi_d_12 - g6s_e;
    bounds[6] = pi_d_2 - ecc;
    bounds[7] = sevenpi_d_12 - g6s_e;
    bounds[8] = twopi_d_3 - g5s_e;
    bounds[9] = threepi_d_4 - g4s_e;
    bounds[10] = fivepi_d_6 - g3s_e;
    bounds[11] = elevenpi_d_12 - g2s_e;
    bounds[12] = pi;

    /* Which interval? */
    for (k = 11; k > 0; k--) {
      if (MA > bounds[k]) break;
    }
    // if (k < 0) k = 0;

    // Values at the two endpoints.
    EA_tab[0] = k * pi_d_12;
    EA_tab[6] = (k + 1) * pi_d_12;

    // First two derivatives at the endpoints. Left endpoint first.
    int sign = (k >= 6) ? 1 : -1;

    x = 1 / (1 - ((6 - k) * pi_d_12 + sign * bounds[abs(6 - k)]));
    y = -0.5 * (k * pi_d_12 - bounds[k]);
    EA_tab[1] = x;
    EA_tab[2] = y * x * x * x;

    x = 1 / (1 - ((5 - k) * pi_d_12 + sign * bounds[abs(5 - k)]));
    y = -0.5 * ((k + 1) * pi_d_12 - bounds[k + 1]);
    EA_tab[7] = x;
    EA_tab[8] = y * x * x * x;

    // Solve a matrix equation to get the rest of the coefficients.
    idx = 1 / (bounds[k + 1] - bounds[k]);

    B0 = idx * (-EA_tab[2] - idx * (EA_tab[1] - idx * pi_d_12));
    B1 = idx * (-2 * EA_tab[2] - idx * (EA_tab[1] - EA_tab[7]));
    B2 = idx * (EA_tab[8] - EA_tab[2]);

    EA_tab[3] = B2 - 4 * B1 + 10 * B0;
    EA_tab[4] = (-2 * B2 + 7 * B1 - 15 * B0) * idx;
    EA_tab[5] = (B2 - 3 * B1 + 6 * B0) * idx * idx;

    // Now use the coefficients of this polynomial to get the initial guess.
    dx = MA - bounds[k];
    EA =
        EA_tab[0] +
        dx * (EA_tab[1] + dx * (EA_tab[2] + dx * (EA_tab[3] + dx * (EA_tab[4] + dx * EA_tab[5]))));
  }

  // Sine and cosine initial guesses using series
  if (EA < pi_d_4) {
    sE = shortsin(EA);
    cE = sqrt(1 - sE * sE);
  } else if (EA > threepi_d_4) {
    sE = shortsin(pi - EA);
    cE = -sqrt(1 - sE * sE);
  } else {
    cE = shortsin(pi_d_2 - EA);
    sE = sqrt(1 - cE * cE);
  }

  Scalar num, denom, dEA;

  // Halley's method to update E
  num = (MA - EA) * one_over_ecc + sE;
  denom = one_over_ecc - cE;
  dEA = num * denom / (denom * denom + 0.5 * sE * num);

  // Use series to update sin and cos
  if (ecc < 0.78 || MA > 0.4) {
    // *E = MAsign * (EA + dEA);
    *sinE = MAsign * (sE * (1 - 0.5 * dEA * dEA) + dEA * cE);
    *cosE = cE * (1 - 0.5 * dEA * dEA) - dEA * sE;

  } else {
    // Use Householder's third order method to guarantee performance
    // in the singular corners
    dEA = num / (denom + dEA * (0.5 * sE + one_sixth * cE * dEA));
    // *E = MAsign * (EA + dEA);
    *sinE = MAsign * (sE * (1 - 0.5 * dEA * dEA) + dEA * cE * (1 - dEA * dEA * one_sixth));
    *cosE = cE * (1 - 0.5 * dEA * dEA) - dEA * sE * (1 - dEA * dEA * one_sixth);
  }

  return;
}

template <typename Scalar>
inline void to_f(const Scalar &ecc, const Scalar &ome, Scalar *sinf,
                                     Scalar *cosf) {
  Scalar denom = 1 + (*cosf);
  if (denom > 1.0e-10) {
    Scalar tanf2 = sqrt((1 + ecc) / ome) * (*sinf) / denom;  // tan(0.5*f)
    Scalar tanf2_2 = tanf2 * tanf2;

    // Then we compute sin(f) and cos(f) using:
    // sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    // cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tanf2_2);
    *sinf = 2 * tanf2 * denom;
    *cosf = (1 - tanf2_2) * denom;
  } else {
    // If cos(E) = -1, E = pi and tan(0.5*E) -> inf and f = E = pi
    *sinf = 0;
    *cosf = -1;
  }
}

template <typename Scalar>
inline void solve_kepler(const Scalar &M, const Scalar &ecc, Scalar *sinf,
                                             Scalar *cosf) {
  calcEA(M, ecc, sinf, cosf);
  to_f(ecc, 1 - ecc, sinf, cosf);
}

}  // namespace kepler
}  // namespace exoplanet

#endif
