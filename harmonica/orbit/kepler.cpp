#include <cmath>
#include <cstdlib>

#include "kepler.hpp"
#include "../constants/constants.hpp"


TrueAnomaly solve_kepler(const double M, const double ecc) {
  // Compute eccentric anomaly.
  EccentricAnomaly ecc_anom = compute_eccentric_anomaly(M, ecc);

  // Compute true anomaly.
  return compute_true_anomaly(ecc, ecc_anom);
}


EccentricAnomaly compute_eccentric_anomaly(const double M, const double ecc) {

  const double g2s_e = 0.2588190451025207623489 * ecc;
  const double g3s_e = 0.5 * ecc;
  const double g4s_e = 0.7071067811865475244008 * ecc;
  const double g5s_e = 0.8660254037844386467637 * ecc;
  const double g6s_e = 0.9659258262890682867497 * ecc;

  double bounds[13];
  double EA_tab[9];

  int k;
  double MA, EA, sE, cE, x, y;
  double B0, B1, B2, dx, idx;
  int MAsign = 1;
  double one_over_ecc = 1e17;
  if (ecc > 1e-17) one_over_ecc = 1. / ecc;

  double sinE;
  double cosE;

  MA = fmod(M, fractions::twopi);
  if (MA < .0) {
    MA += fractions::twopi;
  }
  if (MA > fractions::pi) {
    MAsign = -1;
    MA = fractions::twopi - MA;
  }

  if (2 * MA + 1 - ecc < 0.2) {
    // Series expansion.
    EA = eccentric_anomaly_guess(MA, ecc);
  } else {
    // Polynomial boundaries.
    bounds[0] = 0;
    bounds[1] = fractions::pi_d_12 - g2s_e;
    bounds[2] = fractions::pi_d_6 - g3s_e;
    bounds[3] = fractions::pi_d_4 - g4s_e;
    bounds[4] = fractions::pi_d_3 - g5s_e;
    bounds[5] = fractions::fivepi_d_12 - g6s_e;
    bounds[6] = fractions::pi_d_2 - ecc;
    bounds[7] = fractions::sevenpi_d_12 - g6s_e;
    bounds[8] = fractions::twopi_d_3 - g5s_e;
    bounds[9] = fractions::threepi_d_4 - g4s_e;
    bounds[10] = fractions::fivepi_d_6 - g3s_e;
    bounds[11] = fractions::elevenpi_d_12 - g2s_e;
    bounds[12] = fractions::pi;

    /* Which interval? */
    for (k = 11; k > 0; k--) {
      if (MA > bounds[k]) break;
    }

    // Values at the two endpoints.
    EA_tab[0] = k * fractions::pi_d_12;
    EA_tab[6] = (k + 1) * fractions::pi_d_12;

    // First two derivatives at the endpoints. Left endpoint first.
    int sign = (k >= 6) ? 1 : -1;

    x = 1 / (1 - ((6 - k) * fractions::pi_d_12 + sign * bounds[abs(6 - k)]));
    y = -0.5 * (k * fractions::pi_d_12 - bounds[k]);
    EA_tab[1] = x;
    EA_tab[2] = y * x * x * x;

    x = 1 / (1 - ((5 - k) * fractions::pi_d_12 + sign * bounds[abs(5 - k)]));
    y = -0.5 * ((k + 1) * fractions::pi_d_12 - bounds[k + 1]);
    EA_tab[7] = x;
    EA_tab[8] = y * x * x * x;

    // Solve a matrix equation to get the rest of the coefficients.
    idx = 1 / (bounds[k + 1] - bounds[k]);

    B0 = idx * (-EA_tab[2] - idx * (EA_tab[1] - idx * fractions::pi_d_12));
    B1 = idx * (-2 * EA_tab[2] - idx * (EA_tab[1] - EA_tab[7]));
    B2 = idx * (EA_tab[8] - EA_tab[2]);

    EA_tab[3] = B2 - 4 * B1 + 10 * B0;
    EA_tab[4] = (-2 * B2 + 7 * B1 - 15 * B0) * idx;
    EA_tab[5] = (B2 - 3 * B1 + 6 * B0) * idx * idx;

    // Now use the coefficients of this polynomial to get the initial guess.
    dx = MA - bounds[k];
    EA = EA_tab[0] + dx * (EA_tab[1] + dx * (EA_tab[2] + dx * (EA_tab[3]
         + dx * (EA_tab[4] + dx * EA_tab[5]))));
  }

  // Sine and cosine initial guesses using series
  if (EA < fractions::pi_d_4) {
    sE = sin(EA);
    cE = sqrt(1 - sE * sE);
  } else if (EA > fractions::threepi_d_4) {
    sE = sin(fractions::pi - EA);
    cE = -sqrt(1 - sE * sE);
  } else {
    cE = sin(fractions::pi_d_2 - EA);
    sE = sqrt(1 - cE * cE);
  }

  double num, denom, dEA;

  // Halley's method to update E.
  num = (MA - EA) * one_over_ecc + sE;
  denom = one_over_ecc - cE;
  dEA = num * denom / (denom * denom + 0.5 * sE * num);

  // Use series to update sin and cos.
  if (ecc < 0.78 || MA > 0.4) {

    // *E = MAsign * (EA + dEA);
    sinE = MAsign * (sE * (1 - 0.5 * dEA * dEA) + dEA * cE);
    cosE = cE * (1 - 0.5 * dEA * dEA) - dEA * sE;

  } else {

    // Use Householder's third order method to guarantee performance
    // in the singular corners.
    dEA = num / (denom + dEA * (0.5 * sE + fractions::one_sixth * cE * dEA));
    // *E = MAsign * (EA + dEA);
    sinE = MAsign * (sE * (1 - 0.5 * dEA * dEA) + dEA * cE
           * (1 - dEA * dEA * fractions::one_sixth));
    cosE = cE * (1 - 0.5 * dEA * dEA) - dEA * sE
           * (1 - dEA * dEA * fractions::one_sixth);
  }

  EccentricAnomaly ecc_anom;
  ecc_anom.sinE = sinE;
  ecc_anom.cosE = cosE;

  return ecc_anom;
}


double eccentric_anomaly_guess(const double M, const double ecc) {

  const double ome = 1. - ecc;
  const double sqrt_ome = sqrt(ome);
  const double chi = M / (sqrt_ome * ome);
  const double Lam = sqrt(8 + 9 * chi * chi);
  const double S = cbrt(Lam + 3 * chi);
  const double sigma = 6 * chi / (2 + S * S + 4. / (S * S));
  const double s2 = sigma * sigma;
  const double s4 = s2 * s2;

  const double denom = 1.0 / (s2 + 2);
  const double E = sigma * (1 + s2 * ome * denom * ((s2 + 20) / 60.
                   + s2 * ome * denom * denom * (s2 * s4 + 25 * s4
                   + 340 * s2 + 840) / 1400));

  return E * sqrt_ome;
}


TrueAnomaly compute_true_anomaly(const double ecc,
                                 const EccentricAnomaly& ecc_anom) {

  const double ome = 1. - ecc;
  double sinf;
  double cosf;

  double denom = 1 + ecc_anom.cosE;
  if (denom > 1.0e-10) {

    double tanf2 = sqrt((1 + ecc) / ome) * ecc_anom.sinE / denom;  // tan(0.5*f)
    double tanf2_2 = tanf2 * tanf2;

    // Then we compute sin(f) and cos(f) using:
    // sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    // cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tanf2_2);
    sinf = 2 * tanf2 * denom;
    cosf = (1 - tanf2_2) * denom;

  } else {
    // If cos(E) = -1, E = pi and tan(0.5*E) -> inf and f = E = pi
    sinf = 0.;
    cosf = -1.;
  }

  TrueAnomaly true_anom;
  true_anom.sinf = sinf;
  true_anom.cosf = cosf;

  return true_anom;
}
