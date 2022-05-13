#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fluxes.hpp"
#include "../constants/constants.hpp"


Fluxes::Fluxes(int ld_law,
               py::array_t<double, py::array::c_style> us,
               py::array_t<double, py::array::c_style> rs,
               bool require_gradients) {

  // Unpack python arrays.
  auto us_ = us.unchecked<1>();
  auto rs_ = rs.unchecked<1>();

  if (ld_law == 0) {
    // Normalisation.
    I_0 = 1 / ((1 - us_(0) / 3. - us_(1) / 6.) * M_PI)

    // Quadratic limb darkening law.
    Eigen::Vector<double, 3> u {1, us_(0), us_(1)};
    Eigen::Matrix<double, 3, 3> B {{1., -1., -1.},
                                   {0., 1., 2.},
                                   {0., 0., -1.}};

    // Change of basis.
    p = B * u;

  } else if (ld_law == 1) {
    // Normalisation.
    I_0 = 1 / ((1 - us_(0) / 5. - us_(1) / 3.
                - 3. * us_(2) / 7. - us_(3) / 2.) * M_PI)

    // Non-linear limb darkening law.
    Eigen::Vector<double, 5> u {1, us_(0), us_(1), us_(2), us_(3)};
    Eigen::Matrix<double, 5, 5> B {{1., -1., -1., -1., -1.},
                                   {0., 1., 0., 0., 0.},
                                   {0., 0., 1., 0., 0.},
                                   {0., 0., 0., 1., 0.},
                                   {0., 0., 0., 0., 1.}};
    // Change of basis.
    p = B * u;

  }

  // Todo.
  c.resize(11);
}


void Fluxes::transit_light_curve(const double &d, const double &nu, double &f,
                                 const double* dd_dz[], const double* dnu_dz[],
                                 double* df_dz[]) {

    std::cout << p << std::endl;

}
