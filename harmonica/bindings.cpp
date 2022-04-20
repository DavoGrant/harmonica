#include <pybind11/pybind11.h>

#include "orbit/trajectories.hpp"
#include "light_curve/fluxes.hpp"


PYBIND11_MODULE(bindings, m) {
    m.def("orbit", &orbital_trajectories);
    m.def("light_curve", &light_curve_fluxes);
}
