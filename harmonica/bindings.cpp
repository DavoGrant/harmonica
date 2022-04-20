#include <pybind11/pybind11.h>

#include "orbit/orbit.hpp"
#include "light_curve/light_curve.hpp"


PYBIND11_MODULE(bindings, m) {
    m.def("orbit", &orbit);
    m.def("light_curve", &light_curve);
}
