#include <pybind11/pybind11.h>


namespace py = pybind11;

int orbit(int i, int j) {
    return i + j;
}

int light_curve(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(bindings, m) {
    m.def("orbit", &orbit);
    m.def("light_curve", &light_curve);
}
