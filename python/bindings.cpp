#include <pybind11/pybind11.h>
#include "PositionnerStage.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_stage_positionner, m) {
    m.doc() = "stage-positionner: orientation optimizer and packer";

    // Import pipeline_core so pybind11 resolves the Stage base type
    py::module_::import("pipeline_core");

    py::class_<PositionnerStage, Stage, std::shared_ptr<PositionnerStage>>(m, "PositionnerStage")
        .def(py::init<>())
        .def("name",      &PositionnerStage::name)
        .def("describe",  &PositionnerStage::describe)
        .def("configure", &PositionnerStage::configure)
        .def("contract",  &PositionnerStage::contract)
        .def("process",   &PositionnerStage::process);
}
