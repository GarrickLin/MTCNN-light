// Wrapper for most external modules
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <exception>
#include "make_array.hpp"
#include "np_opencv_converter.hpp"
#include "mtcnn.h"

namespace py = boost::python;

using boost::python::init;


namespace fs {
	namespace python{
		BOOST_PYTHON_MODULE(mtcnn){
			// Main types export
			fs::python::init_and_export_converters();
			py::scope scope = py::scope();

			py::class_<Bbox>("Bbox")
				.def(py::init<>())
				.def_readwrite("score", &Bbox::score)
				.def_readwrite("x1", &Bbox::x1)
				.def_readwrite("y1", &Bbox::y1)
				.def_readwrite("x2", &Bbox::x2)
				.def_readwrite("y2", &Bbox::y2)
				.def_readwrite("area", &Bbox::area)
				.def_readwrite("exist", &Bbox::exist)
				.add_property("ppoint", make_array(&Bbox::ppoint))
				.add_property("regreCoord", make_array(&Bbox::regreCoord))
				;


		}
	}
}