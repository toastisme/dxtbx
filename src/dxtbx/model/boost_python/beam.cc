/*
 * beam.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <string>
#include <sstream>
#include <scitbx/constants.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/boost_python/to_from_dict.h>
#include <scitbx/array_family/boost_python/flex_wrapper.h>

namespace dxtbx { namespace model { namespace boost_python {

  using namespace boost::python;
  using scitbx::deg_as_rad;
  using scitbx::rad_as_deg;

  std::string beam_to_string(const MonoBeam &beam) {
    std::stringstream ss;
    ss << beam;
    return ss.str();
  }

  struct BeamPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const MonoBeam &obj) {
      return boost::python::make_tuple(obj.get_sample_to_source_direction(),
                                       obj.get_wavelength(),
                                       obj.get_divergence(),
                                       obj.get_sigma_divergence(),
                                       obj.get_polarization_normal(),
                                       obj.get_polarization_fraction(),
                                       obj.get_flux(),
                                       obj.get_transmission());
    }

    static boost::python::tuple getstate(boost::python::object obj) {
      const MonoBeam &beam = boost::python::extract<const MonoBeam &>(obj)();
      return boost::python::make_tuple(obj.attr("__dict__"),
                                       beam.get_s0_at_scan_points());
    }

    static void setstate(boost::python::object obj, boost::python::tuple state) {
      MonoBeam &beam = boost::python::extract<MonoBeam &>(obj)();
      DXTBX_ASSERT(boost::python::len(state) == 2);

      // restore the object's __dict__
      boost::python::dict d =
        boost::python::extract<boost::python::dict>(obj.attr("__dict__"))();
      d.update(state[0]);

      // restore the internal state of the C++ object
      scitbx::af::const_ref<vec3<double> > s0_list =
        boost::python::extract<scitbx::af::const_ref<vec3<double> > >(state[1]);
      beam.set_s0_at_scan_points(s0_list);
    }

    static bool getstate_manages_dict() {
      return true;
    }
  };
  
  struct PolyBeamPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const PolyBeam &obj) {
      return boost::python::make_tuple(obj.get_sample_to_source_direction(),
                                       obj.get_sample_to_moderator_distance(),
                                       obj.get_wavelength_range());
    }
    static boost::python::tuple getstate(boost::python::object obj) {
      const PolyBeam &beam = boost::python::extract<const PolyBeam &>(obj)();
      return boost::python::make_tuple(obj.attr("__dict__"));
    }

    static void setstate(boost::python::object obj, boost::python::tuple state) {
      PolyBeam &beam = boost::python::extract<PolyBeam &>(obj)();
      // restore the object's __dict__
      boost::python::dict d =
        boost::python::extract<boost::python::dict>(obj.attr("__dict__"))();
      d.update(state[0]);
    }

    static bool getstate_manages_dict() {
      return true;
    }

  };

  static MonoBeam *make_monochromatic_beam(vec3<double> sample_to_source,
                         double wavelength,
                         double divergence,
                         double sigma_divergence,
                         bool deg) {
    MonoBeam *beam = NULL;
    if (deg) {
      beam = new MonoBeam(sample_to_source,
                      wavelength,
                      deg_as_rad(divergence),
                      deg_as_rad(sigma_divergence));
    } else {
      beam = new MonoBeam(sample_to_source, wavelength, divergence, sigma_divergence);
    }
    return beam;
  }

  static MonoBeam *make_monochromatic_beam_w_s0(vec3<double> s0,
                              double divergence,
                              double sigma_divergence,
                              bool deg) {
    MonoBeam *beam = NULL;
    if (deg) {
      beam = new MonoBeam(s0, deg_as_rad(divergence), deg_as_rad(sigma_divergence));
    } else {
      beam = new MonoBeam(s0, divergence, sigma_divergence);
    }
    return beam;
  }

  static MonoBeam *make_monochromatic_beam_w_all(vec3<double> sample_to_source,
                               double wavelength,
                               double divergence,
                               double sigma_divergence,
                               vec3<double> polarization_normal,
                               double polarization_fraction,
                               double flux,
                               double transmission,
                               bool deg) {
    MonoBeam *beam = NULL;
    if (deg) {
      beam = new MonoBeam(sample_to_source,
                      wavelength,
                      deg_as_rad(divergence),
                      deg_as_rad(sigma_divergence),
                      polarization_normal,
                      polarization_fraction,
                      flux,
                      transmission);
    } else {
      beam = new MonoBeam(sample_to_source,
                      wavelength,
                      divergence,
                      sigma_divergence,
                      polarization_normal,
                      polarization_fraction,
                      flux,
                      transmission);
    }
    return beam;
  }

  static double get_divergence(const MonoBeam &beam, bool deg) {
    double divergence = beam.get_divergence();
    return deg ? rad_as_deg(divergence) : divergence;
  }

  static double get_sigma_divergence(const MonoBeam &beam, bool deg) {
    double sigma_divergence = beam.get_sigma_divergence();
    return deg ? rad_as_deg(sigma_divergence) : sigma_divergence;
  }

  static void set_divergence(MonoBeam &beam, double divergence, bool deg) {
    beam.set_divergence(deg ? deg_as_rad(divergence) : divergence);
  }

  static void set_sigma_divergence(MonoBeam &beam, double sigma_divergence, bool deg) {
    beam.set_sigma_divergence(deg ? deg_as_rad(sigma_divergence) : sigma_divergence);
  }

  static void rotate_around_origin(Beam &beam,
                                   vec3<double> axis,
                                   double angle,
                                   bool deg) {
    double angle_rad = deg ? deg_as_rad(angle) : angle;
    beam.rotate_around_origin(axis, angle_rad);
  }

  static void Beam_set_s0_at_scan_points_from_tuple(MonoBeam &beam,
                                                    boost::python::tuple l) {
    scitbx::af::shared<vec3<double> > s0_list;
    for (std::size_t i = 0; i < boost::python::len(l); ++i) {
      vec3<double> s0 = boost::python::extract<vec3<double> >(l[i]);
      s0_list.push_back(s0);
    }
    beam.set_s0_at_scan_points(s0_list.const_ref());
  }

  static void Beam_set_s0_at_scan_points_from_list(MonoBeam &beam, boost::python::list l) {
    scitbx::af::shared<vec3<double> > s0_list;
    for (std::size_t i = 0; i < boost::python::len(l); ++i) {
      vec3<double> s0 = boost::python::extract<vec3<double> >(l[i]);
      s0_list.push_back(s0);
    }
    beam.set_s0_at_scan_points(s0_list.const_ref());
  }

  template <>
  boost::python::dict to_dict<MonoBeam>(const MonoBeam &obj) {
    boost::python::dict result;
    result["__id__"] = "MonoBeam";
    result["direction"] = obj.get_sample_to_source_direction();
    result["wavelength"] = obj.get_wavelength();
    result["divergence"] = rad_as_deg(obj.get_divergence());
    result["sigma_divergence"] = rad_as_deg(obj.get_sigma_divergence());
    result["polarization_normal"] = obj.get_polarization_normal();
    result["polarization_fraction"] = obj.get_polarization_fraction();
    result["flux"] = obj.get_flux();
    result["transmission"] = obj.get_transmission();
    if (obj.get_num_scan_points() > 0) {
      boost::python::list l;
      scitbx::af::shared<vec3<double> > s0_at_scan_points = obj.get_s0_at_scan_points();
      for (scitbx::af::shared<vec3<double> >::iterator it = s0_at_scan_points.begin();
           it != s0_at_scan_points.end();
           ++it) {
        l.append(boost::python::make_tuple((*it)[0], (*it)[1], (*it)[2]));
      }
      result["s0_at_scan_points"] = l;
    }
    return result;
  }

  template <>
  MonoBeam *from_dict<MonoBeam>(boost::python::dict obj) {
    MonoBeam *b = new MonoBeam(
      boost::python::extract<vec3<double> >(obj["direction"]),
      boost::python::extract<double>(obj["wavelength"]),
      deg_as_rad(boost::python::extract<double>(obj.get("divergence", 0.0))),
      deg_as_rad(boost::python::extract<double>(obj.get("sigma_divergence", 0.0))),
      boost::python::extract<vec3<double> >(
        obj.get("polarization_normal", vec3<double>(0.0, 1.0, 0.0))),
      boost::python::extract<double>(obj.get("polarization_fraction", 0.999)),
      boost::python::extract<double>(obj.get("flux", 0)),
      boost::python::extract<double>(obj.get("transmission", 1)));
    if (obj.has_key("s0_at_scan_points")) {
      boost::python::list s0_at_scan_points =
        boost::python::extract<boost::python::list>(obj["s0_at_scan_points"]);
      Beam_set_s0_at_scan_points_from_list(*b, s0_at_scan_points);
    }
    return b;
  }

  static PolyBeam *make_tof_beam(vec3<double> sample_to_source,
                         double moderator_sample_distance,
                         vec2<double> wavelength_range) {
    return new PolyBeam(sample_to_source, moderator_sample_distance, wavelength_range);
  }

  template<>
  boost::python::dict to_dict<PolyBeam>(const PolyBeam &obj) {
    boost::python::dict result;
    result["__id__"] = "PolyBeam";
    result["direction"] = obj.get_sample_to_source_direction();
    result["sample_to_moderator_distance"] = obj.get_sample_to_moderator_distance();
    result["wavelength_range"] = obj.get_wavelength_range();
    return result;
  }

  template<>
  PolyBeam *from_dict<PolyBeam>(boost::python::dict obj){
    return new PolyBeam(boost::python::extract<vec3<double> >(obj["direction"]),
      boost::python::extract<double>(obj["sample_to_moderator_distance"]),
      boost::python::extract<vec2<double> >(obj["wavelength_range"]));
  }

  void export_beam() {

    // Beam
    class_<Beam, boost::noncopyable>("Beam", no_init)
      .def("get_sample_to_source_direction", &Beam::get_sample_to_source_direction)
      .def("set_sample_to_source_direction", &Beam::set_sample_to_source_direction)
      .def("rotate_around_origin",
           &rotate_around_origin,
           (arg("axis"), arg("angle"), arg("deg") = true));
 
    // PolyBeam : Beam
    class_<PolyBeam, boost::shared_ptr<PolyBeam>, bases<Beam> >("PolyBeam")
      .def(init<const PolyBeam &>())
      .def(init<vec3<double>, double, vec2<double> >((arg("direction"), 
                                       arg("sample_to_moderator_distance"), 
                                       arg("wavelength_range"))))
      .def("__init__",
           make_constructor(&make_tof_beam,
                            default_call_policies(),
                            (arg("direction"),
                             arg("sample_to_moderator_distance"),
                             arg("wavelength_range"))))
      .def("get_sample_to_moderator_distance", &PolyBeam::get_sample_to_moderator_distance)
      .def("set_sample_to_moderator_distance", &PolyBeam::set_sample_to_moderator_distance)
      .def("get_unit_s0", &PolyBeam::get_unit_s0)
      .def("set_unit_s0", &PolyBeam::set_unit_s0)
      .def("get_wavelength_range", &PolyBeam::get_wavelength_range)
      .def("set_wavelength_range", &PolyBeam::set_wavelength_range)
      .def("to_dict", &to_dict<PolyBeam>)
      .def("from_dict", &from_dict<PolyBeam>, return_value_policy<manage_new_object>())
      .def_pickle(PolyBeamPickleSuite());

    // MonoBeam : Beam
    class_<MonoBeam, boost::shared_ptr<MonoBeam>, bases<Beam> >("MonoBeam")
      .def(init<const MonoBeam &>())
      .def(init<vec3<double>, double>((arg("direction"), arg("wavelength"))))
      .def(init<vec3<double> >((arg("s0"))))
      .def("__init__",
           make_constructor(&make_monochromatic_beam,
                            default_call_policies(),
                            (arg("direction"),
                             arg("wavelength"),
                             arg("divergence"),
                             arg("sigma_divergence"),
                             arg("deg") = true)))
      .def(
        "__init__",
        make_constructor(
          &make_monochromatic_beam_w_s0,
          default_call_policies(),
          (arg("s0"), arg("divergence"), arg("sigma_divergence"), arg("deg") = true)))
      .def("__init__",
           make_constructor(&make_monochromatic_beam_w_all,
                            default_call_policies(),
                            (arg("direction"),
                             arg("wavelength"),
                             arg("divergence"),
                             arg("sigma_divergence"),
                             arg("polarization_normal"),
                             arg("polarization_fraction"),
                             arg("flux"),
                             arg("transmission"),
                             arg("deg") = true)))
      .def("__str__", &beam_to_string)
      .def("to_dict", &to_dict<MonoBeam>)
      .def("from_dict", &from_dict<MonoBeam>, return_value_policy<manage_new_object>())
      .def("get_wavelength", &MonoBeam::get_wavelength)
      .def("set_wavelength", &MonoBeam::set_wavelength)
      .def("get_s0", &MonoBeam::get_s0)
      .def("set_s0", &MonoBeam::set_s0)
      .def("get_unit_s0", &MonoBeam::get_unit_s0)
      .def("set_unit_s0", &MonoBeam::set_unit_s0)
      .def("get_divergence", &get_divergence, (arg("deg") = true))
      .def("set_divergence", &set_divergence, (arg("divergence"), arg("deg") = true))
      .def("get_sigma_divergence", &get_sigma_divergence, (arg("deg") = true))
      .def("set_sigma_divergence",
           &set_sigma_divergence,
           (arg("sigma_divergence"), arg("deg") = true))
      .def("get_polarization_normal", &MonoBeam::get_polarization_normal)
      .def("set_polarization_normal", &MonoBeam::set_polarization_normal)
      .def("get_polarization_fraction", &MonoBeam::get_polarization_fraction)
      .def("set_polarization_fraction", &MonoBeam::set_polarization_fraction)
      .def("get_flux", &MonoBeam::get_flux)
      .def("set_flux", &MonoBeam::set_flux)
      .def("get_transmission", &MonoBeam::get_transmission)
      .def("set_transmission", &MonoBeam::set_transmission)
      .add_property("num_scan_points", &MonoBeam::get_num_scan_points)
      .def("get_num_scan_points", &MonoBeam::get_num_scan_points)
      .def("set_s0_at_scan_points", &MonoBeam::set_s0_at_scan_points)
      .def("set_s0_at_scan_points", &Beam_set_s0_at_scan_points_from_tuple)
      .def("set_s0_at_scan_points", &Beam_set_s0_at_scan_points_from_list)
      .def("get_s0_at_scan_points", &MonoBeam::get_s0_at_scan_points)
      .def("get_s0_at_scan_point", &MonoBeam::get_s0_at_scan_point)
      .def("reset_scan_points", &MonoBeam::reset_scan_points)
      .def("__eq__", &MonoBeam::operator==)
      .def("__ne__", &MonoBeam::operator!=)
      .def("is_similar_to",
           &MonoBeam::is_similar_to,
           (arg("other"),
            arg("wavelength_tolerance") = 1e-6,
            arg("direction_tolerance") = 1e-6,
            arg("polarization_normal_tolerance") = 1e-6,
            arg("polarization_fraction_tolerance") = 1e-6))
      .staticmethod("from_dict")
      .def_pickle(BeamPickleSuite());

    scitbx::af::boost_python::flex_wrapper<MonoBeam>::plain("flex_Beam");
  }

}}}  // namespace dxtbx::model::boost_python
