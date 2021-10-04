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

  std::string beam_to_string(const MonochromaticBeam &beam) {
    std::stringstream ss;
    ss << beam;
    return ss.str();
  }

  struct BeamPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const MonochromaticBeam &obj) {
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
      const MonochromaticBeam &beam = boost::python::extract<const MonochromaticBeam &>(obj)();
      return boost::python::make_tuple(obj.attr("__dict__"),
                                       beam.get_s0_at_scan_points());
    }

    static void setstate(boost::python::object obj, boost::python::tuple state) {
      MonochromaticBeam &beam = boost::python::extract<MonochromaticBeam &>(obj)();
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
  
  struct TOFBeamPickleSuite : boost::python::pickle_suite {
    static boost::python::tuple getinitargs(const TOFBeam &obj) {
      return boost::python::make_tuple(obj.get_sample_to_source_direction(),
                                       obj.get_sample_to_moderator_distance(),
                                       obj.get_wavelength_range());
    }
    static boost::python::tuple getstate(boost::python::object obj) {
      const TOFBeam &beam = boost::python::extract<const TOFBeam &>(obj)();
      return boost::python::make_tuple(obj.attr("__dict__"));
    }

    static void setstate(boost::python::object obj, boost::python::tuple state) {
      TOFBeam &beam = boost::python::extract<TOFBeam &>(obj)();
      // restore the object's __dict__
      boost::python::dict d =
        boost::python::extract<boost::python::dict>(obj.attr("__dict__"))();
      d.update(state[0]);
    }

    static bool getstate_manages_dict() {
      return true;
    }

  };

  static MonochromaticBeam *make_monochromatic_beam(vec3<double> sample_to_source,
                         double wavelength,
                         double divergence,
                         double sigma_divergence,
                         bool deg) {
    MonochromaticBeam *beam = NULL;
    if (deg) {
      beam = new MonochromaticBeam(sample_to_source,
                      wavelength,
                      deg_as_rad(divergence),
                      deg_as_rad(sigma_divergence));
    } else {
      beam = new MonochromaticBeam(sample_to_source, wavelength, divergence, sigma_divergence);
    }
    return beam;
  }

  static MonochromaticBeam *make_monochromatic_beam_w_s0(vec3<double> s0,
                              double divergence,
                              double sigma_divergence,
                              bool deg) {
    MonochromaticBeam *beam = NULL;
    if (deg) {
      beam = new MonochromaticBeam(s0, deg_as_rad(divergence), deg_as_rad(sigma_divergence));
    } else {
      beam = new MonochromaticBeam(s0, divergence, sigma_divergence);
    }
    return beam;
  }

  static MonochromaticBeam *make_monochromatic_beam_w_all(vec3<double> sample_to_source,
                               double wavelength,
                               double divergence,
                               double sigma_divergence,
                               vec3<double> polarization_normal,
                               double polarization_fraction,
                               double flux,
                               double transmission,
                               bool deg) {
    MonochromaticBeam *beam = NULL;
    if (deg) {
      beam = new MonochromaticBeam(sample_to_source,
                      wavelength,
                      deg_as_rad(divergence),
                      deg_as_rad(sigma_divergence),
                      polarization_normal,
                      polarization_fraction,
                      flux,
                      transmission);
    } else {
      beam = new MonochromaticBeam(sample_to_source,
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

  static double get_divergence(const MonochromaticBeam &beam, bool deg) {
    double divergence = beam.get_divergence();
    return deg ? rad_as_deg(divergence) : divergence;
  }

  static double get_sigma_divergence(const MonochromaticBeam &beam, bool deg) {
    double sigma_divergence = beam.get_sigma_divergence();
    return deg ? rad_as_deg(sigma_divergence) : sigma_divergence;
  }

  static void set_divergence(MonochromaticBeam &beam, double divergence, bool deg) {
    beam.set_divergence(deg ? deg_as_rad(divergence) : divergence);
  }

  static void set_sigma_divergence(MonochromaticBeam &beam, double sigma_divergence, bool deg) {
    beam.set_sigma_divergence(deg ? deg_as_rad(sigma_divergence) : sigma_divergence);
  }

  static void rotate_around_origin(Beam &beam,
                                   vec3<double> axis,
                                   double angle,
                                   bool deg) {
    double angle_rad = deg ? deg_as_rad(angle) : angle;
    beam.rotate_around_origin(axis, angle_rad);
  }

  static void Beam_set_s0_at_scan_points_from_tuple(MonochromaticBeam &beam,
                                                    boost::python::tuple l) {
    scitbx::af::shared<vec3<double> > s0_list;
    for (std::size_t i = 0; i < boost::python::len(l); ++i) {
      vec3<double> s0 = boost::python::extract<vec3<double> >(l[i]);
      s0_list.push_back(s0);
    }
    beam.set_s0_at_scan_points(s0_list.const_ref());
  }

  static void Beam_set_s0_at_scan_points_from_list(MonochromaticBeam &beam, boost::python::list l) {
    scitbx::af::shared<vec3<double> > s0_list;
    for (std::size_t i = 0; i < boost::python::len(l); ++i) {
      vec3<double> s0 = boost::python::extract<vec3<double> >(l[i]);
      s0_list.push_back(s0);
    }
    beam.set_s0_at_scan_points(s0_list.const_ref());
  }

  template <>
  boost::python::dict to_dict<MonochromaticBeam>(const MonochromaticBeam &obj) {
    boost::python::dict result;
    result["__id__"] = "MonochromaticBeam";
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
  MonochromaticBeam *from_dict<MonochromaticBeam>(boost::python::dict obj) {
    MonochromaticBeam *b = new MonochromaticBeam(
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

  static TOFBeam *make_tof_beam(vec3<double> sample_to_source,
                         double moderator_sample_distance,
                         vec2<double> wavelength_range) {
    return new TOFBeam(sample_to_source, moderator_sample_distance, wavelength_range);
  }

  template<>
  boost::python::dict to_dict<TOFBeam>(const TOFBeam &obj) {
    boost::python::dict result;
    result["__id__"] = "TOFBeam";
    result["direction"] = obj.get_sample_to_source_direction();
    result["sample_to_moderator_distance"] = obj.get_sample_to_moderator_distance();
    result["wavelength_range"] = obj.get_wavelength_range();
    return result;
  }

  template<>
  TOFBeam *from_dict<TOFBeam>(boost::python::dict obj){
    return new TOFBeam(boost::python::extract<vec3<double> >(obj["direction"]),
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
 
    // TOFBeam : Beam
    class_<TOFBeam, boost::shared_ptr<TOFBeam>, bases<Beam> >("TOFBeam")
      .def(init<const TOFBeam &>())
      .def(init<vec3<double>, double, vec2<double> >((arg("direction"), 
                                       arg("sample_to_moderator_distance"), 
                                       arg("wavelength_range"))))
      .def("__init__",
           make_constructor(&make_tof_beam,
                            default_call_policies(),
                            (arg("direction"),
                             arg("sample_to_moderator_distance"),
                             arg("wavelength_range"))))
      .def("get_sample_to_moderator_distance", &TOFBeam::get_sample_to_moderator_distance)
      .def("set_sample_to_moderator_distance", &TOFBeam::set_sample_to_moderator_distance)
      .def("get_unit_s0", &TOFBeam::get_unit_s0)
      .def("set_unit_s0", &TOFBeam::set_unit_s0)
      .def("get_wavelength_range", &TOFBeam::get_wavelength_range)
      .def("set_wavelength_range", &TOFBeam::set_wavelength_range)
      .def("to_dict", &to_dict<TOFBeam>)
      .def("from_dict", &from_dict<TOFBeam>, return_value_policy<manage_new_object>())
      .def_pickle(TOFBeamPickleSuite());

    // MonochromaticBeam : Beam
    class_<MonochromaticBeam, boost::shared_ptr<MonochromaticBeam>, bases<Beam> >("MonochromaticBeam")
      .def(init<const MonochromaticBeam &>())
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
      .def("to_dict", &to_dict<MonochromaticBeam>)
      .def("from_dict", &from_dict<MonochromaticBeam>, return_value_policy<manage_new_object>())
      .def("get_wavelength", &MonochromaticBeam::get_wavelength)
      .def("set_wavelength", &MonochromaticBeam::set_wavelength)
      .def("get_s0", &MonochromaticBeam::get_s0)
      .def("set_s0", &MonochromaticBeam::set_s0)
      .def("get_unit_s0", &MonochromaticBeam::get_unit_s0)
      .def("set_unit_s0", &MonochromaticBeam::set_unit_s0)
      .def("get_divergence", &get_divergence, (arg("deg") = true))
      .def("set_divergence", &set_divergence, (arg("divergence"), arg("deg") = true))
      .def("get_sigma_divergence", &get_sigma_divergence, (arg("deg") = true))
      .def("set_sigma_divergence",
           &set_sigma_divergence,
           (arg("sigma_divergence"), arg("deg") = true))
      .def("get_polarization_normal", &MonochromaticBeam::get_polarization_normal)
      .def("set_polarization_normal", &MonochromaticBeam::set_polarization_normal)
      .def("get_polarization_fraction", &MonochromaticBeam::get_polarization_fraction)
      .def("set_polarization_fraction", &MonochromaticBeam::set_polarization_fraction)
      .def("get_flux", &MonochromaticBeam::get_flux)
      .def("set_flux", &MonochromaticBeam::set_flux)
      .def("get_transmission", &MonochromaticBeam::get_transmission)
      .def("set_transmission", &MonochromaticBeam::set_transmission)
      .add_property("num_scan_points", &MonochromaticBeam::get_num_scan_points)
      .def("get_num_scan_points", &MonochromaticBeam::get_num_scan_points)
      .def("set_s0_at_scan_points", &MonochromaticBeam::set_s0_at_scan_points)
      .def("set_s0_at_scan_points", &Beam_set_s0_at_scan_points_from_tuple)
      .def("set_s0_at_scan_points", &Beam_set_s0_at_scan_points_from_list)
      .def("get_s0_at_scan_points", &MonochromaticBeam::get_s0_at_scan_points)
      .def("get_s0_at_scan_point", &MonochromaticBeam::get_s0_at_scan_point)
      .def("reset_scan_points", &MonochromaticBeam::reset_scan_points)
      .def("__eq__", &MonochromaticBeam::operator==)
      .def("__ne__", &MonochromaticBeam::operator!=)
      .def("is_similar_to",
           &MonochromaticBeam::is_similar_to,
           (arg("other"),
            arg("wavelength_tolerance") = 1e-6,
            arg("direction_tolerance") = 1e-6,
            arg("polarization_normal_tolerance") = 1e-6,
            arg("polarization_fraction_tolerance") = 1e-6))
      .staticmethod("from_dict")
      .def_pickle(BeamPickleSuite());

    scitbx::af::boost_python::flex_wrapper<MonochromaticBeam>::plain("flex_Beam");
  }

}}}  // namespace dxtbx::model::boost_python
