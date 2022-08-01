/*
 * sequence.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DXTBX_MODEL_SCAN_H
#define DXTBX_MODEL_SCAN_H

#include <cmath>
#include <iostream>
#include <map>
#include <scitbx/vec2.h>
#include <scitbx/array_family/shared.h>
#include <scitbx/array_family/simple_io.h>
#include <scitbx/array_family/simple_tiny_io.h>
#include <dxtbx/error.h>
#include "scan_helpers.h"
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <dxtbx/array_family/flex_table.h>
#include <dxtbx/array_family/flex_table_suite.h>

namespace dxtbx { namespace model {

  using scitbx::rad_as_deg;
  using scitbx::vec2;
  using scitbx::constants::pi;

  typedef std::map<std::string, scitbx::af::shared<vec2<int> > > ExpImgRangeMap;

  typedef properties_type_generator < bool, int, std::size_t, double, std::string,
    vec2<double>, vec3<double>, mat3<double>, int6, property_types;

  typedef property_type_generator::type property_types;

  /** A class containing minimal information for a sequence of images */
  class Sequence {
  public:
    Sequence() : image_range_(0, 0), num_images_(0), batch_offset_(0) {}

    /**
     * @param image_range The range of images covered by the sequence
     * @param batch_offset A offset to add to the image number (for tracking of
     *                     unique batch numbers for multi-crystal datasets)
     */
    Sequence(vec2<int> image_range, int batch_offset = 0)
        : image_range_(image_range),
          num_images_(1 + image_range_[1] - image_range_[0]),
          batch_offset_(batch_offset) {
      DXTBX_ASSERT(num_images_ >= 0);
    }

    virtual ~Sequence() {}

    int get_num_images() const {
      return num_images_;
    }

    vec2<int> get_image_range() const {
      return image_range_;
    }

    int get_batch_offset() const {
      return batch_offset_;
    }

    int get_batch_for_image_index(int index) const {
      return index + batch_offset_;
    }

    int get_batch_for_array_index(int index) const {
      return index + batch_offset_ + 1;
    }

    vec2<int> get_batch_range() const {
      return vec2<int>(image_range_[0] + batch_offset_,
                       image_range_[1] + batch_offset_);
    }

    /** (zero based) */
    vec2<int> get_array_range() const {
      return vec2<int>(image_range_[0] - 1, image_range_[1]);
    }

    /** Get the map, not exported to python **/
    ExpImgRangeMap get_valid_image_ranges_map() const {
      return valid_image_ranges_;
    }

    /** Get the element for a given key if it exists, else return empty array**/
    scitbx::af::shared<vec2<int> > get_valid_image_ranges_key(std::string i) const {
      typedef ExpImgRangeMap::const_iterator iterator;
      for (iterator it = valid_image_ranges_.begin(); it != valid_image_ranges_.end();
           ++it) {
        if (it->first == i) {
          return it->second;
        }
      }
      scitbx::af::shared<vec2<int> > empty;
      return empty;
    }

    void set_valid_image_ranges_array(std::string i,
                                      scitbx::af::shared<vec2<int> > values) {
      /** Set a list of valid image range tuples for experiment identifier 'i'**/
      for (std::size_t j = 0; j < values.size(); ++j) {
        vec2<int> pair = values[j];
        DXTBX_ASSERT(pair[0] >= image_range_[0]);
        DXTBX_ASSERT(pair[0] <= image_range_[1]);
        DXTBX_ASSERT(pair[1] >= image_range_[0]);
        DXTBX_ASSERT(pair[1] <= image_range_[1]);
      }
      valid_image_ranges_[i] = values;
    }

    void set_batch_offset(int batch_offset) {
      batch_offset_ = batch_offset;
    }

    virtual void set_image_range(vec2<int> image_range) {
      image_range_ = image_range;
      num_images_ = 1 + image_range_[1] - image_range_[0];
      DXTBX_ASSERT(num_images_ > 0);
    }

    bool is_image_index_valid(double index) const {
      return (image_range_[0] <= index && index <= image_range_[1]);
    }

    bool is_batch_valid(int batch) const {
      vec2<int> batch_range = get_batch_range();
      return (batch_range[0] <= batch && batch <= batch_range[1]);
    }

    bool is_array_index_valid(double index) const {
      return is_image_index_valid(index + 1);
    }

    /** Comparison operator */
    bool operator<(const Sequence &sequence) const {
      return image_range_[0] < sequence.get_image_range()[0];
    }

    /** Comparison operator */
    bool operator<=(const Sequence &sequence) const {
      return get_image_range()[0] <= sequence.get_image_range()[0];
    }

    /** Comparison operator */
    bool operator>(const Sequence &sequence) const {
      return get_image_range()[0] > sequence.get_image_range()[0];
    }

    /** Comparison operator */
    bool operator>=(const Sequence &sequence) const {
      return get_image_range()[0] >= sequence.get_image_range()[0];
    }

  protected:
    vec2<int> image_range_;
    ExpImgRangeMap valid_image_ranges_; /** initialised as an empty map **/
    int num_images_;
    int batch_offset_;
  };

  /** A class to represent a ToF sequence of images */
  class TOFSequence : public Sequence {
  public:
    TOFSequence()
        : Sequence(vec2<int>{0, 0}, 0),
          tof_in_seconds_(num_images_, 0),
          wavelengths_(num_images_, 0) {}

    /**
     * @param image_range The range of images covered by the sequence
     * @param tof_in_seconds The ToF values of each image
     * @param batch_offset An offset to add to the image number (for tracking
     *                      unique batch numbers for multi-crystal datasets)
     *
     */
    TOFSequence(vec2<int> image_range,
                const scitbx::af::shared<double> &tof_in_seconds,
                const scitbx::af::shared<double> &wavelengths,
                int batch_offset = 0)
        : Sequence(image_range, batch_offset),
          tof_in_seconds_(tof_in_seconds),
          wavelengths_(wavelengths) {
      create_splines();
    }

    virtual ~TOFSequence() {}

    TOFSequence(const TOFSequence &rhs)
        : Sequence(rhs.image_range_, rhs.batch_offset_),
          tof_in_seconds_(scitbx::af::reserve(rhs.tof_in_seconds_.size())),
          wavelengths_(scitbx::af::reserve(rhs.wavelengths_.size())) {
      std::copy(rhs.tof_in_seconds_.begin(),
                rhs.tof_in_seconds_.end(),
                std::back_inserter(tof_in_seconds_));
      std::copy(rhs.wavelengths_.begin(),
                rhs.wavelengths_.end(),
                std::back_inserter(wavelengths_));
      create_splines();
    }

    bool is_still() const {
      return false;
    }

    void create_splines() {
      if (wavelengths_.size() > 5) {
        frame_to_wavelength_ = get_spline(wavelengths_);
        frame_to_tof_ = get_spline(tof_in_seconds_);
      }
    }

    scitbx::af::shared<double> get_tof_in_seconds() const {
      scitbx::af::shared<double> tof_in_seconds;
      for (int i = image_range_[0] - 1; i < image_range_[1] - 1; ++i) {
        tof_in_seconds.push_back(tof_in_seconds_[i]);
      }
      return tof_in_seconds;
    }

    scitbx::af::shared<double> get_all_tof_in_seconds() const {
      scitbx::af::shared<double> tof_in_seconds;
      for (std::size_t i = 0; i < tof_in_seconds_.size(); ++i) {
        tof_in_seconds.push_back(tof_in_seconds_[i]);
      }
      return tof_in_seconds;
    }

    scitbx::af::shared<double> get_wavelengths() const {
      scitbx::af::shared<double> wavelengths;
      for (int i = image_range_[0] - 1; i < image_range_[1] - 1; ++i) {
        wavelengths.push_back(wavelengths_[i]);
      }
      return wavelengths;
    }

    scitbx::af::shared<double> get_all_wavelengths() const {
      scitbx::af::shared<double> wavelengths;
      for (std::size_t i = 0; i < wavelengths_.size(); ++i) {
        wavelengths.push_back(wavelengths_[i]);
      }
      return wavelengths;
    }

    boost::math::interpolators::cardinal_cubic_b_spline<double> get_spline(
      scitbx::af::shared<double> data) {
      double t0 = 0;
      double h = 0.01;
      boost::math::interpolators::cardinal_cubic_b_spline<double> spline(
        data.begin(), data.end(), t0, h);
      return spline;
    }

    double get_wavelength_from_frame(const double frame) const {
      DXTBX_ASSERT(wavelengths_.size() > 5);
      return frame_to_wavelength_(frame);
    }

    int get_num_tof_bins() const {
      return static_cast<int>(tof_in_seconds_.size());
    }

    vec2<double> get_tof_range_in_seconds() const {
      return vec2<double>(tof_in_seconds_[image_range_[0] - 1],
                          tof_in_seconds_[image_range_[1] - 2]);
    }

    double get_image_tof(int index) const {
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return tof_in_seconds_[index - image_range_[0]];
    }

    double get_image_wavelength(int index) const {
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return wavelengths_[index - image_range_[0]];
    }

    void append(const TOFSequence &rhs) {
      DXTBX_ASSERT(image_range_[1] + 1 == rhs.image_range_[0]);
      DXTBX_ASSERT(batch_offset_ == rhs.batch_offset_);
      image_range_[1] = rhs.image_range_[1];
      num_images_ = 1 + image_range_[1] - image_range_[0];
    }

    /** Check the sequences are the same */
    bool operator==(const TOFSequence &rhs) const {
      double eps = 1e-7;
      return get_image_range() == rhs.get_image_range()
             && get_batch_offset() == rhs.get_batch_offset()
             && tof_in_seconds_.const_ref().all_approx_equal(
               rhs.tof_in_seconds_.const_ref(), eps);
    }

    /** Check the scans are not the same */
    bool operator!=(const TOFSequence &sequence) const {
      return !(*this == sequence);
    }

    TOFSequence operator[](int index) const {
      // Check index
      DXTBX_ASSERT((index >= 0) && (index < get_num_images()));
      int image_index = get_image_range()[0] + index;

      scitbx::af::shared<double> new_tof_in_seconds(1);
      new_tof_in_seconds[0] = get_image_tof(image_index);

      scitbx::af::shared<double> new_wavelengths(1);
      new_wavelengths[0] = get_image_wavelength(image_index);

      // Return scan
      return TOFSequence(vec2<int>(image_index, image_index),
                         new_tof_in_seconds,
                         new_wavelengths,
                         get_batch_offset());
    }

    /**
     * Append the rhs sequence onto the current sequence
     */
    TOFSequence &operator+=(const TOFSequence &rhs) {
      append(rhs);
      return *this;
    }

    /**
     * Return a new sequence which consists of the contents of this sequence and
     * the contents of the other sequence, provided that they are consistent.
     * If they are not consistent then an AssertionError will result.
     */
    TOFSequence operator+(const TOFSequence &rhs) const {
      TOFSequence lhs(*this);
      lhs += rhs;
      return lhs;
    }

    friend std::ostream &operator<<(std::ostream &os, const TOFSequence &s);

  private:
    scitbx::af::shared<double> tof_in_seconds_;
    scitbx::af::shared<double> wavelengths_;
    boost::math::interpolators::cardinal_cubic_b_spline<double> frame_to_wavelength_;
    boost::math::interpolators::cardinal_cubic_b_spline<double> frame_to_tof_;
  };

  /** A class to represent a scan */
  class Scan {
  public:
    /** The default constructor */
    Scan()
        : image_range_(0, 0),
          num_images_(1),
          batch_offset_(0),
          properties_(flex_table<property_types>(1)) {}

    /**
     * Initialise the class
     * @param image_range The range of images covered by the scan
     * @param oscillation A tuple containing the start angle of the first image
     *                    and the oscillation range (the angular width) of each
     *                    frame
     * @param batch_offset A offset to add to the image number (for tracking of
     *                     unique batch numbers for multi-crystal datasets)
     */
    Scan(vec2<int> image_range, vec2<double> oscillation, int batch_offset = 0)
        : image_range_(image_range),
          num_images_(1 + image_range_[1] - image_range_[0]),
          batch_offset_(batch_offset) {
      DXTBX_ASSERT(num_images_ >= 0);
      properties_ = flex_table<property_types>(num_images_);
      properties_["exposure_times"] = scitbx::af::shared<double>(num_images_, 0.0);
      properties_["epochs"] = scitbx::af::shared<double>(num_images_, 0.0);
      set_oscillation(oscillation);
      DXTBX_ASSERT(properties_.is_consistent());
    }

    /**
     * Initialise the class
     * @param image_range The range of images covered by the scan
     * @param oscillation A tuple containing the start angle of the first image
     *                    and the oscillation range (the angular width) of each
     *                    frame
     * @param exposure_times The exposure duration of each image
     * @param epochs The time of capture for each image
     * @param batch_offset A offset to add to the image number (for tracking of
     *                     unique batch numbers for multi-crystal datasets)
     */
    Scan(vec2<int> image_range,
         vec2<double> oscillation,
         const scitbx::af::shared<double> &exposure_times,
         const scitbx::af::shared<double> &epochs,
         int batch_offset = 0)
        : image_range_(image_range),
          num_images_(1 + image_range_[1] - image_range_[0]),
          batch_offset_(batch_offset) {
      DXTBX_ASSERT(num_images_ >= 0);
      DXTBX_ASSERT(oscillation[1] >= 0);
      properties_ = flex_table<property_types>(num_images_);

      if (exposure_times_.size() == 1 && num_images_ > 1) {
        // assume same exposure time for all images - there is
        // probably a better way of coding this...
        scitbx::af::shared<double> expanded_exposure_times;
        expanded_exposure_times.reserve(num_images_);
        for (int j = 0; j < num_images_; j++) {
          expanded_exposure_times.push_back(exposure_times[0]);
        }
        properties_["exposure_times"] = expanded_exposure_times;
      } else {
        properties_["exposure_times"] = exposure_times;
      }
      properties_["epochs"] = epochs;
      set_oscillation(oscillation);
      DXTBX_ASSERT(properties_.is_consistent());
    }

    Scan(vec2<int> image_range,
         flex_table<property_types> properties_table,
         int batch_offset = 0)
        : image_range_(image_range),
          num_images(1 + image_range_[1] - image_range_[0]),
          batch_offset_(batch_offset) {
      DXTBX_ASSERT(num_images_ >= 0);
      DXTBX_ASSERT(properties_table.size() == num_images);
      properties_ = properties_table;
    }

    /** Copy */
    Scan(const Scan &rhs)
        : image_range_(rhs.image_range),
          valid_image_ranges_(rhs.valid_image_ranges_),
          num_images_(rhs.num_images_),
          batch_offset_(rhs.batch_offset_),
          properties_(rhs.properties_) {}

    /** Virtual destructor */
    virtual ~Scan() {}

    /** Get the still flag */
    bool is_still() const {
      if (!properties_.contains["oscillation"]) {
        return false;
      }
      if (properties_["oscillation"].size() == 0) {
        return false;
      }

      return std::abs(properties_["oscillation"][1]) < min_oscillation_width_;
    }

    /** Get the oscillation */
    vec2<double> get_oscillation() const {
      DXTBX_ASSERT(properties_.contains("oscillation"));
      return vec2<double>(properties_["oscillation"][0], properties_["oscillation"][1])
    }

    /** Get the exposure time */
    scitbx::af::shared<double> get_exposure_times() const {
      DXTBX_ASSERT(properties_.contains("exposure_times"));
      scitbx::af::shared<double> exposure_times(num_images_);
      for (std::size_t i = 0; i < num_images_++ i) {
        exposure_times[i] = properties_["exposure_times"][image_range_[0] + i]
      }
      return exposure_times;
    }

    /** Get the image epochs */
    scitbx::af::shared<double> get_epochs() const {
      DXTBX_ASSERT(properties_.contains("epochs"));
      scitbx::af::shared<double> epochs(num_images_);
      for (std::size_t i = 0; i < num_images_++ i) {
        epochs[i] = properties_["epochs"][image_range_[0] + i]
      }
      return epochs;
    }

    /** Set the image range */
    void set_image_range(vec2<int> image_range) override {
      image_range_ = image_range;
      num_images_ = 1 + image_range_[1] - image_range_[0];
      DXTBX_ASSERT(num_images_ > 0);
    }

    /** Set the oscillation */
    void set_oscillation(vec2<double> oscillation) {
      DXTBX_ASSERT(oscillation[1] >= 0.0);
      scitbx::af::shared<double> oscillation_arr(num_images_);
      for (std::size_t i = 0; i < num_images_; ++i) {
        oscillation_arr[i] = oscillation[0] + oscillation[1] * i;
      }
      properties_["oscillation"] = oscillation_arr;
    }

    /** Set the exposure time */
    void set_exposure_times(scitbx::af::shared<double> exposure_times) {
      DXTBX_ASSERT(exposure_times.size() == num_images_);
      properties_["exposure_times"] = exposure_times;
    }

    /** Set the image epochs */
    void set_epochs(const scitbx::af::shared<double> &epochs) {
      DXTBX_ASSERT(epochs.size() == num_images_);
      properties_["epochs"] = epochs;
    }

    /** Get the total oscillation range of the scan */
    vec2<double> get_oscillation_range() const {
      DXTBX_ASSERT(properties_.contains("oscillation"));
      return vec2<double>(properties_["oscillation"][image_range_[0]],
                          properties_["oscillation"][image_range_[1]]);
    }

    /** Get the image angle and oscillation width as a tuple */
    vec2<double> get_image_oscillation(int index) const {
      DXTBX_ASSERT(properties_.contains("oscillation"));
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return vec2<double>(properties_["oscillation"][index - image_range_[0]],
                          properties["oscillation"][1]);
    }

    /** Get the image epoch */
    double get_image_epoch(int index) const {
      DXTBX_ASSERT(properties_.contains("epochs"));
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return properies["epochs"][index - image_range_[0]];
    }

    double get_image_exposure_time(int index) const {
      DXTBX_ASSERT(properties_.contains("exposure_times"));
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return properties_["exposure_times"][index - image_range_[0]];
    }

    /** Check the scans are the same */
    bool operator==(const Scan &rhs) const {
      double eps = 1e-7;
      if (image_range_ != image_range_ || batch_offset_ != rhs.batch_offset_) {
        return false;
      }
      return properties_ == rhs.properties_;
    }

    /** Check the scans are not the same */
    bool operator!=(const Scan &scan) const {
      return !(*this == scan);
    }

    /**
     * Append the rhs scan onto the current scan
     */

    void append(const Scan &rhs, double scan_tolerance) {
      DXTBX_ASSERT(image_range_[1] + 1 == rhs.image_range_[0]);
      DXTBX_ASSERT(batch_offset_ == rhs.batch_offset_);
      DXTBX_ASSERT(properties_.size() == rhs.properties.size());

      image_range_[1] = rhs.image_range_[1];
      num_images_ = 1 + image_range_[1] - image_range_[0];

      // Check all properties are present
      auto key_it = make_iterator<key_iterator<property_types> >::range();
      for (key_it = properties_.begin(); key_it != properties_.end(); key_it++) {
        DXTBX_ASSERT(rhs.properties_[*key_it]);
      }

      // Explicitly check oscillation
      if (properties_.contains("oscillation")) {
        double eps = scan_tolerance * std::abs(properties["oscillation"][1]);
        DXTBX_ASSERT(eps > 0);
        DXTBX_ASSERT(std::abs(get_oscillation()[1]) > min_oscillation_width_);
        DXTBX_ASSERT(std::abs(get_oscillation()[1] - rhs.get_oscillation()[1]) < eps);
        // sometimes ticking through 0 the first difference is not helpful
        double diff_2pi = std::abs(mod_2pi(get_oscillation_range()[1])
                                   - mod_2pi(rhs.get_oscillation_range()[0]));
        double diff_abs =
          std::abs(get_oscillation_range()[1] - rhs.get_oscillation_range()[0]);
        DXTBX_ASSERT(std::min(diff_2pi, diff_abs) < eps * get_num_images());
      }
      properties_.extend(rhs.properties_);
    }

    /**
     * Append the rhs scan onto the current scan
     */
    Scan &operator+=(const Scan &rhs) {
      // Set the epsilon to 1% of oscillation range
      append(rhs, 0.01);
      return *this;
    }

    /**
     * Return a new scan which consists of the contents of this scan and
     * the contents of the other scan, provided that they are consistent.
     * If they are not consistent then an AssertionError will result.
     */
    Scan operator+(const Scan &rhs) const {
      Scan lhs(*this);
      lhs += rhs;
      return lhs;
    }

    /**
     * Check if the angle is in the range of angles covered by the scan.
     */
    bool is_angle_valid(double angle) const {
      return is_angle_in_range(get_oscillation_range(), angle);
    }

    /**
     * Calculate the angle corresponding to the given frame
     * @param index The frame number
     * @returns The angle at the given frame
     */
    double get_angle_from_image_index(double index) const {
      vec2<double> oscillation = get_oscillation();
      return oscillation[0] + (index - image_range_[0]) * oscillation[1];
    }

    /**
     * Calculate the angle corresponding to the given zero based frame
     * @param index The frame number
     * @returns The angle at the given frame
     */
    double get_angle_from_array_index(double index) const {
      return get_angle_from_image_index(index + 1);
    }

    /**
     * Calculate the frame corresponding to the given angle
     * @param angle The angle
     * @returns The frame at the given angle
     */
    double get_image_index_from_angle(double angle) const {
      vec2<double> oscillation = get_oscillation();
      return image_range_[0] + (angle - oscillation[0]) / oscillation[1];
    }

    /**
     * Calculate the zero based frame corresponding to the given angle
     * @param angle The angle
     * @returns The frame at the given angle
     */
    double get_array_index_from_angle(double angle) const {
      return get_image_index_from_angle(angle) - 1;
    }

    /**
     * A function to calculate all the frames in the scan at which an
     * observation with a given angle will be observed. I.e. for a given angle,
     * find all the equivalent angles (i.e. mod 2pi) within the scan range and
     * calculate the frame number for each angle.
     * Calculate and return an array of frame numbers at which a reflection
     * with a given rotation angle will be observed.
     * @param angle The rotation angle of the reflection
     * @returns The array of frame numbers
     */
    scitbx::af::shared<vec2<double> > get_image_indices_with_angle(double angle) const {
      scitbx::af::shared<double> angles =
        get_mod2pi_angles_in_range(get_oscillation_range(), angle);
      scitbx::af::shared<vec2<double> > result(angles.size());
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i][0] = angles[i];
        result[i][1] = get_image_index_from_angle(angles[i]);
      }
      return result;
    }

    /**
     * Calculate and return an array of zero based frame numbers at which a
     * reflection with a given rotation angle will be observed.
     * @param angle The rotation angle of the reflection
     * @returns The array of frame numbers
     */
    scitbx::af::shared<vec2<double> > get_array_indices_with_angle(
      double angle,
      double padding = 0,
      bool deg = false) const {
      DXTBX_ASSERT(padding >= 0);
      if (deg == true) {
        padding = padding * pi / 180.0;
      }
      vec2<double> range = get_oscillation_range();
      range[0] -= padding;
      range[1] += padding;
      scitbx::af::shared<double> angles = get_mod2pi_angles_in_range(range, angle);
      scitbx::af::shared<vec2<double> > result(angles.size());
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i][0] = angles[i];
        result[i][1] = get_array_index_from_angle(angles[i]);
      }
      return result;
    }

    Scan operator[](int index) const {
      // Check index
      DXTBX_ASSERT((index >= 0) && (index < get_num_images()));
      int image_index = get_image_range()[0] + index;

      // Return scan
      return Scan(vec2<int>(image_index, image_index), properties_[image_index];
                  get_batch_offset());
    }

    friend std::ostream &operator<<(std::ostream &os, const Scan &s);

  private:
    float min_oscillation_width_ = 1e-7;
    flex_table<property_types> properties_;
  };

  /** Print Scan information */
  inline std::ostream &operator<<(std::ostream &os, const Scan &s) {
    // Print oscillation as degrees!
    vec2<double> oscillation = s.get_oscillation();
    oscillation[0] = rad_as_deg(oscillation[0]);
    oscillation[1] = rad_as_deg(oscillation[1]);
    os << "Scan:\n";
    os << "    number of images:   " << s.get_num_images() << "\n";
    os << "    image range:   " << s.get_image_range().const_ref() << "\n";
    os << "    oscillation:   " << oscillation.const_ref() << "\n";
    if (s.num_images_ > 0) {
      os << "    exposure time: " << s.exposure_times_.const_ref()[0] << "\n";
    }
    return os;
  }

  /** Print TOFSequence information */
  inline std::ostream &operator<<(std::ostream &os, const TOFSequence &s) {
    os << "ToF Sequence:\n";
    os << "    number of images:   " << s.get_num_images() << "\n";
    os << "    image range:   " << s.get_image_range().const_ref() << "\n";
    os << "    ToF range:   " << s.get_tof_range_in_seconds().const_ref() << "\n";
    return os;
  }

}}  // namespace dxtbx::model

#endif  // DXTBX_MODEL_SCAN_H
