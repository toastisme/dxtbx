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
#include <scitbx/constants.h>
#include <dxtbx/error.h>
#include "scan_helpers.h"
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace dxtbx { namespace model {

  using scitbx::rad_as_deg;
  using scitbx::vec2;
  using scitbx::constants::pi;

  typedef std::map<std::string, scitbx::af::shared<vec2<int> > > ExpImgRangeMap;

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
          wavelengths_(num_images_, 0),
          frame_to_tof_(boost::none),
          frame_to_wavelength_(boost::none),
          tof_to_frame_(boost::none),
          wavelength_to_frame_(boost::none) {}

    /**
     * @param image_range The range of images covered by the sequence
     * @param tof_in_seconds The ToF values of each image
     * @param batch_offset An offset to add to the image number (for tracking
     *                      unique batch numbers for multi-crystal datasets)
k     *
     */
    TOFSequence(vec2<int> image_range,
                const scitbx::af::shared<double> &tof_in_seconds,
                const scitbx::af::shared<double> &wavelengths,
                int batch_offset = 0)
        : Sequence(image_range, batch_offset),
          tof_in_seconds_(tof_in_seconds),
          wavelengths_(wavelengths),
          frame_to_tof_(boost::none),
          frame_to_wavelength_(boost::none),
          tof_to_frame_(boost::none),
          wavelength_to_frame_(boost::none) {
      create_splines();
    }

    virtual ~TOFSequence() {}

    TOFSequence(const TOFSequence &rhs)
        : Sequence(rhs.image_range_, rhs.batch_offset_),
          tof_in_seconds_(scitbx::af::reserve(rhs.tof_in_seconds_.size())),
          wavelengths_(scitbx::af::reserve(rhs.wavelengths_.size())),
          frame_to_tof_(boost::none),
          frame_to_wavelength_(boost::none),
          tof_to_frame_(boost::none),
          wavelength_to_frame_(boost::none) {
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

    double get_tof_wavelength_in_ang(double L, double tof) const {
      return ((scitbx::constants::Planck * tof) / (scitbx::constants::m_n * L))
             * std::pow(10, 10);
    }

    scitbx::af::shared<double> get_tof_wavelengths_in_ang(
      const scitbx::af::shared<double> L,
      const scitbx::af::shared<double> tof) const {
      DXTBX_ASSERT(L.size() == tof.size());
      scitbx::af::shared<double> wavelengths;
      for (std::size_t i = 0; i < tof.size(); ++i) {
        wavelengths.push_back(get_tof_wavelength_in_ang(L[i], tof[i]));
      }
      return wavelengths;
    }

    void create_splines() {
      if (wavelengths_.size() < 5) {
        return;
      }

      std::vector<double> frames = get_frames_vec();
      std::vector<double> wavelengths = get_wavelengths_vec();
      std::vector<double> tof = get_tof_in_seconds_vec();

      DXTBX_ASSERT(frames.size() > 0);
      DXTBX_ASSERT(wavelengths.size() > 0);
      DXTBX_ASSERT(tof.size() > 0);

      frame_to_wavelength_ = get_barycentric_spline(frames, wavelengths);
      frame_to_tof_ = get_barycentric_spline(frames, tof);
      tof_to_frame_ = get_barycentric_spline(tof, frames);
      wavelength_to_frame_ = get_barycentric_spline(wavelengths, frames);
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

    boost::math::interpolators::cardinal_cubic_b_spline<double> get_cubic_b_spline(
      scitbx::af::shared<double> data) {
      double t0 = 0;
      double h = 0.01;
      boost::math::interpolators::cardinal_cubic_b_spline<double> spline(
        data.begin(), data.end(), t0, h);
      return spline;
    }

    boost::math::interpolators::barycentric_rational<double> get_barycentric_spline(
      std::vector<double> x,
      std::vector<double> y) {
      boost::math::interpolators::barycentric_rational<double> spline(
        x.data(), y.data(), x.size());
      return spline;
    }

    double get_wavelength_from_frame(const double frame) const {
      DXTBX_ASSERT(frame_to_wavelength_);
      return frame_to_wavelength_.get()(frame);
    }

    double get_tof_from_frame(const double frame) const {
      DXTBX_ASSERT(frame_to_tof_);
      return frame_to_tof_.get()(frame);
    }

    scitbx::af::shared<double> get_tof_from_frames(
      const scitbx::af::shared<double> frames) const {
      DXTBX_ASSERT(frame_to_tof_);
      scitbx::af::shared<double> tof;
      for (std::size_t i = 0; i < frames.size(); ++i) {
        tof.push_back(frame_to_tof_.get()(frames[i]));
      }
      return tof;
    }

    double get_frame_from_wavelength(const double wavelength) const {
      DXTBX_ASSERT(wavelength_to_frame_);
      return wavelength_to_frame_.get()(wavelength);
    }

    double get_frame_from_tof(const double tof) const {
      DXTBX_ASSERT(tof_to_frame_);
      return tof_to_frame_.get()(tof);
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
    std::vector<double> get_frames_vec() const {
      DXTBX_ASSERT(tof_in_seconds_.size() > 0);
      std::vector<double> frames;
      for (std::size_t i = 0; i < tof_in_seconds_.size(); ++i) {
        frames.push_back(i + 1);
      }
      return frames;
    }

    std::vector<double> get_tof_in_seconds_vec() const {
      DXTBX_ASSERT(tof_in_seconds_.size() > 0);
      std::vector<double> tof_in_seconds;
      for (std::size_t i = 0; i < tof_in_seconds_.size(); ++i) {
        tof_in_seconds.push_back(tof_in_seconds_[i]);
      }
      return tof_in_seconds;
    }

    std::vector<double> get_wavelengths_vec() const {
      DXTBX_ASSERT(wavelengths_.size() > 0);
      std::vector<double> wavelengths;
      for (std::size_t i = 0; i < wavelengths_.size(); ++i) {
        wavelengths.push_back(wavelengths_[i]);
      }
      return wavelengths;
    }
    scitbx::af::shared<double> tof_in_seconds_;
    scitbx::af::shared<double> wavelengths_;
    boost::optional<boost::math::interpolators::barycentric_rational<double> >
      wavelength_to_frame_;
    boost::optional<boost::math::interpolators::barycentric_rational<double> >
      tof_to_frame_;
    boost::optional<boost::math::interpolators::barycentric_rational<double> >
      frame_to_wavelength_;
    boost::optional<boost::math::interpolators::barycentric_rational<double> >
      frame_to_tof_;
  };

  /** A class to represent a scan */
  class Scan : public Sequence {
  public:
    /** The default constructor */
    Scan() : Sequence(vec2<int>{0, 0}, 0), oscillation_(0.0, 0.0), is_still_(false) {}

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
        : Sequence(image_range, batch_offset),
          oscillation_(oscillation),
          exposure_times_(num_images_, 0.0),
          epochs_(num_images_, 0.0) {
      DXTBX_ASSERT(num_images_ >= 0);
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
        : Sequence(image_range, batch_offset),
          oscillation_(oscillation),
          exposure_times_(exposure_times),
          epochs_(epochs) {
      DXTBX_ASSERT(num_images_ >= 0);

      if (exposure_times_.size() == 1 && num_images_ > 1) {
        // assume same exposure time for all images - there is
        // probably a better way of coding this...
        scitbx::af::shared<double> expanded_exposure_times;
        expanded_exposure_times.reserve(num_images_);
        for (int j = 0; j < num_images_; j++) {
          expanded_exposure_times.push_back(exposure_times[0]);
          exposure_times_ = expanded_exposure_times;
        }
      }
      DXTBX_ASSERT(exposure_times_.size() == num_images_);
      DXTBX_ASSERT(epochs_.size() == num_images_);
      DXTBX_ASSERT(oscillation_[1] >= 0.0);
    }

    /** Copy */
    Scan(const Scan &rhs)
        : Sequence(rhs.image_range_, rhs.batch_offset_),
          oscillation_(rhs.oscillation_),
          is_still_(rhs.is_still_),
          exposure_times_(scitbx::af::reserve(rhs.exposure_times_.size())),
          epochs_(scitbx::af::reserve(rhs.epochs_.size())) {
      std::copy(rhs.epochs_.begin(), rhs.epochs_.end(), std::back_inserter(epochs_));
      std::copy(rhs.exposure_times_.begin(),
                rhs.exposure_times_.end(),
                std::back_inserter(exposure_times_));
    }

    /** Virtual destructor */
    virtual ~Scan() {}

    /** Get the still flag */
    bool is_still() const {
      return std::abs(oscillation_[1]) < min_oscillation_width_;
    }

    /** Get the oscillation */
    vec2<double> get_oscillation() const {
      return oscillation_;
    }

    /** Get the exposure time */
    scitbx::af::shared<double> get_exposure_times() const {
      return exposure_times_;
    }

    /** Get the image epochs */
    scitbx::af::shared<double> get_epochs() const {
      return epochs_;
    }

    /** Set the image range */
    void set_image_range(vec2<int> image_range) override {
      image_range_ = image_range;
      num_images_ = 1 + image_range_[1] - image_range_[0];
      epochs_.resize(num_images_);
      exposure_times_.resize(num_images_);
      DXTBX_ASSERT(num_images_ > 0);
    }

    /** Set the oscillation */
    void set_oscillation(vec2<double> oscillation) {
      DXTBX_ASSERT(oscillation[1] >= 0.0);
      oscillation_ = oscillation;
    }

    /** Set the exposure time */
    void set_exposure_times(scitbx::af::shared<double> exposure_times) {
      DXTBX_ASSERT(exposure_times.size() == num_images_);
      exposure_times_ = exposure_times;
    }

    /** Set the image epochs */
    void set_epochs(const scitbx::af::shared<double> &epochs) {
      DXTBX_ASSERT(epochs.size() == num_images_);
      epochs_ = epochs;
    }

    /** Get the total oscillation range of the scan */
    vec2<double> get_oscillation_range() const {
      return vec2<double>(oscillation_[0],
                          oscillation_[0] + num_images_ * oscillation_[1]);
    }

    /** Get the image angle and oscillation width as a tuple */
    vec2<double> get_image_oscillation(int index) const {
      return vec2<double>(oscillation_[0] + (index - image_range_[0]) * oscillation_[1],
                          oscillation_[1]);
    }

    /** Get the image epoch */
    double get_image_epoch(int index) const {
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return epochs_[index - image_range_[0]];
    }

    double get_image_exposure_time(int index) const {
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return exposure_times_[index - image_range_[0]];
    }

    /** Check the scans are the same */
    bool operator==(const Scan &rhs) const {
      double eps = 1e-7;
      return image_range_ == rhs.image_range_ && batch_offset_ == rhs.batch_offset_
             && std::abs(oscillation_[0] - rhs.oscillation_[0]) < eps
             && std::abs(oscillation_[1] - rhs.oscillation_[1]) < eps
             && exposure_times_.const_ref().all_approx_equal(
               rhs.exposure_times_.const_ref(), eps)
             && epochs_.const_ref().all_approx_equal(rhs.epochs_.const_ref(), eps);
    }

    /** Check the scans are not the same */
    bool operator!=(const Scan &scan) const {
      return !(*this == scan);
    }

    /**
     * Append the rhs scan onto the current scan
     */

    void append(const Scan &rhs, double scan_tolerance) {
      DXTBX_ASSERT(is_still() == rhs.is_still());
      if (is_still()) {
        append_still(rhs);
      } else {
        append_rotation(rhs, scan_tolerance);
      }
    }

    void append_still(const Scan &rhs) {
      DXTBX_ASSERT(image_range_[1] + 1 == rhs.image_range_[0]);
      DXTBX_ASSERT(batch_offset_ == rhs.batch_offset_);
      image_range_[1] = rhs.image_range_[1];
      num_images_ = 1 + image_range_[1] - image_range_[0];
      exposure_times_.reserve(exposure_times_.size() + exposure_times_.size());
      epochs_.reserve(epochs_.size() + epochs_.size());
      std::copy(rhs.exposure_times_.begin(),
                rhs.exposure_times_.end(),
                std::back_inserter(exposure_times_));
      std::copy(rhs.epochs_.begin(), rhs.epochs_.end(), std::back_inserter(epochs_));
    }

    void append_rotation(const Scan &rhs, double scan_tolerance) {
      double eps = scan_tolerance * std::abs(oscillation_[1]);
      DXTBX_ASSERT(eps > 0);
      DXTBX_ASSERT(std::abs(oscillation_[1]) > min_oscillation_width_);
      DXTBX_ASSERT(image_range_[1] + 1 == rhs.image_range_[0]);
      DXTBX_ASSERT(std::abs(oscillation_[1] - rhs.oscillation_[1]) < eps);
      DXTBX_ASSERT(batch_offset_ == rhs.batch_offset_);
      // sometimes ticking through 0 the first difference is not helpful
      double diff_2pi = std::abs(mod_2pi(get_oscillation_range()[1])
                                 - mod_2pi(rhs.get_oscillation_range()[0]));
      double diff_abs =
        std::abs(get_oscillation_range()[1] - rhs.get_oscillation_range()[0]);
      DXTBX_ASSERT(std::min(diff_2pi, diff_abs) < eps * get_num_images());
      image_range_[1] = rhs.image_range_[1];
      num_images_ = 1 + image_range_[1] - image_range_[0];
      exposure_times_.reserve(exposure_times_.size() + exposure_times_.size());
      epochs_.reserve(epochs_.size() + epochs_.size());
      std::copy(rhs.exposure_times_.begin(),
                rhs.exposure_times_.end(),
                std::back_inserter(exposure_times_));
      std::copy(rhs.epochs_.begin(), rhs.epochs_.end(), std::back_inserter(epochs_));
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
      return oscillation_[0] + (index - image_range_[0]) * oscillation_[1];
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
      return image_range_[0] + (angle - oscillation_[0]) / oscillation_[1];
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

      // Create the new epoch array
      scitbx::af::shared<double> new_epochs(1);
      new_epochs[0] = get_image_epoch(image_index);
      scitbx::af::shared<double> new_exposure_times(1);
      new_exposure_times[0] = get_image_exposure_time(image_index);

      // Return scan
      return Scan(vec2<int>(image_index, image_index),
                  get_image_oscillation(image_index),
                  new_exposure_times,
                  new_epochs,
                  get_batch_offset());
    }

    friend std::ostream &operator<<(std::ostream &os, const Scan &s);

  private:
    vec2<double> oscillation_;
    float min_oscillation_width_ = 1e-7;
    bool is_still_;
    scitbx::af::shared<double> exposure_times_;
    scitbx::af::shared<double> epochs_;
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
