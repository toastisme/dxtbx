/*
 * scan.h
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

namespace dxtbx { namespace model {

  using scitbx::rad_as_deg;
  using scitbx::vec2;
  using scitbx::constants::pi;

  typedef std::map<std::string, scitbx::af::shared<vec2<int> > > ExpImgRangeMap;

  class ScanBase {
  public:
    ScanBase() : image_range_(0, 0), num_images_(0), batch_offset_(0) {}

    virtual ~ScanBase() {}

    virtual vec2<int> get_image_range() const {
      return image_range_;
    }

    /** Get the map, not exported to python **/
    virtual ExpImgRangeMap get_valid_image_ranges_map() const {
      return valid_image_ranges_;
    }

    /** Get the element for a given key if it exists, else return empty array**/
    virtual scitbx::af::shared<vec2<int> > get_valid_image_ranges_key(
      std::string i) const {
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

    virtual int get_batch_offset() const {
      return batch_offset_;
    }

    virtual int get_batch_for_image_index(int index) const {
      return index + batch_offset_;
    }

    virtual int get_batch_for_array_index(int index) const {
      return index + batch_offset_ + 1;
    }

    virtual vec2<int> get_batch_range() const {
      return vec2<int>(image_range_[0] + batch_offset_,
                       image_range_[1] + batch_offset_);
    }

    virtual vec2<int> get_array_range() const {
      return vec2<int>(image_range_[0] - 1, image_range_[1]);
    }

    virtual int get_num_images() const {
      return num_images_;
    }

    /** Set a list of valid image range tuples for experiment identifier 'i'**/
    virtual void set_valid_image_ranges_array(std::string i,
                                              scitbx::af::shared<vec2<int> > values) {
      for (std::size_t j = 0; j < values.size(); ++j) {
        vec2<int> pair = values[j];
        DXTBX_ASSERT(pair[0] >= image_range_[0]);
        DXTBX_ASSERT(pair[0] <= image_range_[1]);
        DXTBX_ASSERT(pair[1] >= image_range_[0]);
        DXTBX_ASSERT(pair[1] <= image_range_[1]);
      }
      valid_image_ranges_[i] = values;
    }

    virtual void set_image_range(vec2<int> image_range) {
      image_range_ = image_range;
      num_images_ = 1 + image_range_[1] - image_range_[0];
      DXTBX_ASSERT(num_images_ > 0);
    }

    virtual void set_batch_offset(int batch_offset) {
      batch_offset_ = batch_offset;
    }

    virtual bool is_image_index_valid(double index) const {
      return (image_range_[0] <= index && index <= image_range_[1]);
    }

    virtual bool is_batch_valid(int batch) const {
      vec2<int> batch_range = get_batch_range();
      return (batch_range[0] <= batch && batch <= batch_range[1]);
    }

    virtual bool is_array_index_valid(double index) const {
      return is_image_index_valid(index + 1);
    }

    virtual bool is_still() const {
      return false;
    }

  protected:
    vec2<int> image_range_;
    int num_images_;
    int batch_offset_;
    ExpImgRangeMap valid_image_ranges_; /** initialised as an empty map **/
  };

  class Scan : public ScanBase {
  public:
    Scan() : oscillation_(0.0, 0.0) {}

    /**
     * @param image_range The range of images covered by the scan
     * @param oscillation A tuple containing the start angle of the first image
     *                    and the oscillation range (the angular width) of each
     *                    frame
     * @param batch_offset A offset to add to the image number (for tracking of
     *                     unique batch numbers for multi-crystal datasets)
     */
    Scan(vec2<int> image_range, vec2<double> oscillation, int batch_offset = 0)
        : oscillation_(oscillation) {
      image_range_ = image_range;
      batch_offset_ = batch_offset;

      num_images_ = 1 + image_range_[1] - image_range_[0];
      DXTBX_ASSERT(num_images_ >= 0);

      scitbx::af::shared<double> exposure_times;
      scitbx::af::shared<double> epochs;
      exposure_times.reserve(num_images_);
      epochs.reserve(num_images_);
      for (int j = 0; j < num_images_; j++) {
        exposure_times.push_back(0.0);
        epochs.push_back(0.0);
      }
      exposure_times_ = exposure_times;
      epochs_ = epochs;
    }

    /**
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
        : oscillation_(oscillation), exposure_times_(exposure_times), epochs_(epochs) {
      image_range_ = image_range;
      batch_offset_ = batch_offset;
      num_images_ = 1 + image_range_[1] - image_range_[0];
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
        : oscillation_(rhs.oscillation_),
          exposure_times_(scitbx::af::reserve(rhs.exposure_times_.size())),
          epochs_(scitbx::af::reserve(rhs.epochs_.size())) {
      image_range_ = rhs.image_range_;
      valid_image_ranges_ = rhs.valid_image_ranges_;
      num_images_ = rhs.num_images_;
      batch_offset_ = rhs.batch_offset_;
      std::copy(rhs.epochs_.begin(), rhs.epochs_.end(), std::back_inserter(epochs_));
      std::copy(rhs.exposure_times_.begin(),
                rhs.exposure_times_.end(),
                std::back_inserter(exposure_times_));
    }

    virtual ~Scan() {}

    bool is_still() const {
      return std::abs(oscillation_[1]) < min_oscillation_width_;
    }

    vec2<double> get_oscillation() const {
      return oscillation_;
    }

    scitbx::af::shared<double> get_exposure_times() const {
      return exposure_times_;
    }

    scitbx::af::shared<double> get_epochs() const {
      return epochs_;
    }

    void set_image_range(vec2<int> image_range) {
      image_range_ = image_range;
      num_images_ = 1 + image_range_[1] - image_range_[0];
      epochs_.resize(num_images_);
      exposure_times_.resize(num_images_);
      DXTBX_ASSERT(num_images_ > 0);
    }

    void set_oscillation(vec2<double> oscillation) {
      DXTBX_ASSERT(oscillation[1] >= 0.0);
      oscillation_ = oscillation;
    }

    void set_exposure_times(scitbx::af::shared<double> exposure_times) {
      DXTBX_ASSERT(exposure_times.size() == num_images_);
      exposure_times_ = exposure_times;
    }

    void set_epochs(const scitbx::af::shared<double> &epochs) {
      DXTBX_ASSERT(epochs.size() == num_images_);
      epochs_ = epochs;
    }

    vec2<double> get_oscillation_range() const {
      return vec2<double>(oscillation_[0],
                          oscillation_[0] + num_images_ * oscillation_[1]);
    }

    /** Get the image angle and oscillation width as a tuple */
    vec2<double> get_image_oscillation(int index) const {
      return vec2<double>(oscillation_[0] + (index - image_range_[0]) * oscillation_[1],
                          oscillation_[1]);
    }

    double get_image_epoch(int index) const {
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return epochs_[index - image_range_[0]];
    }

    double get_image_exposure_time(int index) const {
      DXTBX_ASSERT(image_range_[0] <= index && index <= image_range_[1]);
      return exposure_times_[index - image_range_[0]];
    }

    bool operator==(const Scan &rhs) const {
      double eps = 1e-7;
      return image_range_ == rhs.image_range_ && batch_offset_ == rhs.batch_offset_
             && std::abs(oscillation_[0] - rhs.oscillation_[0]) < eps
             && std::abs(oscillation_[1] - rhs.oscillation_[1]) < eps
             && exposure_times_.const_ref().all_approx_equal(
               rhs.exposure_times_.const_ref(), eps)
             && epochs_.const_ref().all_approx_equal(rhs.epochs_.const_ref(), eps);
    }

    bool operator!=(const Scan &scan) const {
      return !(*this == scan);
    }

    bool operator<(const Scan &scan) const {
      return image_range_[0] < scan.image_range_[0];
    }

    bool operator<=(const Scan &scan) const {
      return image_range_[0] <= scan.image_range_[0];
    }

    bool operator>(const Scan &scan) const {
      return image_range_[0] > scan.image_range_[0];
    }

    bool operator>=(const Scan &scan) const {
      return image_range_[0] >= scan.image_range_[0];
    }

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
      DXTBX_ASSERT(std::abs(oscillation_[1]) > 0.0);
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
     * Calculate all the frames in the scan at which an
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
    scitbx::af::shared<double> exposure_times_;
    scitbx::af::shared<double> epochs_;
    double min_oscillation_width_ = 1e-7;
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

  /** A class to represent a time-of-flight histogram data as a sequence of 2D images*/
  class TOFSequence : public ScanBase {
  public:
    TOFSequence() : tof_in_seconds_(num_images_, 0), wavelengths_(num_images_, 0) {}

    /**
     * @param image_range The range of images covered by the sequence (inclusive)
     * @param tof The time-of-flight (s) of each image
     * @param wavelengths The wavelength (A) of each image
     * @param batch_offset An offset to add to the image number (for tracking
     *                      unique batch numbers for multi-crystal datasets)
     *
     */
    TOFSequence(vec2<int> image_range,
                const scitbx::af::shared<double> &tof,
                const scitbx::af::shared<double> &wavelengths,
                int batch_offset = 0)
        : tof_(tof), wavelengths_(wavelengths) {
      image_range_ = image_range;
      batch_offset_ = batch_offset;

      num_images_ = 1 + image_range_[1] - image_range_[0];
      DXTBX_ASSERT(num_images_ >= 0);
      DXTBX_ASSERT(tof_.size() == num_images_);
      DXTBX_ASSERT(wavelengths_.size() == num_images_);
    }

    virtual ~TOFSequence() {}

    scitbx::af::shared<double> get_tof() const {
      DXTBX_ASSERT(tof_.size() >= image_range_[0] - 1);
      DXTBX_ASSERT(tof_.size() >= image_range_[1] - 1);
      return scitbx::af::shared<double>(tof_.begin() + image_range_[0] - 1,
                                        tof_.begin() + image_range_[1]);
    }

    scitbx::af::shared<double> get_wavelengths() const {
      DXTBX_ASSERT(wavelengths_.size() >= image_range_[0] - 1);
      DXTBX_ASSERT(wavelengths_.size() >= image_range_[1] - 1);
      return scitbx::af::shared<double>(wavelengths_.begin() + image_range_[0] - 1,
                                        wavelengths_.begin() + image_range_[1]);
    }

    int get_num_tof_bins() const {
      return static_cast<int>(tof_.size());
    }

    vec2<double> get_tof_range() const {
      return vec2<double>(tof_[image_range_[0] - 1], tof_[image_range_[1] - 1]);
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
             && wavelengths_.const_ref().all_approx_equal(rhs.wavelengths_.const_ref())
             && tof_.const_ref().all_approx_equal(rhs.tof_.const_ref(), eps);
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
      return TOFSequence(
        vec2<int>(image_index, image_index),
        scitbx::af::shared<double>(tof_.begin() + image_range_[0] - 1,
                                   tof_.begin() + image_range_[1]),
        scitbx::af::shared<double>(wavelengths_.begin() + image_range_[0] - 1,
                                   wavelengths_.begin() + image_range_[1]),
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
    scitbx::af::shared<double> tof_;          // (s)
    scitbx::af::shared<double> wavelengths_;  // (A)
  };

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
