
#ifndef DXTBX_MODEL_SCAN_PROPERTIES_H
#define DXTBX_MODEL_SCAN_PROPERTIES_H

#include <scitbx/array_family/shared.h>
#include <map>
#include <dxtbx/error.h>

namespace dxtbx { namespace model {

  template <typename VarientType>
  class ScanProperties {
  public:
    typedef std::map<std::string, VarientType> map_type;
    typedef typename map_type::key_type key_type;
    typedef typename map_type::mapped_type mapped_type;
    typedef typename map_type::value_type map_value_type;
    typedef typename map_type::iterator iterator;
    typedef typename map_type::const_iterator const_iterator;
    typedef typename map_type::size_type size_type;
    ScanProperties() {}

    ScanProperties(const ScanProperties &rhs) : properties_(rhs.properties_) {}

    ScanProperties operator[](int index) const {}

    void add(const std::string &key, const scitbx::af::shared<double> &value) {
      if (properties_->size() != 0) {
        DXTBX_ASSERT(value.size() == properties_->size());
      }
      properties_[key] = value;
    }

    bool contains(const std::string &key) const {
      std::map<std::string, scitbx::af::shared<double> >::iterator it;
      it = properties_->find(key);
      return (it != properties_.end());
    }

    bool contains(const scitbx::af::shared<std::string> &keys) const {
      for (std::size_t i = 0; i < keys.size(); ++i) {
        if (!contains(keys[i])) {
          return false;
        }
      }
      return true;
    }

    scitbx::af::shared<double> get(const std::string &key) const {
      std::map<std::string, scitbx::af::shared<double> >::iterator it;
      it = properties_->find(key);
      DXTBX_ASSERT(it != properties_.end());
      return it->second;
    }

    double get_at_idx(const std::string &key, const int &idx) const {
      DXTBX_ASSERT(idx >= 0 && idx < properties_->size());
      return get(key)[idx];
    }

    void resize(const int &new_size) {
      for (auto &kvp : properties_) {
        kvp->second.resize(new_size);
      }
    }

    void remove(const std::string &key) {
      properties_->erase(key);
    }

  private:
    boost::shared_ptr<map_type> properties_;
  };

}}      // namespace dxtbx::model
#endif  // DXTBX_MODEL_SCAN_PROPERTIES_H