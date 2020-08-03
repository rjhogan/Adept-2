

#ifndef AdeptGradientIndex_H
#define AdeptGradientIndex_H 1

#include <adept/Stack.h>

namespace adept {
  namespace internal {

    // Arrays inherit from this class to provide optional storage of
    // the gradient index of the first value of the array depending on
    // whether the array is active or not
    template <bool IsActive>
    struct GradientIndex {
      // Constructor used when linking to existing data where gradient
      // index is known
      GradientIndex(Index val = -9999) : value_(val) { }
      // Constructor used for fixed array objects where length is
      // known
      GradientIndex(Index n, bool) : value_(ADEPT_ACTIVE_STACK->register_gradients(n)) { }
      GradientIndex(Index val, Index offset) : value_(val+offset) { }
      Index get() const { return value_; }
      void set(Index val) { value_ = val; }
      void clear() { value_ = -9999; }
      template <typename Type>
      void set(const Type* data, const Storage<Type>* storage) {
	value_ = (storage->gradient_index() + (data - storage->data()));
      }
      void assert_inactive() {
	throw invalid_operation("Operation applied that is invalid with active arrays"
				ADEPT_EXCEPTION_LOCATION);
      }
      void unregister(Index n) { ADEPT_ACTIVE_STACK->unregister_gradients(value_, n); }
#ifdef ADEPT_MOVE_SEMANTICS
      void swap_value(GradientIndex& rhs) noexcept {
	Index tmp_value = rhs.get();
	rhs.set(value_);
	value_ = tmp_value;
      }
#endif
    private:
      Index value_;
    };

    template <>
    struct GradientIndex<false> {
      GradientIndex(Index val = -9999) { }
      GradientIndex(Index, bool) { }
      GradientIndex(Index val, Index offset) { }
      Index get() const { return -9999; }
      void set(Index val) { }
      void clear() { }
      template <typename Type>
      void set(const Type* data, const Storage<Type>* storage) { }
      void assert_inactive() { }
      void unregister(Index) { }
#ifdef ADEPT_MOVE_SEMANTICS
      void swap_value(GradientIndex& rhs) noexcept { }
#endif
    };

  };
};

#endif
