extern crate arrayidx;
extern crate memrepr;
#[cfg(feature = "mklml")] extern crate mklml_ffi;

use arrayidx::{IndexNd};
use memrepr::*;

use std::cell::{RefCell, Ref, RefMut};
use std::rc::{Rc};

//#[cfg(feature = "gpu")] pub mod gpu;
pub mod linalg;

#[derive(Clone)]
pub struct DynArray<T> where T: Copy {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  memcopy:  Rc<RefCell<HeapMem<T>>>,
}

impl<T> DynArray<T> where T: ZeroBits {
  pub fn zeros(size: IndexNd) -> Self {
    let mem = HeapMem::zeros(size.flat_len());
    let offset = IndexNd::zero(size.dim());
    let stride = size.to_packed_stride();
    DynArray{
      size,
      offset,
      stride,
      memcopy:  Rc::new(RefCell::new(mem)),
    }
  }
}

impl<T> DynArray<T> where T: Copy {
  pub fn size(&self) -> &IndexNd {
    &self.size
  }

  pub fn stride(&self) -> &IndexNd {
    &self.stride
  }

  pub fn is_packed(&self) -> bool {
    self.size.is_packed(&self.stride) && self.offset.is_zero()
  }

  pub fn shaped(&self, new_size: IndexNd) -> Self {
    assert_eq!(new_size.flat_len(), self.size.flat_len());
    assert!(self.is_packed());
    let new_offset = IndexNd::zero(new_size.dim());
    let new_stride = new_size.to_packed_stride();
    DynArray{
      size:     new_size,
      offset:   new_offset,
      stride:   new_stride,
      memcopy:  self.memcopy.clone(),
    }
  }

  pub fn borrow<'a>(&'a self) -> DynArrayRef<'a, T> {
    DynArrayRef{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      memcopy:  self.memcopy.borrow(),
    }
  }

  pub fn borrow_mut<'a>(&'a self) -> DynArrayRefMut<'a, T> {
    DynArrayRefMut{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      memcopy:  self.memcopy.borrow_mut(),
    }
  }
}

pub struct DynArrayRef<'a, T> where T: Copy {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  memcopy:  Ref<'a, HeapMem<T>>,
}

impl<'a, T> DynArrayRef<'a, T> where T: Copy {
  pub fn size(&self) -> &IndexNd {
    &self.size
  }

  pub fn stride(&self) -> &IndexNd {
    &self.stride
  }

  pub fn is_packed(&self) -> bool {
    self.size.is_packed(&self.stride) && self.offset.is_zero()
  }

  pub unsafe fn as_ptr(&self) -> *const T {
    self.memcopy.as_ptr()
  }
}

pub struct DynArrayRefMut<'a, T> where T: Copy {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  memcopy:  RefMut<'a, HeapMem<T>>,
}

impl<'a, T> DynArrayRefMut<'a, T> where T: Copy {
  pub fn size(&self) -> &IndexNd {
    &self.size
  }

  pub fn stride(&self) -> &IndexNd {
    &self.stride
  }

  pub fn is_packed(&self) -> bool {
    self.size.is_packed(&self.stride) && self.offset.is_zero()
  }

  pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
    self.memcopy.as_mut_ptr()
  }
}
