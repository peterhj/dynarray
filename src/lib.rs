extern crate arrayidx;
extern crate memrepr;
#[cfg(feature = "mklml")] extern crate mklml_ffi;

use arrayidx::{IndexNd};
use memrepr::*;

use std::cell::{RefCell, Ref, RefMut};
use std::rc::{Rc};

pub mod linalg;

#[derive(Clone)]
pub struct Rearray<T> where T: Copy + 'static {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  memcopy:  Rc<RefCell<HeapMem<T>>>,
}

impl<T> Rearray<T> where T: ZeroBits {
  pub fn zeros(size: IndexNd) -> Rearray<T> {
    let mem = HeapMem::zeros(size.flat_len());
    let offset = IndexNd::zero(size.dim());
    let stride = size.to_packed_stride();
    Rearray{
      size,
      offset,
      stride,
      memcopy:  Rc::new(RefCell::new(mem)),
    }
  }
}

impl<T> Rearray<T> where T: Copy + 'static {
  pub fn size(&self) -> &IndexNd {
    &self.size
  }

  pub fn stride(&self) -> &IndexNd {
    &self.stride
  }

  pub fn is_packed(&self) -> bool {
    self.size.is_packed(&self.stride) && self.offset.is_zero()
  }

  pub fn shaped(&self, new_size: IndexNd) -> Rearray<T> {
    assert!(self.is_packed());
    assert_eq!(new_size.flat_len(), self.size.flat_len());
    let new_offset = IndexNd::zero(new_size.dim());
    let new_stride = new_size.to_packed_stride();
    Rearray{
      size:     new_size,
      offset:   new_offset,
      stride:   new_stride,
      memcopy:  self.memcopy.clone(),
    }
  }

  pub fn borrow<'a>(&'a self) -> RearrayRef<'a, T> {
    RearrayRef{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      memcopy:  self.memcopy.borrow(),
    }
  }

  pub fn borrow_mut<'a>(&'a self) -> RearrayRefMut<'a, T> {
    RearrayRefMut{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      memcopy:  self.memcopy.borrow_mut(),
    }
  }
}

pub struct RearrayRef<'a, T> where T: Copy + 'static {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  memcopy:  Ref<'a, HeapMem<T>>,
}

impl<'a, T> RearrayRef<'a, T> where T: Copy + 'static {
  pub fn flat_size(&self) -> usize {
    self.size.flat_len()
  }

  pub fn size(&self) -> &IndexNd {
    &self.size
  }

  pub fn stride(&self) -> &IndexNd {
    &self.stride
  }

  pub fn is_packed(&self) -> bool {
    self.size.is_packed(&self.stride) && self.offset.is_zero()
  }

  pub fn as_ptr(&self) -> *const T {
    self.memcopy.as_ptr()
  }
}

pub struct RearrayRefMut<'a, T> where T: Copy + 'static {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  memcopy:  RefMut<'a, HeapMem<T>>,
}

impl<'a, T> RearrayRefMut<'a, T> where T: Copy + 'static {
  pub fn flat_size(&self) -> usize {
    self.size.flat_len()
  }

  pub fn size(&self) -> &IndexNd {
    &self.size
  }

  pub fn stride(&self) -> &IndexNd {
    &self.stride
  }

  pub fn is_packed(&self) -> bool {
    self.size.is_packed(&self.stride) && self.offset.is_zero()
  }

  pub fn as_ptr_mut(&self) -> *mut T {
    self.memcopy.as_ptr_mut()
  }

  pub fn flat_map_mut<F: FnMut(&mut T)>(&mut self, mut func: F) {
    if self.is_packed() {
      let flen = self.flat_size();
      for x in self.memcopy.as_slice_mut().iter_mut().take(flen) {
        func(x);
      }
    } else {
      unimplemented!();
    }
  }
}
