extern crate arrayidx;
#[cfg(feature = "mklml")]
extern crate mklml_sys;
extern crate podmem;

use arrayidx::{IndexNd};
use podmem::{PodRegion, PodRegionMut, ZeroBits};
use podmem::heap::{HeapMem};

use std::cell::{RefCell, Ref};
use std::rc::{Rc};

pub mod linalg;

#[derive(Clone)]
enum RearrayRepr<T: Copy + 'static> {
  Dense(Rc<DenseRearrayRepr<T>>),
  Mini(IndexNd, MiniRearrayRepr<T>),
}

impl<T> RearrayRepr<T> where T: Copy + 'static {
  fn size(&self) -> &IndexNd {
    match self {
      &RearrayRepr::Dense(ref dense) => &dense.size,
      &RearrayRepr::Mini(ref size, _) => size,
    }
  }

  fn stride(&self) -> IndexNd {
    match self {
      &RearrayRepr::Dense(ref dense) => dense.stride.clone(),
      &RearrayRepr::Mini(ref size, _) => size.to_packed_stride(),
    }
  }

  fn is_packed(&self) -> bool {
    match self {
      &RearrayRepr::Dense(ref dense) => {
        dense.size.is_packed(&dense.stride) && dense.offset.is_zero()
      }
      &RearrayRepr::Mini(..) => true,
    }
  }
}

impl<T> RearrayRepr<T> where T: ZeroBits + 'static {
  fn densify(&mut self) {
    match self {
      &mut RearrayRepr::Dense(_) => {}
      &mut RearrayRepr::Mini(ref size, ref mini) => {
        let new_dense = match mini {
          &MiniRearrayRepr::Zeros => {
            DenseRearrayRepr::zeros(size.clone())
          }
          &MiniRearrayRepr::Constants(_c) => {
            // TODO
            // FIXME: fill with `c`.
            //let mut new_dense = unsafe { DenseRearrayRepr::<T>::alloc(size.clone()) };
            unimplemented!();
          }
        };
        *self = RearrayRepr::Dense(Rc::new(new_dense));
      }
    }
  }
}

pub struct DenseRearrayRepr<T: Copy + 'static> {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  mem:      HeapMem<T>,
}

impl<T> Clone for DenseRearrayRepr<T> where T: Copy + 'static {
  fn clone(&self) -> DenseRearrayRepr<T> {
    // FIXME: preserve layout.
    let mut new_mem = unsafe { HeapMem::alloc(self.size.flat_len()) };
    new_mem.as_slice_mut().copy_from_slice(self.mem.as_slice());
    let offset = IndexNd::zero(self.size.dim());
    let stride = self.size.to_packed_stride();
    DenseRearrayRepr{
      size: self.size.clone(),
      offset,
      stride,
      mem:  new_mem,
    }
  }
}

impl<T> DenseRearrayRepr<T> where T: ZeroBits + 'static {
  pub fn zeros(size: IndexNd) -> DenseRearrayRepr<T> {
    let mem = HeapMem::zeros(size.flat_len());
    let offset = IndexNd::zero(size.dim());
    let stride = size.to_packed_stride();
    DenseRearrayRepr{
      size,
      offset,
      stride,
      mem,
    }
  }
}

impl<T> DenseRearrayRepr<T> where T: Copy + 'static {
  pub unsafe fn alloc(size: IndexNd) -> DenseRearrayRepr<T> {
    let mem = HeapMem::alloc(size.flat_len());
    let offset = IndexNd::zero(size.dim());
    let stride = size.to_packed_stride();
    DenseRearrayRepr{
      size,
      offset,
      stride,
      mem,
    }
  }

  pub fn copy_from(&mut self, src: &DenseRearrayRepr<T>) {
    self.mem.as_slice_mut().copy_from_slice(src.mem.as_slice());
  }
}

#[derive(Clone, Copy)]
pub enum MiniRearrayRepr<T: Copy + 'static> {
  Zeros,
  Constants(T),
}

pub struct Rearray<T: Copy + 'static> {
  repr: RefCell<RearrayRepr<T>>,
}

impl<T> Clone for Rearray<T> where T: Copy + 'static {
  fn clone(&self) -> Rearray<T> {
    let repr = self.repr.try_borrow().unwrap_or_else(|_| panic!("bug"));
    Rearray{
      repr: RefCell::new(repr.clone()),
    }
  }
}

impl<T> Rearray<T> where T: Copy + 'static {
  pub fn zeros(size: IndexNd) -> Rearray<T> {
    Rearray{
      repr: RefCell::new(RearrayRepr::Mini(size, MiniRearrayRepr::Zeros)),
    }
  }

  pub fn size(&self) -> IndexNd {
    let repr = self.repr.try_borrow().unwrap_or_else(|_| panic!("bug"));
    repr.size().clone()
  }

  pub fn stride(&self) -> IndexNd {
    let repr = self.repr.try_borrow().unwrap_or_else(|_| panic!("bug"));
    repr.stride()
  }

  pub fn is_packed(&self) -> bool {
    let repr = self.repr.try_borrow().unwrap_or_else(|_| panic!("bug"));
    repr.is_packed()
  }
}

impl<T> Rearray<T> where T: ZeroBits + 'static {
  pub fn view<'a>(&'a self) -> RearrayView<'a, T> {
    let is_dense_orig = {
      let repr = self.repr.try_borrow().unwrap_or_else(|_| panic!("bug"));
      match &*repr {
        &RearrayRepr::Dense(_) => true,
        _ => false,
      }
    };
    if !is_dense_orig {
      let mut repr = self.repr.try_borrow_mut().unwrap_or_else(|_| panic!("bug"));
      repr.densify();
    }
    let repr = self.repr.try_borrow().unwrap_or_else(|_| panic!("bug"));
    let (size, offset, stride) = match &*repr {
      &RearrayRepr::Dense(ref dense) => {
        let size = dense.size.clone();
        let offset = dense.offset.clone();
        let stride = dense.stride.clone();
        (size, offset, stride)
      }
      _ => unreachable!(),
    };
    let mem = Ref::map(repr, |repr| match repr {
      &RearrayRepr::Dense(ref dense) => &dense.mem,
      _ => unreachable!(),
    });
    RearrayView{
      size,
      offset,
      stride,
      mem,
    }
  }

  pub fn view_mut<'a>(&'a mut self) -> RearrayViewMut<'a, T> {
    let repr = self.repr.get_mut();
    repr.densify();
    match repr {
      &mut RearrayRepr::Dense(ref mut dense) => {
        let size = dense.size.clone();
        let offset = dense.offset.clone();
        let stride = dense.stride.clone();
        let owned_dense = Rc::make_mut(dense);
        RearrayViewMut{
          size,
          offset,
          stride,
          mem:  &mut owned_dense.mem,
        }
      }
      _ => unimplemented!(),
    }
  }
}

pub struct RearrayView<'a, T> where T: Copy + 'static {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  mem:      Ref<'a, HeapMem<T>>,
}

impl<'a, T> RearrayView<'a, T> where T: Copy + 'static {
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
    (&*self.mem).as_ptr()
  }
}

pub struct RearrayViewMut<'a, T> where T: Copy + 'static {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  mem:      &'a mut HeapMem<T>,
}

impl<'a, T> RearrayViewMut<'a, T> where T: Copy + 'static {
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
    (&*self.mem).as_ptr_mut()
  }
}
