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

pub trait Layout {}

pub enum Dense {}

impl Layout for Dense {}

pub trait View<'a, L: Layout> {
  type V;

  fn view(&'a self) -> Self::V;
}

pub trait ViewMut<'a, L: Layout> {
  type VM;

  fn view_mut(&'a mut self) -> Self::VM;
}

#[derive(Clone)]
enum RearrayRepr<T: Copy + 'static> {
  Dense(Rc<DenseRearray<T>>),
  ScalarRep(IndexNd, Scalar),
}

impl<T> RearrayRepr<T> where T: Copy + 'static {
  fn size(&self) -> &IndexNd {
    match self {
      &RearrayRepr::Dense(ref dense) => &dense.size,
      &RearrayRepr::ScalarRep(ref size, _) => size,
    }
  }

  fn stride(&self) -> IndexNd {
    match self {
      &RearrayRepr::Dense(ref dense) => dense.stride.clone(),
      &RearrayRepr::ScalarRep(ref size, _) => size.to_packed_stride(),
    }
  }

  fn is_packed(&self) -> bool {
    match self {
      &RearrayRepr::Dense(ref dense) => {
        dense.size.is_packed(&dense.stride) && dense.offset.is_zero()
      }
      &RearrayRepr::ScalarRep(..) => true,
    }
  }
}

impl<T> RearrayRepr<T> where T: ZeroBits + 'static {
  fn densify(&mut self) {
    match self {
      &mut RearrayRepr::Dense(_) => {}
      &mut RearrayRepr::ScalarRep(ref size, ref scalar) => {
        let new_dense = match scalar {
          &Scalar::Zero => {
            DenseRearray::zeros(size.clone())
          }
        };
        *self = RearrayRepr::Dense(Rc::new(new_dense));
      }
    }
  }
}

pub struct DenseRearray<T: Copy + 'static> {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  mem:      HeapMem<T>,
}

impl<T> Clone for DenseRearray<T> where T: Copy + 'static {
  fn clone(&self) -> DenseRearray<T> {
    // FIXME: preserve layout.
    let mut new_mem = unsafe { HeapMem::alloc(self.size.flat_len()) };
    new_mem.as_slice_mut().copy_from_slice(self.mem.as_slice());
    let offset = IndexNd::zero(self.size.dim());
    let stride = self.size.to_packed_stride();
    DenseRearray{
      size: self.size.clone(),
      offset,
      stride,
      mem:  new_mem,
    }
  }
}

impl<T> DenseRearray<T> where T: ZeroBits + 'static {
  pub fn zeros(size: IndexNd) -> DenseRearray<T> {
    let mem = HeapMem::zeros(size.flat_len());
    let offset = IndexNd::zero(size.dim());
    let stride = size.to_packed_stride();
    DenseRearray{
      size,
      offset,
      stride,
      mem,
    }
  }
}

impl<T> DenseRearray<T> where T: Copy + 'static {
  pub unsafe fn alloc(size: IndexNd) -> DenseRearray<T> {
    let mem = HeapMem::alloc(size.flat_len());
    let offset = IndexNd::zero(size.dim());
    let stride = size.to_packed_stride();
    DenseRearray{
      size,
      offset,
      stride,
      mem,
    }
  }

  pub fn copy_from(&mut self, src: &DenseRearray<T>) {
    self.mem.as_slice_mut().copy_from_slice(src.mem.as_slice());
  }
}

pub struct DenseRearrayView<'a, T> where T: Copy + 'static {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  mem:      Ref<'a, HeapMem<T>>,
}

impl<'a, T> DenseRearrayView<'a, T> where T: Copy + 'static {
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

pub struct DenseRearrayViewMut<'a, T> where T: Copy + 'static {
  size:     IndexNd,
  offset:   IndexNd,
  stride:   IndexNd,
  mem:      &'a mut HeapMem<T>,
}

impl<'a, T> DenseRearrayViewMut<'a, T> where T: Copy + 'static {
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

  pub fn as_ptr_mut(&self) -> *mut T {
    (&*self.mem).as_ptr_mut()
  }
}

#[derive(Clone, Copy)]
/*pub enum Scalar<T: Copy + 'static> {*/
pub enum Scalar {
  Zero,
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
      repr: RefCell::new(RearrayRepr::ScalarRep(size, Scalar::Zero)),
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
  pub fn dense_view<'a>(&'a self) -> DenseRearrayView<'a, T> {
    <Rearray<T> as View<'a, Dense>>::view(self)
  }

  pub fn dense_view_mut<'a>(&'a mut self) -> DenseRearrayViewMut<'a, T> {
    <Rearray<T> as ViewMut<'a, Dense>>::view_mut(self)
  }
}

impl<'a, T> View<'a, Dense> for Rearray<T> where T: ZeroBits + 'static {
  type V = DenseRearrayView<'a, T>;

  fn view(&'a self) -> DenseRearrayView<'a, T> {
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
    DenseRearrayView{
      size,
      offset,
      stride,
      mem,
    }
  }
}

impl<'a, T> ViewMut<'a, Dense> for Rearray<T> where T: ZeroBits + 'static {
  type VM = DenseRearrayViewMut<'a, T>;

  fn view_mut(&'a mut self) -> DenseRearrayViewMut<'a, T> {
    let repr = self.repr.get_mut();
    repr.densify();
    match repr {
      &mut RearrayRepr::Dense(ref mut dense) => {
        let size = dense.size.clone();
        let offset = dense.offset.clone();
        let stride = dense.stride.clone();
        let owned_dense = Rc::make_mut(dense);
        DenseRearrayViewMut{
          size,
          offset,
          stride,
          mem:  &mut owned_dense.mem,
        }
      }
      _ => unreachable!(),
    }
  }
}
