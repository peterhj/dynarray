use crate::*;

#[cfg(feature = "mklml")]
use mklml_sys::cblas::*;

#[inline]
fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

pub trait VectorInPlaceOps<T: Copy + 'static> {
  fn matrix_vector_mult(&mut self, alpha: T, a: Rearray<T>, x: Rearray<T>, beta: T);
  fn transpose_matrix_vector_mult(&mut self, alpha: T, a: Rearray<T>, x: Rearray<T>, beta: T);
}

#[cfg(feature = "mklml")]
impl VectorInPlaceOps<f32> for Rearray<f32> {
  fn matrix_vector_mult(&mut self, alpha: f32, a: Rearray<f32>, x: Rearray<f32>, beta: f32) {
    assert_eq!(a.size()[0], self.size()[0]);
    assert_eq!(a.size()[1], x.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.dense_view();
    let x = x.dense_view();
    let y = self.dense_view_mut();
    unsafe { cblas_sgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_ptr_mut(), sz2int(y.stride()[0]),
    ) };
  }

  fn transpose_matrix_vector_mult(&mut self, alpha: f32, a: Rearray<f32>, x: Rearray<f32>, beta: f32) {
    assert_eq!(a.size()[0], x.size()[0]);
    assert_eq!(a.size()[1], self.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.dense_view();
    let x = x.dense_view();
    let y = self.dense_view_mut();
    unsafe { cblas_sgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_ptr_mut(), sz2int(y.stride()[0]),
    ) };
  }
}

#[cfg(feature = "mklml")]
impl VectorInPlaceOps<f64> for Rearray<f64> {
  fn matrix_vector_mult(&mut self, alpha: f64, a: Rearray<f64>, x: Rearray<f64>, beta: f64) {
    assert_eq!(a.size()[0], self.size()[0]);
    assert_eq!(a.size()[1], x.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.dense_view();
    let x = x.dense_view();
    let y = self.dense_view_mut();
    unsafe { cblas_dgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_ptr_mut(), sz2int(y.stride()[0]),
    ) };
  }

  fn transpose_matrix_vector_mult(&mut self, alpha: f64, a: Rearray<f64>, x: Rearray<f64>, beta: f64) {
    assert_eq!(a.size()[0], x.size()[0]);
    assert_eq!(a.size()[1], self.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.dense_view();
    let x = x.dense_view();
    let y = self.dense_view_mut();
    unsafe { cblas_dgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_ptr_mut(), sz2int(y.stride()[0]),
    ) };
  }
}

pub trait MatrixInPlaceOps<T: Copy + 'static> {
  fn matrix_mult(&mut self, alpha: T, a: Rearray<T>, b: Rearray<T>, beta: T);
  fn left_transpose_matrix_mult(&mut self, alpha: T, a: Rearray<T>, b: Rearray<T>, beta: T);
  fn right_transpose_matrix_mult(&mut self, alpha: T, a: Rearray<T>, b: Rearray<T>, beta: T);
}

#[cfg(feature = "mklml")]
impl MatrixInPlaceOps<f32> for Rearray<f32> {
  fn matrix_mult(&mut self, alpha: f32, a: Rearray<f32>, b: Rearray<f32>, beta: f32) {
    let nrows = self.size()[0];
    let ncols = self.size()[1];
    assert_eq!(a.size()[0], nrows);
    assert_eq!(a.size()[1], b.size()[0]);
    assert_eq!(b.size()[1], ncols);
    assert_eq!(a.stride()[0], 1);
    assert_eq!(b.stride()[0], 1);
    assert_eq!(self.stride()[0], 1);
    let a = a.dense_view();
    let b = b.dense_view();
    let y = self.dense_view_mut();
    unsafe { cblas_sgemm(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(nrows),
        sz2int(ncols),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        b.as_ptr(), sz2int(b.stride()[1]),
        beta,
        y.as_ptr_mut(), sz2int(y.stride()[1]),
    ) };
  }

  fn left_transpose_matrix_mult(&mut self, alpha: f32, a: Rearray<f32>, b: Rearray<f32>, beta: f32) {
    unimplemented!();
  }

  fn right_transpose_matrix_mult(&mut self, alpha: f32, a: Rearray<f32>, b: Rearray<f32>, beta: f32) {
    unimplemented!();
  }
}

#[cfg(feature = "mklml")]
impl MatrixInPlaceOps<f64> for Rearray<f64> {
  fn matrix_mult(&mut self, alpha: f64, a: Rearray<f64>, b: Rearray<f64>, beta: f64) {
    let nrows = self.size()[0];
    let ncols = self.size()[1];
    assert_eq!(a.size()[0], nrows);
    assert_eq!(a.size()[1], b.size()[0]);
    assert_eq!(b.size()[1], ncols);
    assert_eq!(a.stride()[0], 1);
    assert_eq!(b.stride()[0], 1);
    assert_eq!(self.stride()[0], 1);
    let a = a.dense_view();
    let b = b.dense_view();
    let y = self.dense_view_mut();
    unsafe { cblas_dgemm(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(nrows),
        sz2int(ncols),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        b.as_ptr(), sz2int(b.stride()[1]),
        beta,
        y.as_ptr_mut(), sz2int(y.stride()[1]),
    ) };
  }

  fn left_transpose_matrix_mult(&mut self, alpha: f64, a: Rearray<f64>, b: Rearray<f64>, beta: f64) {
    unimplemented!();
  }

  fn right_transpose_matrix_mult(&mut self, alpha: f64, a: Rearray<f64>, b: Rearray<f64>, beta: f64) {
    unimplemented!();
  }
}
