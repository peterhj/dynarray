use crate::*;

#[cfg(feature = "mklml")] use mklml_ffi::*;

#[inline]
fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

pub trait VectorInPlaceOps<T> where T: Copy {
  fn matrix_vector_mult(&mut self, a: DynArray<T>, x: DynArray<T>);
  fn transpose_matrix_vector_mult(&mut self, a: DynArray<T>, x: DynArray<T>);
}

#[cfg(feature = "mklml")]
impl VectorInPlaceOps<f32> for DynArray<f32> {
  fn matrix_vector_mult(&mut self, a: DynArray<f32>, x: DynArray<f32>) {
    assert_eq!(a.size()[0], self.size()[0]);
    assert_eq!(a.size()[1], x.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.borrow();
    let x = x.borrow();
    let mut y = self.borrow_mut();
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe { cblas_sgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_mut_ptr(), sz2int(y.stride()[0]),
    ) };
  }

  fn transpose_matrix_vector_mult(&mut self, a: DynArray<f32>, x: DynArray<f32>) {
    assert_eq!(a.size()[0], x.size()[0]);
    assert_eq!(a.size()[1], self.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.borrow();
    let x = x.borrow();
    let mut y = self.borrow_mut();
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe { cblas_sgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_mut_ptr(), sz2int(y.stride()[0]),
    ) };
  }
}

#[cfg(feature = "mklml")]
impl VectorInPlaceOps<f64> for DynArray<f64> {
  fn matrix_vector_mult(&mut self, a: DynArray<f64>, x: DynArray<f64>) {
    assert_eq!(a.size()[0], self.size()[0]);
    assert_eq!(a.size()[1], x.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.borrow();
    let x = x.borrow();
    let mut y = self.borrow_mut();
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    unsafe { cblas_dgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasNoTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_mut_ptr(), sz2int(y.stride()[0]),
    ) };
  }

  fn transpose_matrix_vector_mult(&mut self, a: DynArray<f64>, x: DynArray<f64>) {
    assert_eq!(a.size()[0], x.size()[0]);
    assert_eq!(a.size()[1], self.size()[0]);
    assert_eq!(a.stride()[0], 1);
    let a = a.borrow();
    let x = x.borrow();
    let mut y = self.borrow_mut();
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;
    unsafe { cblas_dgemv(
        CBLAS_LAYOUT_CblasColMajor,
        CBLAS_TRANSPOSE_CblasTrans,
        sz2int(a.size()[0]),
        sz2int(a.size()[1]),
        alpha,
        a.as_ptr(), sz2int(a.stride()[1]),
        x.as_ptr(), sz2int(x.stride()[0]),
        beta,
        y.as_mut_ptr(), sz2int(y.stride()[0]),
    ) };
  }
}

pub trait MatrixInPlaceOps<T> where T: Copy {
  fn matrix_mult(&mut self, a: DynArray<T>, b: DynArray<T>);
  fn left_transpose_matrix_mult(&mut self, a: DynArray<T>, b: DynArray<T>);
  fn right_transpose_matrix_mult(&mut self, a: DynArray<T>, b: DynArray<T>);
}
