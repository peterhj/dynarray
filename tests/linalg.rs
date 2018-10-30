extern crate arrayidx;
extern crate dynarray;

use arrayidx::*;
use dynarray::*;
use dynarray::linalg::*;

#[test]
fn test_sgemv() {
  let x: DynArray<f32> = DynArray::zeros(IndexNd{components: vec![8]});
  let a: DynArray<f32> = DynArray::zeros(IndexNd{components: vec![17, 8]});
  let mut y: DynArray<f32> = DynArray::zeros(IndexNd{components: vec![17]});
  y.matrix_vector_mult(a.clone(), x.clone());
}
