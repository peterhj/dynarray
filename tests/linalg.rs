extern crate arrayidx;
extern crate rearray;

use arrayidx::*;
use rearray::*;
use rearray::linalg::*;

#[test]
fn test_sgemv() {
  let a: Rearray<f32> = Rearray::zeros(IndexNd::from(vec![17, 8]));
  let x: Rearray<f32> = Rearray::zeros(IndexNd::from(vec![8]));
  let mut y: Rearray<f32> = Rearray::zeros(IndexNd::from(vec![17]));
  y.matrix_vector_mult(1.0, a.clone(), x.clone(), 0.0);
}

#[test]
fn test_sgemm() {
  let a: Rearray<f32> = Rearray::zeros(IndexNd::from(vec![17, 8]));
  let b: Rearray<f32> = Rearray::zeros(IndexNd::from(vec![8, 15]));
  let mut y: Rearray<f32> = Rearray::zeros(IndexNd::from(vec![17, 15]));
  y.matrix_mult(1.0, a.clone(), b.clone(), 0.0);
}
