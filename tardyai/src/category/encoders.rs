use std::path::Path;

use dfdx::prelude::*;

pub type LabelFn<const N: usize, Category> = dyn Fn(&Path) -> Category;

pub trait IntoOneHot<const N: usize>: Default {
    fn into_one_hot(&self, dev: &AutoDevice) -> Tensor<Rank1<N>, f32, AutoDevice>;
}

impl IntoOneHot<2> for bool {
    fn into_one_hot(&self, dev: &AutoDevice) -> Tensor<Rank1<2>, f32, AutoDevice> {
        let mut t = dev.zeros::<(Const<2>,)>();
        t[[0]] = !*self as usize as f32;
        t[[1]] = *self as usize as f32;
        t
    }
}
