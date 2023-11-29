use std::marker::PhantomData;

use dfdx::{
    data::{ExactSizeDataset, IteratorBatchExt, IteratorCollateExt, IteratorStackExt},
    losses::cross_entropy_with_logits_loss,
    nn::{ModuleMut, ZeroGrads},
    optim::{Adam, Optimizer},
    shapes::Const,
    tensor::{AutoDevice, Trace},
    tensor_ops::{AdamConfig, Backward},
};
use indicatif::ProgressIterator;

use crate::{
    category::{datasets::DirectoryImageDataset, encoders::IntoOneHot},
    error::Error,
    models::resnet::{Resnet34Built, Resnet34Model},
};

// Hard code sizes and types of datasets for now. We will generalize later.
pub struct VisualLearner<'a, const N: usize, Category> {
    device: AutoDevice,
    dataset: &'a DirectoryImageDataset<'a, N, Category>,
    model: Resnet34Model<N, f32>,
    optimizer: Adam<Resnet34Built<N, f32>, f32, AutoDevice>,
}

impl<'a, const N: usize, Category: IntoOneHot<N>> VisualLearner<'a, N, Category> {
    pub fn builder(
        device: AutoDevice,
    ) -> builder::Builder<'a, builder::WithoutDataset, N, Category> {
        builder::Builder {
            device,
            dataset: None,
            model: None,
            _phantom: PhantomData,
        }
    }

    fn new(
        device: AutoDevice,
        dataset: &'a DirectoryImageDataset<'a, N, Category>,
        model: Resnet34Model<N, f32>,
    ) -> Self {
        let adam = Adam::new(&model.model, AdamConfig::default());
        Self {
            device,
            dataset,
            model,
            optimizer: adam,
        }
    }

    pub fn train(&mut self, epochs: usize) -> Result<(), Error> {
        let mut rng = rand::thread_rng();
        let mut grads = self.model.model.alloc_grads();
        for epoch in 0..epochs {
            log::info!("Epoch {}", epoch);
            for (image, is_cat) in self
                .dataset
                .shuffled(&mut rng)
                .map(Result::unwrap)
                .map(|(image, is_cat)| (image, is_cat.into_one_hot(&self.device)))
                .batch_exact(Const::<16>)
                .collate()
                .stack()
                .progress()
            {
                let logits = self.model.model.forward_mut(image.traced(grads));
                let loss = cross_entropy_with_logits_loss(logits, is_cat);
                grads = loss.backward();
                self.optimizer.update(&mut self.model.model, &grads)?;
                self.model.model.zero_grads(&mut grads);
            }
        }
        Ok(())
    }
}

pub mod builder {
    use super::*;
    use crate::{category::datasets::DirectoryImageDataset, models::resnet::Resnet34Model};

    pub struct WithoutDataset;
    pub struct WithoutModel;
    pub struct Ready;

    pub struct Builder<'a, T, const N: usize, Category> {
        pub(super) device: AutoDevice,
        pub(super) dataset: Option<&'a DirectoryImageDataset<'a, N, Category>>,
        pub(super) model: Option<Resnet34Model<N, f32>>,
        pub(super) _phantom: PhantomData<T>,
    }

    impl<'a, const N: usize, Category> Builder<'a, WithoutDataset, N, Category> {
        pub fn dataset(
            self,
            dataset: &'a DirectoryImageDataset<'a, N, Category>,
        ) -> Builder<'a, WithoutModel, N, Category> {
            Builder {
                device: self.device,
                dataset: Some(dataset),
                model: None,
                _phantom: PhantomData,
            }
        }
    }

    impl<'a, const N: usize, Category> Builder<'a, WithoutModel, N, Category> {
        pub fn model(self, model: Resnet34Model<N, f32>) -> Builder<'a, Ready, N, Category> {
            Builder {
                device: self.device,
                dataset: self.dataset,
                model: Some(model),
                _phantom: PhantomData,
            }
        }
    }

    impl<'a, const N: usize, Category: IntoOneHot<N>> Builder<'a, Ready, N, Category> {
        pub fn build(self) -> VisualLearner<'a, N, Category> {
            let model = self.model.unwrap();
            VisualLearner::new(self.device, self.dataset.unwrap(), model)
        }
    }
}
