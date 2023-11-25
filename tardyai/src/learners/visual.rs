use std::marker::PhantomData;

use dfdx::{
    data::{
        ExactSizeDataset, IteratorBatchExt, IteratorCollateExt, IteratorStackExt, OneHotEncode,
    },
    losses::cross_entropy_with_logits_loss,
    nn::{ModuleMut, ZeroGrads},
    optim::{Adam, Optimizer},
    shapes::Const,
    tensor::{AutoDevice, TensorFrom, Trace},
    tensor_ops::{AdamConfig, Backward},
};

use crate::{
    datasets::DirectoryImageDataset,
    error::Error,
    models::resnet::{Resnet34Built, Resnet34Model},
};

// Hard code sizes and types of datasets for now. We will generalize later.
pub struct VisualLearner<'a> {
    device: AutoDevice,
    dataset: DirectoryImageDataset<'a>,
    model: Resnet34Model<2, f32>,
    optimizer: Adam<Resnet34Built<2, f32>, f32, AutoDevice>,
}

impl<'a> VisualLearner<'a> {
    pub fn builder(device: AutoDevice) -> builder::Builder<'a, builder::WithoutDataset> {
        builder::Builder {
            device,
            dataset: None,
            model: None,
            _phantom: PhantomData,
        }
    }

    fn new(
        device: AutoDevice,
        dataset: DirectoryImageDataset<'a>,
        model: Resnet34Model<2, f32>,
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
                .map(|(image, is_cat)| {
                    let mut one_hotted = [0.0; 2];
                    one_hotted[is_cat as usize] = 1.0;
                    (image, self.device.tensor(one_hotted))
                })
                .batch_exact(Const::<16>)
                .collate()
                .stack()
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
    use crate::{datasets::DirectoryImageDataset, models::resnet::Resnet34Model};

    pub struct WithoutDataset;
    pub struct WithoutModel;
    pub struct Ready;

    pub struct Builder<'a, T> {
        pub(super) device: AutoDevice,
        pub(super) dataset: Option<DirectoryImageDataset<'a>>,
        pub(super) model: Option<Resnet34Model<2, f32>>,
        pub(super) _phantom: PhantomData<T>,
    }

    impl<'a> Builder<'a, WithoutDataset> {
        pub fn dataset(self, dataset: DirectoryImageDataset<'a>) -> Builder<'a, WithoutModel> {
            Builder {
                device: self.device,
                dataset: Some(dataset),
                model: None,
                _phantom: PhantomData,
            }
        }
    }

    impl<'a> Builder<'a, WithoutModel> {
        pub fn model(self, model: Resnet34Model<2, f32>) -> Builder<'a, Ready> {
            Builder {
                device: self.device,
                dataset: self.dataset,
                model: Some(model),
                _phantom: PhantomData,
            }
        }
    }

    impl<'a> Builder<'a, Ready> {
        pub fn build(self) -> VisualLearner<'a> {
            let model = self.model.unwrap();
            VisualLearner::new(self.device, self.dataset.unwrap(), model)
        }
    }
}
