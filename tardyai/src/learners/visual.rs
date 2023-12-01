use std::{marker::PhantomData, path::Path, time::Instant};

use dfdx::{
    data::{ExactSizeDataset, IteratorBatchExt, IteratorCollateExt, IteratorStackExt},
    losses::cross_entropy_with_logits_loss,
    nn::{Module, ModuleMut, SaveToSafetensors, ZeroGrads},
    optim::{Adam, Optimizer},
    shapes::Const,
    tensor::{AsArray, AutoDevice, Trace},
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
    train_dataset: &'a DirectoryImageDataset<'a, N, Category>,
    valid_dataset: Option<&'a DirectoryImageDataset<'a, N, Category>>,
    model: Resnet34Model<N, f32>,
    optimizer: Adam<Resnet34Built<N, f32>, f32, AutoDevice>,
    save_each_block: bool,
    start_epoch: usize,
}

const BATCH_SIZE: usize = 16;

impl<'a, const N: usize, Category: IntoOneHot<N>> VisualLearner<'a, N, Category> {
    pub fn builder(
        device: AutoDevice,
    ) -> builder::Builder<'a, builder::WithoutDataset, N, Category> {
        builder::Builder {
            device,
            train_dataset: None,
            valid_dataset: None,
            model: None,
            save_each_block: false,
            start_epoch: 0,
            _phantom: PhantomData,
        }
    }

    fn new(
        device: AutoDevice,
        train_dataset: &'a DirectoryImageDataset<'a, N, Category>,
        valid_dataset: Option<&'a DirectoryImageDataset<'a, N, Category>>,
        model: Resnet34Model<N, f32>,
        save_each_block: bool,
        start_epoch: usize,
    ) -> Self {
        let adam = Adam::new(&model.model, AdamConfig::default());
        Self {
            device,
            train_dataset,
            valid_dataset,
            model,
            optimizer: adam,
            save_each_block,
            start_epoch,
        }
    }

    pub fn train(&mut self, epochs: usize) -> Result<(), Error> {
        let mut rng = rand::thread_rng();
        let mut grads = self.model.model.alloc_grads();
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for epoch in self.start_epoch..self.start_epoch + epochs {
            log::info!("Epoch {}", epoch);
            for (image, is_cat) in self
                .train_dataset
                .shuffled(&mut rng)
                .map(Result::unwrap)
                .map(|(image, is_cat)| (image, is_cat.into_one_hot(&self.device)))
                .batch_exact(Const::<BATCH_SIZE>)
                .collate()
                .stack()
                .progress()
            {
                let logits = self.model.model.forward_mut(image.traced(grads));
                let loss = cross_entropy_with_logits_loss(logits, is_cat);
                total_epoch_loss += loss.array();
                num_batches += 1;

                grads = loss.backward();
                self.optimizer.update(&mut self.model.model, &grads)?;
                self.model.model.zero_grads(&mut grads);

                if self.save_each_block {
                    self.save(format!("model-epoch-{}.safetensors", epoch))?;
                }
            }
            let dur = start.elapsed();

            log::info!(
                "Epoch {epoch} in {dur:?} ({:.3} batches/s): avg sample loss: {:.5}",
                num_batches as f32 / dur.as_secs_f32(),
                BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
            );

            // if self.valid_dataset.is_some() {
            //     let valid_loss = self.valid_loss()?;
            //     log::info!("Valid loss: {:.5}", valid_loss);
            // }
        }
        Ok(())
    }

    pub fn valid_loss(&mut self) -> Result<f32, Error> {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        log::info!("Calculating validation loss");
        for (img, is_cat) in self
            .valid_dataset
            .ok_or(Error::NoValidationDataset)?
            .iter()
            .map(Result::unwrap)
            .map(|(image, is_cat)| (image, is_cat.into_one_hot(&self.device)))
            .batch_exact(BATCH_SIZE)
            .collate()
            .stack()
            .progress()
        {
            let logits = self.model.model.forward(img);
            let loss = cross_entropy_with_logits_loss(logits, is_cat);
            total_epoch_loss += loss.array();
            num_batches += 1;
        }
        Ok(BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        Ok(self.model.model.save_safetensors(path)?)
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
        pub(super) train_dataset: Option<&'a DirectoryImageDataset<'a, N, Category>>,
        pub(super) valid_dataset: Option<&'a DirectoryImageDataset<'a, N, Category>>,
        pub(super) model: Option<Resnet34Model<N, f32>>,
        pub(super) save_each_block: bool,
        pub(super) start_epoch: usize,
        pub(super) _phantom: PhantomData<T>,
    }

    impl<'a, const N: usize, Category> Builder<'a, WithoutDataset, N, Category> {
        pub fn with_train_dataset(
            self,
            dataset: &'a DirectoryImageDataset<'a, N, Category>,
        ) -> Builder<'a, WithoutModel, N, Category> {
            Builder {
                device: self.device,
                train_dataset: Some(dataset),
                valid_dataset: self.valid_dataset,
                model: None,
                save_each_block: self.save_each_block,
                start_epoch: self.start_epoch,
                _phantom: PhantomData,
            }
        }
    }

    impl<'a, const N: usize, Category> Builder<'a, WithoutModel, N, Category> {
        pub fn with_model(self, model: Resnet34Model<N, f32>) -> Builder<'a, Ready, N, Category> {
            Builder {
                device: self.device,
                train_dataset: self.train_dataset,
                valid_dataset: self.valid_dataset,
                model: Some(model),
                save_each_block: self.save_each_block,
                start_epoch: self.start_epoch,
                _phantom: PhantomData,
            }
        }
    }

    impl<'a, const N: usize, Category: IntoOneHot<N>> Builder<'a, Ready, N, Category> {
        pub fn build(self) -> VisualLearner<'a, N, Category> {
            VisualLearner::new(
                self.device,
                self.train_dataset.unwrap(),
                self.valid_dataset,
                self.model.unwrap(),
                self.save_each_block,
                self.start_epoch,
            )
        }
    }

    impl<'a, const N: usize, Category: IntoOneHot<N>, T> Builder<'a, T, N, Category> {
        pub fn save_each_block(mut self) -> Self {
            self.save_each_block = true;
            self
        }

        pub fn start_epoch(mut self, start_epoch: usize) -> Self {
            self.start_epoch = start_epoch;
            self
        }

        pub fn with_valid_dataset(
            mut self,
            valid_dataset: &'a DirectoryImageDataset<'a, N, Category>,
        ) -> Self {
            self.valid_dataset = Some(valid_dataset);
            self
        }
    }
}
