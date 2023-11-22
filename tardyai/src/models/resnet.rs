use std::fs::File;

use ::safetensors::SafeTensors;
use dfdx::prelude::*;
use memmap2::MmapOptions;

use crate::{
    download::{download_model, ModelUrl},
    Error,
};

type BasicBlock<const C: usize> = Residual<(
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
    ReLU,
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
)>;

type Downsample<const C: usize, const D: usize> = GeneralizedResidual<
    (
        Conv2D<C, D, 3, 2, 1>,
        BatchNorm2D<D>,
        ReLU,
        Conv2D<D, D, 3, 1, 1>,
        BatchNorm2D<D>,
    ),
    (Conv2D<C, D, 1, 2, 0>, BatchNorm2D<D>),
>;

type Head = (
    Conv2D<3, 64, 7, 2, 3>,
    BatchNorm2D<64>,
    ReLU,
    MaxPool2D<3, 2, 1>,
);

pub type Tail<const NUM_CLASSES: usize> = (AvgPoolGlobal, Linear<512, NUM_CLASSES>);

// Layer clusters are in groups of [2, 2, 2, 2]
pub type Resnet18<const NUM_CLASSES: usize> = (
    Head,
    (BasicBlock<64>, ReLU, BasicBlock<64>, ReLU),
    (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
    (Downsample<128, 256>, ReLU, BasicBlock<256>, ReLU),
    (Downsample<256, 512>, ReLU, BasicBlock<512>, ReLU),
    Tail<NUM_CLASSES>,
);

pub struct Resnet18Model<const NUM_CLASSES: usize, E: Dtype>
where
    Resnet18<NUM_CLASSES>: BuildOnDevice<AutoDevice, E>,
    AutoDevice: Device<E>,
{
    model: <Resnet18<NUM_CLASSES> as BuildOnDevice<AutoDevice, E>>::Built,
}

impl<E, const N: usize> Resnet18Model<N, E>
where
    E: Dtype + dfdx::tensor::safetensors::SafeDtype,
    AutoDevice: Device<E>,
    Resnet18<N>: BuildOnDevice<AutoDevice, E>,
{
    pub fn build(dev: AutoDevice) -> Self {
        let model = dev.build_module::<Resnet18<N>, E>();
        Self { model }
    }

    pub fn download_model(&mut self) -> Result<(), Error> {
        let model_file = download_model(ModelUrl::Resnet18)?;
        self.model.load_safetensors(&model_file)?;
        Ok(())
    }
}

// Layer clusters are in groups of [3, 4, 6, 4]
pub type Resnet34<const NUM_CLASSES: usize> = (
    Head,
    (
        BasicBlock<64>,
        ReLU,
        BasicBlock<64>,
        ReLU,
        BasicBlock<64>,
        ReLU,
    ),
    (
        // tuples are only supported with up to 6 items in `dfdx`
        (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
        (BasicBlock<128>, ReLU, BasicBlock<128>, ReLU),
    ),
    (
        // tuples are only supported with up to 6 items in `dfdx`
        (
            Downsample<128, 256>,
            ReLU,
            BasicBlock<256>,
            ReLU,
            BasicBlock<256>,
            ReLU,
        ),
        (
            BasicBlock<256>,
            ReLU,
            BasicBlock<256>,
            ReLU,
            BasicBlock<256>,
            ReLU,
        ),
    ),
    (
        Downsample<256, 512>,
        ReLU,
        BasicBlock<512>,
        ReLU,
        BasicBlock<512>,
        ReLU,
    ),
    Tail<NUM_CLASSES>,
);

pub struct Resnet34Model<const NUM_CLASSES: usize, E>
where
    E: Dtype + dfdx::tensor::safetensors::SafeDtype,
    Resnet34<NUM_CLASSES>: BuildOnDevice<AutoDevice, E>,
    AutoDevice: Device<E>,
{
    model: <Resnet34<NUM_CLASSES> as BuildOnDevice<AutoDevice, E>>::Built,
}

impl<E, const N: usize> Resnet34Model<N, E>
where
    E: Dtype + dfdx::tensor::safetensors::SafeDtype,
    AutoDevice: Device<E>,
    Resnet34<N>: BuildOnDevice<AutoDevice, E>,
{
    pub fn build(dev: AutoDevice) -> Self {
        let model = dev.build_module::<Resnet34<N>, E>();
        Self { model }
    }

    pub fn download_model(&mut self) -> Result<(), Error> {
        log::info!("Downloading model from {}", ModelUrl::Resnet34.url());
        let model_file = download_model(ModelUrl::Resnet34)?;
        // self.model.load_safetensors(&model_file)?;

        let file = File::open(model_file).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();
        let mut names = tensors.tensors();
        names.sort_by_key(|t| t.0.clone());
        for (name, tensor) in names {
            log::info!("Name: {name}: {:?}", tensor.shape());
        }
        Ok(())
    }
}
