use std::fs::File;

use ::safetensors::SafeTensors;
use dfdx::{prelude::*, tensor::safetensors::SafeDtype};
use memmap2::MmapOptions;
use once_cell::sync::Lazy;

use crate::{
    download::{download_model, ModelUrl},
    models::loader::NamedTensorVisitor,
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

type Stem = (
    Conv2D<3, 64, 7, 2, 3>,
    BatchNorm2D<64>,
    ReLU,
    MaxPool2D<3, 2, 1>,
);

pub type Head<const NUM_CLASSES: usize> = (AvgPoolGlobal, Linear<512, NUM_CLASSES>);

// Layer clusters are in groups of [2, 2, 2, 2]
pub type Resnet18Body = (
    Stem,
    (BasicBlock<64>, ReLU, BasicBlock<64>, ReLU),
    (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
    (Downsample<128, 256>, ReLU, BasicBlock<256>, ReLU),
    (Downsample<256, 512>, ReLU, BasicBlock<512>, ReLU),
);

pub type Resnet18<const NUM_CLASSES: usize> = (Resnet18Body, Head<NUM_CLASSES>);

pub struct Resnet18Model<const NUM_CLASSES: usize, E: Dtype>
where
    Resnet18<NUM_CLASSES>: BuildOnDevice<AutoDevice, E>,
    AutoDevice: Device<E>,
{
    model: <Resnet18<NUM_CLASSES> as BuildOnDevice<AutoDevice, E>>::Built,
}

impl<E, const N: usize> Resnet18Model<N, E>
where
    E: Dtype + SafeDtype,
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
pub type Resnet34Body = (
    Stem,
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
);

pub type Resnet34<const NUM_CLASSES: usize> = (Resnet34Body, Head<NUM_CLASSES>);

pub struct Resnet34Model<const NUM_CLASSES: usize, E>
where
    E: Dtype + SafeDtype,
    Resnet34<NUM_CLASSES>: BuildOnDevice<AutoDevice, E>,
    AutoDevice: Device<E>,
{
    model: <Resnet34<NUM_CLASSES> as BuildOnDevice<AutoDevice, E>>::Built,
}

static RESNET34_LAYERS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "resnet.embedder.embedder.convolution.weight",
        "resnet.embedder.embedder.normalization.weight",
        "resnet.embedder.embedder.normalization.bias",
        "resnet.embedder.embedder.normalization.running_mean",
        "resnet.embedder.embedder.normalization.running_var",
        "resnet.embedder.embedder.normalization.num_batches_tracked",
        "resnet.embedder.embedder.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.0.layer.0.convolution.weight",
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.weight",
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.bias",
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.running_mean",
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.running_var",
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.0.layer.1.convolution.weight",
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.weight",
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.bias",
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.running_mean",
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.running_var",
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.1.layer.0.convolution.weight",
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.weight",
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.bias",
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.running_mean",
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.running_var",
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.1.layer.1.convolution.weight",
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.weight",
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.bias",
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.running_mean",
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.running_var",
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.2.layer.0.convolution.weight",
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.weight",
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.bias",
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.running_mean",
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.running_var",
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.2.layer.1.convolution.weight",
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.weight",
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.bias",
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.running_mean",
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.running_var",
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.0.layer.0.convolution.weight",
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.weight",
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.bias",
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.running_mean",
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.running_var",
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.0.layer.1.convolution.weight",
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.weight",
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.bias",
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.running_mean",
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.running_var",
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.0.shortcut.convolution.weight",
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.weight",
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.bias",
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.running_mean",
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.running_var",
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.1.layer.0.convolution.weight",
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.weight",
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.bias",
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.running_mean",
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.running_var",
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.1.layer.1.convolution.weight",
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.weight",
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.bias",
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.running_mean",
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.running_var",
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.2.layer.0.convolution.weight",
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.weight",
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.bias",
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.running_mean",
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.running_var",
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.2.layer.1.convolution.weight",
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.weight",
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.bias",
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.running_mean",
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.running_var",
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.3.layer.0.convolution.weight",
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.weight",
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.bias",
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.running_mean",
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.running_var",
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.3.layer.1.convolution.weight",
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.weight",
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.bias",
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.running_mean",
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.running_var",
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.0.layer.0.convolution.weight",
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.weight",
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.bias",
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.running_mean",
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.running_var",
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.0.layer.1.convolution.weight",
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.weight",
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.bias",
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.running_mean",
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.running_var",
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.0.shortcut.convolution.weight",
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.weight",
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.bias",
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.running_mean",
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.running_var",
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.1.layer.0.convolution.weight",
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.weight",
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.bias",
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.running_mean",
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.running_var",
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.1.layer.1.convolution.weight",
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.weight",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.bias",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_mean",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_var",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.1.layer.1.convolution.weight",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.weight",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.bias",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_mean",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_var",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.2.layer.0.convolution.weight",
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.weight",
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.bias",
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.running_mean",
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.running_var",
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.2.layer.1.convolution.weight",
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.weight",
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.bias",
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.running_mean",
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.running_var",
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.3.layer.0.convolution.weight",
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.weight",
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.bias",
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.running_mean",
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.running_var",
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.3.layer.1.convolution.weight",
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.weight",
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.bias",
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.running_mean",
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.running_var",
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.4.layer.0.convolution.weight",
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.weight",
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.bias",
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.running_mean",
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.running_var",
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.4.layer.1.convolution.weight",
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.weight",
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.bias",
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.running_mean",
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.running_var",
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.5.layer.0.convolution.weight",
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.weight",
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.bias",
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.running_mean",
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.running_var",
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.num_batches_tracked",
        // TODO: Figure out why these were skipped in the model, no bueno
        // "resnet.encoder.stages.2.layers.5.layer.1.convolution.weight",
        // "resnet.encoder.stages.2.layers.5.layer.1.normalization.weight",
        // "resnet.encoder.stages.2.layers.5.layer.1.normalization.bias",
        // "resnet.encoder.stages.2.layers.5.layer.1.normalization.running_mean",
        // "resnet.encoder.stages.2.layers.5.layer.1.normalization.running_var",
        // "resnet.encoder.stages.2.layers.5.layer.1.normalization.num_batches_tracked",
        // "resnet.encoder.stages.2.layers.5.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.0.layer.0.convolution.weight",
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.weight",
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.bias",
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.running_mean",
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.running_var",
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.0.layer.1.convolution.weight",
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.weight",
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.bias",
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.running_mean",
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.running_var",
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.0.shortcut.convolution.weight",
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.weight",
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.bias",
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.running_mean",
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.running_var",
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.1.layer.0.convolution.weight",
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.weight",
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.bias",
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.running_mean",
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.running_var",
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.1.layer.1.convolution.weight",
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.weight",
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.bias",
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.running_mean",
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.running_var",
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.2.layer.0.convolution.weight",
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.weight",
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.bias",
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.running_mean",
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.running_var",
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.2.layer.1.convolution.weight",
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.weight",
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.bias",
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.running_mean",
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.running_var",
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.num_batches_tracked",
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.num_batches_tracked",
        "classifier.1.weight",
        "classifier.1.bias",
    ]
});

impl<E, const N: usize> Resnet34Model<N, E>
where
    E: Dtype + SafeDtype,
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

        // TODO: Some how make something like this work
        // self.model.load_safetensors(&model_file)?;

        let file = File::open(model_file).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();

        let _ = <<Resnet34<N> as BuildOnDevice<AutoDevice, E>>::Built as TensorCollection<
            E,
            AutoDevice,
        >>::iter_tensors(&mut RecursiveWalker {
            m: &mut self.model,
            f: &mut NamedTensorVisitor::new(RESNET34_LAYERS.clone(), &tensors),
        })?;

        Ok(())
    }
}
