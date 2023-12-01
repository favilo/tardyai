use std::{fs::File, path::Path};

use ::safetensors::SafeTensors;
use dfdx::prelude::*;
use dfdx_core::nn_traits::LoadSafeTensors as _;
// use memmap2::MmapOptions;
use rand::distributions::uniform::SampleUniform;

use crate::{
    download::{download_model, ModelUrl},
    // models::loader::NamedTensorVisitor,
    // error::Error,
};

mod layers;

// use self::layers::RESNET34_LAYERS;

#[derive(Debug, Default, Clone, Sequential)]
pub struct BasicBlockInternal<const C: usize> {
    conv1: Conv2DConstConfig<C, C, 3, 1, 1>,
    bn1: BatchNorm2DConstConfig<C>,
    relu: ReLU,
    conv2: Conv2DConstConfig<C, C, 3, 1, 1>,
    bn2: BatchNorm2DConstConfig<C>,
}

#[derive(Debug, Default, Clone, Sequential)]
pub struct DownsampleA<const C: usize, const D: usize> {
    conv1: Conv2DConstConfig<C, D, 3, 2, 1>,
    bn1: BatchNorm2DConstConfig<D>,
    relu: ReLU,
    conv2: Conv2DConstConfig<D, D, 3, 1, 1>,
    bn2: BatchNorm2DConstConfig<D>,
}

#[derive(Debug, Default, Clone, Sequential)]
pub struct DownsampleB<const C: usize, const D: usize> {
    conv1: Conv2DConstConfig<C, D, 1, 2, 0>,
    bn1: BatchNorm2DConstConfig<D>,
}

pub type BasicBlock<const C: usize> = ResidualAdd<BasicBlockInternal<C>>;

pub type Downsample<const C: usize, const D: usize> =
    GeneralizedAdd<DownsampleA<C, D>, DownsampleB<C, D>>;

#[derive(Debug, Default, Clone, Sequential)]
pub struct Stem {
    conv: Conv2DConstConfig<3, 64, 7, 2, 3>,
    bn: BatchNorm2DConstConfig<64>,
    relu: ReLU,
    pool: MaxPool2DConst<3, 2, 1>,
}

#[derive(Debug, Default, Clone, Sequential)]
#[built(Resnet18Body)]
pub struct Resnet18BodyConfig {
    stem: Stem,
    l1: (BasicBlock<64>, ReLU, BasicBlock<64>, ReLU),
    l2: (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
    l3: (Downsample<128, 256>, ReLU, BasicBlock<256>, ReLU),
    l4: (Downsample<256, 512>, ReLU, BasicBlock<512>, ReLU),
}

#[derive(Debug, Default, Clone, Sequential)]
#[built(Head)]
pub struct HeadConfig<const NUM_CLASSES: usize> {
    pool: AvgPoolGlobal,
    l: LinearConstConfig<512, NUM_CLASSES>,
}

#[derive(Debug, Default, Clone, CustomModule)]
#[built(Resnet18)]
pub struct Resnet18Config<const N: usize> {
    #[module]
    body: Resnet18BodyConfig,
    #[module]
    head: HeadConfig<N>,
}

impl<S: Shape, T: Tape<E, D>, const N: usize, E: Dtype, D: Device<E>> Module<Tensor<S, E, D, T>>
    for Resnet18<N, E, D>
where
    Resnet18Body<E, D>: Module<Tensor<S, E, D, NoneTape>>,
    <Resnet18Body<E, D> as Module<Tensor<S, E, D, NoneTape>>>::Output: PutTape<T>,
    Head<N, E, D>: Module<
        <<Resnet18Body<E, D> as Module<Tensor<S, E, D, NoneTape>>>::Output as PutTape<T>>::Output,
    >,
{
    type Output =
        <Head<N, E, D> as Module<
            <<Resnet18Body<E, D> as Module<Tensor<S, E, D>>>::Output as dfdx::prelude::PutTape<
                T,
            >>::Output,
        >>::Output;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        let (input, tape) = input.split_tape();
        let x = self.body.try_forward(input)?;
        let x = self.head.try_forward(x.put_tape(tape))?;
        Ok(x)
    }
}

impl<E, const N: usize, D: Device<E>> Resnet18<N, E, D>
where
    E: Dtype + num_traits::float::Float + SampleUniform,
    AutoDevice: Device<E>,
    Resnet18<N, E, AutoDevice>: BuildOnDevice<E, AutoDevice>,
    Resnet18Body<E, AutoDevice>: LoadSafeTensors,
{
    pub fn download_model(&mut self) -> Result<(), crate::error::Error> {
        let model_file = download_model(ModelUrl::Resnet18)?;
        // TODO: This isn't implemented in dfdx-derives yet.
        // self.body.load_safetensors(&model_file)?;
        Ok(())
    }
}

// Layer clusters are in groups of [3, 4, 6, 4]
// pub type Resnet34Body = (
//     Stem,
//     (
//         BasicBlock<64>,
//         ReLU,
//         BasicBlock<64>,
//         ReLU,
//         BasicBlock<64>,
//         ReLU,
//     ),
//     (
//         // tuples are only supported with up to 6 items in `dfdx`
//         (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
//         (BasicBlock<128>, ReLU, BasicBlock<128>, ReLU),
//     ),
//     (
//         // tuples are only supported with up to 6 items in `dfdx`
//         (
//             Downsample<128, 256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//         ),
//         (
//             BasicBlock<256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//         ),
//     ),
//     (
//         Downsample<256, 512>,
//         ReLU,
//         BasicBlock<512>,
//         ReLU,
//         BasicBlock<512>,
//         ReLU,
//     ),
// );

// #[derive(Default, Clone, Sequential, LoadSafeTensors)]
// #[built(Resnet34)]
// pub struct Resnet34Config<const NUM_CLASSES: usize> {
//     #[serialize("embedder")]
//     stem: Stem,
//     l1: (
//         BasicBlock<64>,
//         ReLU,
//         BasicBlock<64>,
//         ReLU,
//         BasicBlock<64>,
//         ReLU,
//     ),
//     l2: (
//         (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
//         (BasicBlock<128>, ReLU, BasicBlock<128>, ReLU),
//     ),
//     l3: (
//         (
//             Downsample<128, 256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//         ),
//         (
//             BasicBlock<256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//             BasicBlock<256>,
//             ReLU,
//         ),
//     ),
//     l4: (
//         Downsample<256, 512>,
//         ReLU,
//         BasicBlock<512>,
//         ReLU,
//         BasicBlock<512>,
//         ReLU,
//     ),
//     l5: Head<NUM_CLASSES>,
// }

// // pub type Resnet34<const NUM_CLASSES: usize> = (Resnet34Body, Head<NUM_CLASSES>);
// // type Resnet34Built<const NUM_CLASSES: usize, E> =
// //     <Resnet34<NUM_CLASSES> as BuildOnDevice<AutoDevice, E>>::Built;

// pub struct Resnet34Model<const NUM_CLASSES: usize, E>
// where
//     E: Dtype,
//     Resnet34<NUM_CLASSES, E, AutoDevice>: BuildOnDevice<E, AutoDevice>,
//     AutoDevice: Device<E>,
// {
//     pub model: Resnet34<NUM_CLASSES, E, AutoDevice>,
// }

// impl<E, const N: usize> Resnet34Model<N, E>
// where
//     E: Dtype + rand::distributions::uniform::SampleUniform + num_traits::float::Float,
//     AutoDevice: Device<E>,
//     Resnet34<N, E, AutoDevice>: BuildOnDevice<E, AutoDevice>,
// {
//     pub fn build(dev: AutoDevice) -> Self {
//         let config = Resnet34Config::<N>::default();
//         let model = dev.build_module(config);
//         Self { model }
//     }

//     pub fn download_model(&mut self) -> Result<(), crate::error::Error> {
//         log::info!("Downloading model from {}", ModelUrl::Resnet34.url());
//         let model_file = download_model(ModelUrl::Resnet34)?;

//         // TODO: Somehow make something like this work
//         self.model.load_safetensors(&model_file)?;

//         // let file = File::open(model_file).unwrap();
//         // let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
//         // let tensors = SafeTensors::deserialize(&buffer).unwrap();

//         // let _ = <Resnet34Built<N, E> as TensorCollection<E, AutoDevice>>::iter_tensors(
//         //     &mut RecursiveWalker {
//         //         m: &mut self.model,
//         //         f: &mut NamedTensorVisitor::new(&RESNET34_LAYERS, &tensors),
//         //     },
//         // )?;

//         Ok(())
//     }

//     pub fn load_model(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
//         self.model.load_safetensors(path)?;
//         Ok(())
//     }
// }
