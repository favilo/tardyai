#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Context, Result};
use dfdx::{data::ExactSizeDataset, prelude::*};
use tardyai::{
    datasets::DirectoryImageDataset, models::resnet::Resnet34Model, untar_images, DatasetUrl, learners::visual::VisualLearner,
};

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    color_eyre::install()?;

    let path: PathBuf = untar_images(DatasetUrl::Pets)
        .context("downloading Pets")?
        .join("images");
    log::info!("Images are in: {}", path.display());

    let dev = AutoDevice::default();

    // Silly thing about the Pets dataset, all the cats have a capital first letter in their
    // filename, all the dogs are lowercase only
    let is_cat = |path: &Path| {
        path.file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.chars().next().map(|c| c.is_uppercase()))
            .unwrap_or(false)
    };

    let dataset = DirectoryImageDataset::new(path, dev.clone(), is_cat)?;
    log::info!("Found {} files", dataset.files().len());

    log::info!("Building the ResNet-34 model");
    let model = Resnet34Model::<2, f32>::build(dev.clone());
    log::info!("Done building model");

    let mut learner = VisualLearner::builder(dev.clone())
        .dataset(dataset)
        .model(model)
        .build();

    log::info!("Training");
    learner.train(10)?;
    log::info!("Done training");

    // model.download_model()?;

    // let (image, is_cat) = dataset.get(1)?;
    // let categories = model.model.forward(image);

    // log::info!("Is Cat? {}", is_cat);
    // log::info!("Categories: {:?}", categories.shape().concrete());
    // log::trace!("Categories: {:#?}", categories.array());

    // let max_category = categories
    //     .softmax()
    //     .array()
    //     .into_iter()
    //     .map(ordered_float::OrderedFloat)
    //     .enumerate()
    //     .max_by_key(|t| t.1)
    //     .unwrap();
    // log::info!("(Category, Weight): {:?}", max_category);

    Ok(())
}
