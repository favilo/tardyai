#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::path::{Path, PathBuf};

use clap::Parser;
use color_eyre::eyre::{Context, Result};
use dfdx::prelude::*;
use tardyai::{category::splitters::RatioSplitter, prelude::*};

#[derive(Debug, Parser)]
#[command(author = "Favil Orbedios")]
struct Args {
    /// The seed to create the [AutoDevice] with, default 0
    #[clap(long, short = 's', default_value = "0")]
    seed: u64,

    /// If set, load the model from this file
    #[clap(long = "model", short = 'm')]
    model_file: Option<PathBuf>,

    /// The epoch to start training at, default 0
    #[clap(long = "epoch", short = 'e', default_value = "0")]
    start_epoch: usize,

    /// The number of epochs to train for, default 3
    #[clap(long, short = 'n', default_value = "3")]
    epochs: usize,
}

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    color_eyre::install()?;

    let args = Args::parse();

    let path: PathBuf = untar_images(DatasetUrl::Pets)
        .context("downloading Pets")?
        .join("images");
    log::info!("Images are in: {}", path.display());

    let dev = AutoDevice::seed_from_u64(args.seed);

    // Silly thing about the Pets dataset, all the cats have a capital first letter in their
    // filename, all the dogs are lowercase only
    let is_cat = |path: &Path| {
        path.file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.chars().next().map(|c| c.is_uppercase()))
            .unwrap_or(false)
    };

    let dataset_loader = DirectoryImageDataLoader::builder(path, dev.clone())
        .with_label_fn(&is_cat)
        .with_splitter(RatioSplitter::with_seed_validation(0, 0.1))
        .build()?;
    let dataset = dataset_loader.training();
    log::info!("Found {} files", dataset.files().len());

    log::info!("Building the ResNet-34 model");
    let mut model = Resnet34Model::<2, f32>::build(dev.clone());
    log::info!("Done building model");

    if let Some(model_file) = args.model_file {
        log::info!("Loading old model");
        model.load_model(model_file)?;
        log::info!("Done loading old model");
    }

    let mut learner = VisualLearner::builder(dev.clone())
        .save_each_block()
        .start_epoch(args.start_epoch)
        .with_valid_dataset(dataset_loader.validation())
        .with_train_dataset(dataset)
        .with_model(model)
        .build();

    let valid_loss = learner.valid_loss()?;
    log::info!("Valid loss: {:.5}", valid_loss);

    log::info!("Training");
    learner.train(args.epochs)?;
    log::info!("Done training");

    learner.save("model.safetensors")?;

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
