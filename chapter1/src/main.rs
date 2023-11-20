use std::path::{Path, PathBuf};

use color_eyre::eyre::{Context, Result};
use dfdx::tensor::AutoDevice;
use tardyai::{datasets::DirectoryImageDataset, untar_images, Url};

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    color_eyre::install()?;

    let path: PathBuf = untar_images(Url::Pets)
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

    Ok(())
}
