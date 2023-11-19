use std::path::PathBuf;

use color_eyre::eyre::{Context, Result};
use tardyai::{untar_images, Url};

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    color_eyre::install()?;

    let path: PathBuf = untar_images(Url::Pets)
        .context("downloading Pets")?
        .join("images");
    log::info!("Images are in: {}", path.display());
    Ok(())
}
