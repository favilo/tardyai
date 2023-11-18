use std::path::PathBuf;

use color_eyre::eyre::{Context, Result};

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    color_eyre::install()?;
    let path: PathBuf = tardyai::untar_images(tardyai::Url::Pets)
        .context("downloading Pets")?
        .join("images");
    Ok(())
}
