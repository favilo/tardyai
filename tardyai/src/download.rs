use std::{
    fs::File,
    io::{self, Read, Seek, Write},
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use tar::Archive;

use crate::error::Error;

const S3_BASE: &str = "https://s3.amazonaws.com/fast-ai-";
const S3_IMAGE: &str = "imageclas/";

#[derive(Debug, Clone, Copy)]
pub enum DatasetUrl {
    Pets,
}

impl DatasetUrl {
    pub fn url(self) -> String {
        match self {
            Self::Pets => {
                format!("{S3_BASE}{S3_IMAGE}oxford-iiit-pet.tgz")
            }
        }
    }
}

const HF_BASE: &'static str = "https://huggingface.co/";

#[derive(Debug, Clone, Copy)]
pub(crate) enum ModelUrl {
    Resnet18,
    Resnet34,
}

impl ModelUrl {
    pub(crate) fn url(self) -> String {
        match self {
            ModelUrl::Resnet18 => {
                format!("{HF_BASE}microsoft/resnet-18/resolve/main/model.safetensors?download=true")
            }
            ModelUrl::Resnet34 => {
                format!("{HF_BASE}microsoft/resnet-34/resolve/main/model.safetensors?download=true")
            }
        }
    }
}

fn ensure_dir(path: &PathBuf) -> Result<(), Error> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

fn get_home_dir() -> Result<PathBuf, Error> {
    let home = homedir::get_my_home()?
        .expect("home directory needs to exist")
        .join(".tardyai");
    Ok(home)
}

pub fn untar_images(url: DatasetUrl) -> Result<PathBuf, Error> {
    let home = get_home_dir()?;
    let dest_dir = home.join("archive");
    ensure_dir(&dest_dir)?;
    let archive_file = download_file(url.url(), &dest_dir, None)?;

    let dest_dir = home.join("data");
    let dir = extract_archive(&archive_file, &dest_dir)?;

    Ok(dir)
}

pub(crate) fn download_model(url: ModelUrl) -> Result<PathBuf, Error> {
    let home = get_home_dir()?;
    let dest_dir = home.join("models");
    ensure_dir(&dest_dir)?;
    let model_file = download_file(url.url(), &dest_dir, Some(&format!("{url:?}.safetensors")))?;
    Ok(model_file)
}

fn download_file(
    url: String,
    dest_dir: &Path,
    default_name: Option<&str>,
) -> Result<PathBuf, Error> {
    let mut response = reqwest::blocking::get(&url)?;
    let file_name = default_name
        .or(response.url().path_segments().and_then(|s| s.last()))
        .and_then(|name| if name.is_empty() { None } else { Some(name) })
        .ok_or(Error::DownloadNameNotSpecified(url.clone()))?;

    let downloaded_file = dest_dir.join(file_name);

    // TODO: check if the archive is valid and exists
    if downloaded_file.exists() {
        log::info!("File already exists: {}", downloaded_file.display());
        return Ok(downloaded_file);
    }

    log::info!("Downloading {} to: {}", &url, downloaded_file.display());
    let mut dest = File::create(&downloaded_file)?;
    let pb = indicatif::ProgressBar::new(response.content_length().unwrap_or(0));
    let mut buf = [0; 262144]; // 256KiB buffer
    while response.read(&mut buf)? > 0 {
        dest.write_all(&buf)?;
        pb.inc(buf.len() as u64);
    }
    Ok(downloaded_file)
}

fn extract_archive(archive_file: &Path, dest_dir: &Path) -> Result<PathBuf, Error> {
    let tar_gz = File::open(archive_file)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);

    log::info!(
        "Extracting archive {} to: {}",
        archive_file.display(),
        dest_dir.display()
    );
    let dir = {
        let entry = &archive
            .entries()?
            .next()
            .ok_or(Error::TarEntry("No entries in archive"))??;
        entry.path()?.into_owned()
    };
    let archive_dir = dest_dir.join(dir);
    if archive_dir.exists() {
        log::info!("Archive already extracted to: {}", archive_dir.display());
        return Ok(archive_dir);
    }

    let tar = archive.into_inner();
    let mut tar_gz = tar.into_inner();
    tar_gz.seek(io::SeekFrom::Start(0))?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);
    archive.unpack(dest_dir)?;

    Ok(archive_dir)
}
