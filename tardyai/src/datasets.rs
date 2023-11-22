use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use dashmap::DashMap;
use dfdx::{
    data::ExactSizeDataset,
    shapes::{Const, Rank3},
    tensor::{AutoDevice, Tensor, TensorFromVec},
};
use image::{imageops::FilterType, io::Reader as ImageReader, EncodableLayout, ImageFormat};
use walkdir::WalkDir;

use crate::Error;

fn image_extensions() -> HashSet<&'static str> {
    let mut set = HashSet::default();
    set.extend(ImageFormat::Jpeg.extensions_str());
    set.extend(ImageFormat::Png.extensions_str());
    set.extend(ImageFormat::Gif.extensions_str());
    set.extend(ImageFormat::WebP.extensions_str());
    set.extend(ImageFormat::Tiff.extensions_str());
    set.extend(ImageFormat::Bmp.extensions_str());
    set.extend(ImageFormat::Qoi.extensions_str());
    set
}

pub struct DirectoryImageDataset<'fun> {
    files: Vec<PathBuf>,
    dev: AutoDevice,
    label_fn: Box<dyn Fn(&Path) -> bool + 'fun>,
    tensors: DashMap<PathBuf, Tensor<Rank3<3, 224, 224>, f32, AutoDevice>>,
}

impl<'fun> DirectoryImageDataset<'fun> {
    pub fn new(
        parent: PathBuf,
        dev: AutoDevice,
        label_fn: impl Fn(&Path) -> bool + 'fun,
    ) -> Result<Self, Error> {
        let exts = image_extensions();

        let walker = WalkDir::new(parent).follow_links(true).into_iter();
        let files = walker
            .filter_map(|entry| {
                let entry = entry.ok()?;
                entry
                    .path()
                    .extension()
                    .and_then(|ext| Some(exts.contains(ext.to_str()?)))?
                    .then_some(entry)
            })
            .map(|entry| entry.path().to_owned())
            .collect();
        Ok(Self {
            files,
            dev,
            label_fn: Box::new(label_fn),
            tensors: Default::default(),
        })
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }
}

impl ExactSizeDataset for DirectoryImageDataset<'_> {
    type Item<'a> = Result<(Tensor<Rank3<3, 224, 224>, f32, AutoDevice>, bool), Error>
    where
        Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        let image_file = &self.files[index];
        let label = (self.label_fn)(image_file);
        if self.tensors.contains_key(image_file) {
            return Ok((self.tensors.get(image_file).unwrap().clone(), label));
        }
        // Read the image and resize it to 224x224, and 3 channels
        let image = ImageReader::open(image_file)?
            .decode()?
            .resize_exact(224, 224, FilterType::Triangle)
            .into_rgb8();

        // Shrink the byte values to f32 between [0, 1]
        let bytes: Vec<f32> = image.as_bytes().iter().map(|&b| b as f32 / 255.0).collect();

        // Create the tensor and the label
        let tensor = self
            .dev
            .tensor_from_vec(bytes, (Const::<3>, Const::<224>, Const::<224>));
        self.tensors.insert(image_file.clone(), tensor.clone());
        Ok((tensor, label))
    }

    fn len(&self) -> usize {
        self.files.len()
    }
}
