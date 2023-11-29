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

use super::encoders::{IntoOneHot, LabelFn};
use crate::error::Error;

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

pub struct DirectoryImageDataset<'fun, const N: usize, Category> {
    files: Vec<PathBuf>,
    dev: AutoDevice,
    label_fn: &'fun LabelFn<N, Category>,
    tensors: DashMap<PathBuf, Tensor<Rank3<3, 224, 224>, f32, AutoDevice>>,
}

impl<'fun, const N: usize, Category> DirectoryImageDataset<'fun, N, Category> {
    fn new(
        files: &[PathBuf],
        dev: AutoDevice,
        label_fn: &'fun LabelFn<N, Category>,
    ) -> Result<Self, Error> {
        Ok(Self {
            files: files.to_owned(),
            dev,
            label_fn,
            tensors: Default::default(),
        })
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }
}

impl<const N: usize, Category> ExactSizeDataset for DirectoryImageDataset<'_, N, Category> {
    type Item<'a> = Result<(Tensor<Rank3<3, 224, 224>, f32, AutoDevice>, Category), Error>
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

pub struct DirectoryDataLoader<'fun, const N: usize, Category> {
    training: DirectoryImageDataset<'fun, N, Category>,
    validation: DirectoryImageDataset<'fun, N, Category>,
    test: DirectoryImageDataset<'fun, N, Category>,
}

impl<'fun, const N: usize, Category: IntoOneHot<N>> DirectoryDataLoader<'fun, N, Category> {
    pub fn builder(
        parent: impl AsRef<Path>,
        dev: AutoDevice,
    ) -> data_loader::Builder<'fun, N, Category> {
        data_loader::Builder::new(parent.as_ref().to_owned(), dev)
    }

    pub fn training(&self) -> &DirectoryImageDataset<'fun, N, Category> {
        &self.training
    }

    pub fn validation(&self) -> &DirectoryImageDataset<'fun, N, Category> {
        &self.validation
    }

    pub fn test(&self) -> &DirectoryImageDataset<'fun, N, Category> {
        &self.test
    }
}

mod data_loader {
    use std::path::PathBuf;

    use dfdx::tensor::AutoDevice;

    use crate::category::splitters::{RatioSplitter, Splitter};

    use super::*;

    pub struct Builder<'fun, const N: usize, Category> {
        parent: PathBuf,
        dev: AutoDevice,
        splitter: Option<Box<dyn Splitter<PathBuf>>>,
        label_fn: Option<&'fun LabelFn<N, Category>>,
    }

    impl<'fun, const N: usize, Category: IntoOneHot<N>> Builder<'fun, N, Category> {
        pub fn new(parent: PathBuf, dev: AutoDevice) -> Self {
            Self {
                parent,
                dev,
                splitter: None,
                label_fn: None,
            }
        }

        pub fn with_splitter(mut self, splitter: impl Splitter<PathBuf> + 'static) -> Self {
            self.splitter = Some(Box::new(splitter));
            self
        }

        pub fn with_label_fn(mut self, label_fn: &'fun LabelFn<N, Category>) -> Self {
            self.label_fn = Some(label_fn);
            self
        }

        pub fn build(self) -> Result<DirectoryDataLoader<'fun, N, Category>, Error> {
            let exts = image_extensions();

            let mut splitter = self
                .splitter
                .unwrap_or_else(|| Box::new(RatioSplitter::default()));
            let label_fn = self.label_fn.unwrap_or(&|_| Default::default());

            let walker = WalkDir::new(self.parent).follow_links(true).into_iter();
            let files: Vec<_> = walker
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
            let (training, validation, test) = splitter.split(files);
            let training = DirectoryImageDataset::new(&training, self.dev.clone(), label_fn)?;
            let validation = DirectoryImageDataset::new(&validation, self.dev.clone(), label_fn)?;
            let test = DirectoryImageDataset::new(&test, self.dev, label_fn)?;

            Ok(DirectoryDataLoader {
                training,
                validation,
                test,
            })
        }
    }
}
