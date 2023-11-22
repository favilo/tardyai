use ::safetensors::SafeTensors;
use dfdx::{prelude::*, tensor::safetensors::SafeDtype};
use num_traits::NumCast;

use crate::Error;

pub(crate) struct NamedTensorVisitor<'a> {
    names: Vec<&'static str>,
    idx: usize,
    tensors: &'a SafeTensors<'a>,
}

impl<'a> NamedTensorVisitor<'a> {
    pub(crate) fn new(names: Vec<&'static str>, tensors: &'a SafeTensors<'a>) -> Self {
        Self {
            names,
            idx: 0,
            tensors,
        }
    }
}

impl<'a, E: Dtype + SafeDtype, D: Device<E>> TensorVisitor<E, D> for NamedTensorVisitor<'a> {
    type Viewer = ViewTensorMut;

    type Err = Error;

    type E2 = E;

    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        _opts: TensorOptions<S, E, D>,
        t: <Self::Viewer as TensorViewer>::View<'_, Tensor<S, E, D>>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err> {
        log::debug!(
            "Loading tensor shape: {:?}, {:?}",
            t.shape(),
            &self.names.get(self.idx)
        );
        t.load_safetensor(
            &mut self.tensors,
            &self.names.get(self.idx).ok_or(Error::NotEnoughNames)?,
        )?;
        self.idx += 1;
        Ok(None)
    }

    fn visit_scalar<N: NumCast>(
        &mut self,
        _opts: ScalarOptions<N>,
        n: <Self::Viewer as TensorViewer>::View<'_, N>,
    ) -> Result<Option<N>, Self::Err> {
        log::debug!("Loading scalar: {:?}", &self.names.get(self.idx));
        let tensor = self
            .tensors
            .tensor(self.names.get(self.idx).ok_or(Error::NotEnoughNames)?)?;
        let data = tensor.data();
        let mut array = [0; 8];
        array.copy_from_slice(data);
        let val = f64::from_le_bytes(array);
        *n = N::from(val).ok_or(Error::NumberFormatException)?;

        self.idx += 1;
        Ok(None)
    }
}
