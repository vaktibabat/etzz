use crate::parsing::Dataset;
use ndarray::Axis;

pub mod decision_tree;

pub trait Model {
    fn fit(&mut self, dataset: &Dataset);
    fn predict(&self, instance: &[f64]) -> usize;
}

fn num_mistakes<M: Model>(model: &mut M, dataset: &Dataset) -> usize {
    let mut mistakes = 0;

    for (data_row, target_row) in dataset.data.axis_iter(Axis(0)).zip(dataset.target.axis_iter(Axis(0))) {
        let prediction = model.predict(data_row.to_slice().unwrap());
        let class = target_row.iter().position(|x| *x == 1f64).unwrap(); // The actual class.
                                                                                              // The classes are in one-hot encoding
                                                                                              // So this is just finding the 1
        
        if prediction != class {
            mistakes += 1;
        }
    }

    mistakes
}

// k is the number of test sets we want to have.
// For example if the dataset is of size 150 and k=10, we'll have 10 sets of size 15,
// And test the model on a different set every time
pub fn k_fold_cross_validation<M: Model>(model: &mut M, k: usize, dataset: &Dataset) -> f64 {
    let mut scores: Vec<f64> = vec![];
    let test_set_size = dataset.data.nrows() / k;

    for i in 0..k {
        let indices: Vec<usize> = ((i * test_set_size)..((i + 1) * test_set_size)).collect();
        let validation_data = dataset.data.select(Axis(0), &indices);
        let validation_target = dataset.target.select(Axis(0), &indices);
        let validation = Dataset {data: validation_data, target: validation_target};
        let indices: Vec<usize> = (0..dataset.data.nrows()).filter(|x| !indices.contains(x)).collect();
        let train_data = dataset.data.select(Axis(0), &indices);
        let train_target = dataset.target.select(Axis(0), &indices);
        let train = Dataset {data: train_data, target: train_target};

        model.fit(&train);
        let score = (validation.data.nrows() as f64 - num_mistakes(model, &validation) as f64) / validation.data.nrows() as f64;

        scores.push(score);
    }

    scores.iter().sum::<f64>() / scores.len() as f64
}