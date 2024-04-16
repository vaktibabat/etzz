use super::Dataset;
use ndarray::{Array2, ArrayView};
use std::fs::File;
use std::io::{self, Read};

const NUM_FEATURES: usize = 4;
const NUM_CLASSES: usize = 3;

/// Convert the class name from a string to a number.
/// This is converted to one-hot encoding later
fn class_name_to_number(class_name: &str) -> io::Result<usize> {
    match class_name {
        "Iris-setosa" => Ok(0),
        "Iris-versicolor" => Ok(1),
        "Iris-virginica" => Ok(2),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Invalid class name {:?}", class_name),
        )),
    }
}

/// Parse a line in the dataset. Return the sample's features and its class
/// The dataset is taken from here: https://www.kaggle.com/datasets/saurabh00007/iriscsv?resource=download
fn parse_dataset_line(line: &str) -> Option<(Vec<f64>, usize)> {
    let split = line.split(',');
    let split = split.collect::<Vec<&str>>();

    if split.len() == 5 {
        return None;
    }

    // Skip the first column which is the id
    let features: Vec<f64> = split[1..=NUM_FEATURES]
        .iter()
        .map(|x| x.parse::<f64>().unwrap())
        .collect();
    let class_name = class_name_to_number(split[NUM_FEATURES + 1]);

    Some((features, class_name.unwrap()))
}

// Return matrix that represents the dataset
pub fn parse_dataset(path: &str) -> Dataset {
    let file = File::open(path);
    let mut data = Array2::zeros((0, NUM_FEATURES));
    let mut target = Array2::zeros((0, NUM_CLASSES));
    let mut contents = String::new();

    file.unwrap().read_to_string(&mut contents).unwrap();

    for line in contents.lines().skip(1).take_while(|x| !x.is_empty()) {
        let (features, class_idx) = parse_dataset_line(line).unwrap();
        let one_hot_target: Vec<f64> = (0..NUM_CLASSES)
            .map(|x| if x == class_idx { 1f64 } else { 0f64 })
            .collect();

        data.push_row(ArrayView::from(&features)).unwrap();
        target.push_row(ArrayView::from(&one_hot_target)).unwrap();
    }

    Dataset { data, target }
}

#[cfg(tests)]
mod tests {
    #[test]
    fn parse_test() {
        let indices = vec![0, 69, 111, 33];
        let dataset =
            parse_dataset("/home/sag0li/Projects/Rust_DecisionTree/decisiontree/datasets/iris.csv");
        let data_rows = dataset.data.select(Axis(0), &indices);
        let target_rows = dataset.target.select(Axis(0), &indices);

        let exp_data: Vec<f64> = vec![
            5.1, 3.5, 1.4, 0.2, 5.6, 2.5, 3.9, 1.1, 6.4, 2.7, 5.3, 1.9, 5.5, 4.2, 1.4, 0.2,
        ];
        let exp_data_matrix = Array2::from_shape_vec((4, 4), exp_data).unwrap();
        let exp_target: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let exp_target_matrix = Array2::from_shape_vec((4, 3), exp_target).unwrap();

        assert_eq!(data_rows, exp_data_matrix);
        assert_eq!(target_rows, exp_target_matrix);
    }
}
