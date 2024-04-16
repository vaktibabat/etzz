use crate::parsing::Dataset;
use ndarray::{Array2, ArrayView1, Axis};
use std::fmt::Display;

use super::Model;

const NUM_FEATURES: usize = 4;

#[derive(Debug, Clone)]
struct TreeNode {
    // Feature and threshold are optional because leaf nodes have no specific feature/threshold
    feature: Option<usize>,
    threshold: Option<f64>,
    majority_class: usize,
    // Sepal Width <= 2.45
    left: Option<Box<TreeNode>>, // Left child of this node, i.e. feature <= threshold
    right: Option<Box<TreeNode>>, // Right child of this node, i.e. feature > threshold
    depth: usize,
}

#[derive(Clone, Debug)]
pub struct DecisionTree {
    root: Option<Box<TreeNode>>,
    max_depth: usize, // Regularization hyperparameter
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> DecisionTree {
        DecisionTree {
            root: None,
            max_depth,
        }
    }

    // Return the feature and threshold that minimize the cost function J
    fn find_best_split(&self, dataset: &Dataset) -> (usize, f64) {
        let data = &dataset.data;
        let (mut best_feature, mut best_threshold, mut best_loss) =
            (0, f64::INFINITY, f64::INFINITY);

        for feature in 0..NUM_FEATURES {
            let mut col = data.column(feature).clone().to_vec();
            col.sort_by(|a, b| a.total_cmp(b));
            col.dedup();
            // Midpoint between every two sequential datapoints
            let thresholds = (0..col.len() - 1).map(|i| (col[i] + col[i + 1]) / 2f64);

            for threshold in thresholds {
                let curr_loss = split_loss(dataset, feature, threshold);

                if curr_loss < best_loss {
                    best_feature = feature;
                    best_threshold = threshold;
                    best_loss = curr_loss;
                }
            }
        }

        (best_feature, best_threshold)
    }

    fn make_decision_tree(&mut self, dataset: &Dataset, depth: usize) -> Option<TreeNode> {
        let target = &dataset.target;
        let classes = target.sum_axis(Axis(0)).to_vec();
        let majority = classes
            .iter()
            .enumerate()
            .max_by(|(_, b), (_, d)| b.total_cmp(d))
            .unwrap()
            .0;

        if depth > self.max_depth {
            return None;
        }

        // Base case: If the dataset is pure (no need to split further), create a leaf with the majority class
        // We also stop splitting if this node has depth larger than the maximum allowed depth
        if is_pure(&classes) || depth == self.max_depth {
            Some(TreeNode {
                feature: None,
                threshold: None,
                majority_class: majority,
                left: None,
                right: None,
                depth,
            })
        } else {
            let (feature, threshold) = self.find_best_split(dataset);
            let left_dataset = left_dataset(dataset, feature, threshold);
            let right_dataset = right_dataset(dataset, feature, threshold);
            let left_child = self
                .make_decision_tree(&left_dataset, depth + 1)
                .map(Box::new);
            let right_child = self
                .make_decision_tree(&right_dataset, depth + 1)
                .map(Box::new);

            Some(TreeNode {
                feature: Some(feature),
                threshold: Some(threshold),
                left: left_child,
                right: right_child,
                majority_class: majority,
                depth,
            })
        }
    }
}

impl Model for DecisionTree {
    fn predict(&self, features: &[f64]) -> usize {
        let mut curr_node = self.root.as_ref().unwrap();

        loop {
            if curr_node.left.is_none() && curr_node.right.is_none() {
                break;
            } else {
                let feature = curr_node.feature.unwrap();
                let threshold = curr_node.threshold.unwrap();
                curr_node = if features[feature] <= threshold {
                    &curr_node.left.as_ref().unwrap()
                } else {
                    &curr_node.right.as_ref().unwrap()
                };
            }
        }

        curr_node.majority_class
    }

    fn fit(&mut self, dataset: &Dataset) {
        self.root = Some(Box::new(self.make_decision_tree(dataset, 0).unwrap()));
    }
}

impl Display for TreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut prefix = "-".repeat(self.depth);
        
        // This is not a leaf node
        if let (Some(feature), Some(threshold)) = (self.feature, self.threshold) {
            let node_data = format!("{} <= {}", feature, threshold);
            prefix.push_str(&node_data);

            write!(f, "{}\n", prefix)?;
            write!(f, "{}", self.left.as_ref().unwrap())?;
            write!(f, "{}", self.right.as_ref().unwrap())?;
        }
        else {
            let node_data = format!("LEAF majority {}", self.majority_class);
            prefix.push_str(&node_data);

            write!(f, "{}\n", prefix)?;
        }

        Ok(())
    }
}

impl Display for DecisionTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\nMax Depth is {}\n", self.max_depth)?;
        write!(f, "{}", self.root.as_ref().unwrap())?;

        Ok(())
    }
}

fn is_pure(class_numbers: &[f64]) -> bool {
    gini_impurity(class_numbers) == 0f64
}

fn left_dataset(dataset: &Dataset, feature: usize, threshold: f64) -> Dataset {
    let data = &dataset.data;
    let target = &dataset.target;
    let mut data_rows: Vec<ArrayView1<f64>> = Vec::new();
    let mut target_rows: Vec<ArrayView1<f64>> = Vec::new();

    for row in data.axis_iter(Axis(0)).enumerate() {
        if row.1[feature] <= threshold {
            data_rows.push(row.1);
            target_rows.push(target.index_axis(Axis(0), row.0));
        }
    }

    let data = Array2::from_shape_vec(
        (data_rows.len(), data.ncols()),
        data_rows.iter().flatten().copied().collect(),
    )
    .unwrap();
    let target = Array2::from_shape_vec(
        (target_rows.len(), target.ncols()),
        target_rows.iter().flatten().copied().collect(),
    )
    .unwrap();

    Dataset { data, target }
}

fn right_dataset(dataset: &Dataset, feature: usize, threshold: f64) -> Dataset {
    let data = &dataset.data;
    let target = &dataset.target;
    let mut data_rows: Vec<ArrayView1<f64>> = Vec::new();
    let mut target_rows: Vec<ArrayView1<f64>> = Vec::new();

    for row in data.axis_iter(Axis(0)).enumerate() {
        if row.1[feature] > threshold {
            data_rows.push(row.1);
            target_rows.push(target.index_axis(Axis(0), row.0));
        }
    }

    let data = Array2::from_shape_vec(
        (data_rows.len(), data.ncols()),
        data_rows.iter().flatten().copied().collect(),
    )
    .unwrap();
    let target = Array2::from_shape_vec(
        (target_rows.len(), target.ncols()),
        target_rows.iter().flatten().copied().collect(),
    )
    .unwrap();

    Dataset { data, target }
}

fn gini_impurity(class_numbers: &[f64]) -> f64 {
    let total: f64 = class_numbers.iter().sum();
    let probs: f64 = class_numbers
        .iter()
        .map(|x| (x / total) * (x / total))
        .sum();

    1f64 - probs
}

// What is the cost of splitting the dataset based on (feature, threshold)?
fn split_loss(dataset: &Dataset, feature: usize, threshold: f64) -> f64 {
    let data = &dataset.data;
    // Get indices of row in the left node
    let left_indices: Vec<usize> = data
        .axis_iter(Axis(0))
        .enumerate()
        .filter(|(_, row)| row[feature] <= threshold)
        .map(|(i, _)| i)
        .collect();
    // Get indices of row in the right node
    let right_indices: Vec<usize> = data
        .axis_iter(Axis(0))
        .enumerate()
        .filter(|(_, row)| row[feature] > threshold)
        .map(|(i, _)| i)
        .collect();
    // How many instances in each node are in each class?
    let left_classes = dataset
        .target
        .select(Axis(0), &left_indices)
        .sum_axis(Axis(0))
        .to_vec();
    let right_classes = dataset
        .target
        .select(Axis(0), &right_indices)
        .sum_axis(Axis(0))
        .to_vec();
    // Compute the weights of the left/right nodes
    let num_left = left_indices.len() as f64;
    let num_right = right_indices.len() as f64;
    let num_total = num_left + num_right;

    (num_left / num_total) * gini_impurity(&left_classes)
        + (num_right / num_total) * gini_impurity(&right_classes)
}
