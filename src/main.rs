pub mod model;
pub mod parsing;

use model::{Model, decision_tree, k_fold_cross_validation};
use parsing::iris;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path of the training dataset
    #[arg(short, long)]
    train_path: String,
    
    /// The maximum depth of the tree
    #[arg(short, long)]
    max_depth: usize,
}

fn main() {
    let args = Args::parse();
    
    let dataset = iris::parse_dataset(
        &args.train_path,
    );
    let mut dt = decision_tree::DecisionTree::new(args.max_depth);
    dt.fit(&dataset);
    
    println!("The Resulting Decision Tree is {}", dt);
    println!("10-Fold cross validation score is {}", k_fold_cross_validation(&mut dt, 10, &dataset));
}

#[test]
fn prediction_test() {
    let dataset = iris::parse_dataset(
        "/home/sag0li/Projects/Rust_DecisionTree/decisiontree/datasets/iris.csv",
    );
    let mut dt = decision_tree::DecisionTree::new(2);
    dt.fit(&dataset);
    let prediction = dt.predict(&[1.3f64, 2.4f64, 4.5f64, 2.3f64]);
    assert_eq!(prediction, 2);
    let prediction = dt.predict(&[1.7, 1.8, 2.3, 4.5]);
    assert_eq!(prediction, 0);
}

#[test]
fn k_fold_test() {
    let dataset = iris::parse_dataset(
        "/home/sag0li/Projects/Rust_DecisionTree/decisiontree/datasets/iris.csv",
    );
    let mut dt = decision_tree::DecisionTree::new(2);
    let k_fold = k_fold_cross_validation(&mut dt, 10, &dataset);

    println!("K-fold cross validation score: {}", k_fold);
}