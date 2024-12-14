mod tfidf;
mod args;

use crate::tfidf::{
    get_document_paths_in_directory,
    document_path_to_content,
    document_and_term_to_count,
    document_to_count,
    document_to_term_to_tf,
    document_to_sorted_terms,
    Document,
    DocumentContent,
    TermToDocCountMap,
    DocToTermCountMap,
};

use clap::Parser;

use std::collections::HashMap;
use itertools::Itertools;

fn main() {


    let dir_paths = [
        "docs.gl/gl2",
        // "docs.gl/gl3",
        // "docs.gl/gl4",
        // "docs.gl/specs",
        // "docs.gl/el3",
        // "docs.gl/es1",
        // "docs.gl/es2",
        // "docs.gl/es3",
        // "docs.gl/sl4",
    ];

    let document_paths: Vec<String> = dir_paths
        .iter()
        .flat_map(|document_path| get_document_paths_in_directory(document_path))
        .sorted()
        .collect::<Vec<String>>();

    // Get content from each document
    let document_to_content_map: HashMap<Document, Option<DocumentContent>> =
        document_path_to_content(&document_paths);

    // Count occurrences f each term in each document
    let (document_to_term_to_count_map, _): (DocToTermCountMap, TermToDocCountMap) =
        document_and_term_to_count(&document_to_content_map);

    // Calculate total number of terms for each document
    let document_to_count_map: HashMap<Document, usize> =
        document_to_count(&document_to_term_to_count_map);

    // Calculate relative term frequency for each term in each document
    let document_to_term_to_frequency_map: HashMap<Document, HashMap<String, f64>> =
        document_to_term_to_tf(&document_to_term_to_count_map, &document_to_count_map);

    // Sort terms based on their relative frequency
    let document_to_sorted_terms_map: HashMap<Document, Vec<&String>> =
        document_to_sorted_terms(&document_to_term_to_frequency_map);

    // Display results
    for (document_name, terms) in &document_to_sorted_terms_map {
        for term in terms {
            println!(
                "Document {}: Term '{}' relative term frequency {}",
                document_name.path,
                term,
                document_to_term_to_frequency_map
                    .get(document_name)
                    .unwrap()
                    .get(*term)
                    .unwrap()
            );
        }
    }
}