use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::fs::{read_dir, DirEntry, File};
use std::io;
use std::io::{BufReader, Error, ErrorKind};
use xml::reader::EventReader;
use xml::reader::XmlEvent as ReaderEvent;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Document {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DocumentContent {
    pub content: String,
}

fn read_xml_document(document_path: &str) -> io::Result<String> {
    let document: File = File::open(document_path)?;
    let document_reader: BufReader<File> = BufReader::new(document);

    let parser: EventReader<BufReader<File>> = EventReader::new(document_reader);

    let mut content = String::new();

    parser.into_iter().for_each(|event| {
        if let Ok(ReaderEvent::Characters(s)) = event {
            content.push_str(&s);
            content.push_str(" ");
        }
    });
    Ok(content)
}

fn dir_entry_to_string(dir_entry: DirEntry) -> Result<String, Error> {
    match dir_entry.path().into_os_string().into_string() {
        Ok(document_path) => Ok(document_path),
        Err(document_path) => Err(Error::new(
            ErrorKind::InvalidInput,
            format!("Invalid UTF-8: {:?}", document_path),
        )),
    }
}

#[allow(dead_code)]
fn result_dir_entry_to_string(result_dir_entry: Result<DirEntry, Error>) -> Option<String> {
    match result_dir_entry {
        Ok(dir_entry) => match dir_entry_to_string(dir_entry) {
            Ok(document_path) => Some(document_path),
            Err(e) => {
                println!("Error: dir_entry_to_string: {e}");
                None
            }
        },
        Err(e) => {
            println!("Error: result_dir_entry: {e}");
            None
        }
    }
}

fn document_path_to_content(
    document_paths: &[String],
) -> HashMap<Document, Option<DocumentContent>> {
    let xml_document_contents: HashMap<Document, Option<DocumentContent>> = document_paths
        .iter()
        .map(|document_path| {
            let document_content = read_xml_document(document_path)
                .ok()
                .map(|content| DocumentContent { content });
            (
                Document {
                    path: document_path.to_string(),
                },
                document_content,
            )
        })
        .collect();

    xml_document_contents
}

#[allow(dead_code)]
fn get_document_paths_in_directory(dir_path: &str) -> Vec<String> {
    let document_paths: Vec<String> = match read_dir(dir_path) {
        Ok(dir_entries) => dir_entries.filter_map(result_dir_entry_to_string).collect(),
        Err(e) => {
            println!("Error: reading {dir_path}: {e}");
            Vec::new()
        }
    };

    document_paths
}

#[allow(dead_code)]
fn print_document_contents(document_contents: HashMap<Document, Option<DocumentContent>>) {
    document_contents
        .iter()
        .for_each(|(document_path, document_content)| {
            println!(
                "{} => {}",
                document_path.path,
                match document_content {
                    Some(content) => content.content.len(),
                    None => 0,
                }
            );
        });
}

#[derive(Debug)]
struct Lexer<'a> {
    content: &'a [char],
}

impl<'a> Lexer<'a> {
    fn new(content: &'a [char]) -> Self {
        Self { content }
    }

    fn trim_left(&mut self) {
        while !self.content.is_empty() && self.content[0].is_whitespace() {
            self.content = &self.content[1..];
        }
    }

    fn chop(&mut self, n: usize) -> &'a [char] {
        let token = &self.content[0..n];
        self.content = &self.content[n..];
        token
    }

    fn extract_token<T: Fn(char) -> bool>(
        &mut self,
        content: &'a [char],
        predicate: T,
    ) -> Option<&'a [char]> {
        if predicate(content[0]) {
            let mut n = 0;
            while n < content.len() && predicate(content[n]) {
                n += 1;
            }
            return Some(self.chop(n));
        } else {
            None
        }
    }

    fn next_token(&mut self) -> Option<&'a [char]> {
        self.trim_left();
        if self.content.is_empty() {
            return None;
        }

        if let Some(numeric_token) = self.extract_token(self.content, |c| c.is_numeric()) {
            return Some(numeric_token);
        }

        if let Some(alphanumeric_token) =
            self.extract_token(self.content, |c| c.is_alphanumeric() || c == '_')
        {
            return Some(alphanumeric_token);
        }

        return Some(self.chop(1));
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = &'a [char];

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

fn terms_in_content(document_content: Vec<char>) -> Vec<String> {
    let lexer: Lexer = Lexer::new(&document_content);
    lexer.map(|token: &[char]| token.iter().collect()).collect()
}

fn content_as_string(content_struct: &DocumentContent) -> Vec<char> {
    content_struct.content.to_string().chars().collect()
}

fn content_from_document(
    document_path: &Document,
    document_to_content: &HashMap<Document, Option<DocumentContent>>,
) -> Vec<char> {
    match &document_to_content.get(document_path) {
        Some(Some(content_struct)) => content_as_string(content_struct),
        _ => vec![],
    }
}

fn document_to_term_count(
    document_to_count_per_term: &HashMap<Document, HashMap<String, usize>>,
) -> HashMap<Document, usize> {
    let mut totals: HashMap<Document, usize> = HashMap::new();
    for (document_name, term_to_count) in document_to_count_per_term {
        let term_count: usize = term_to_count.values().sum();
        totals.insert(document_name.clone(), term_count);
    }
    totals
}

fn document_or_term_to_absolute_frequency(
    document_path_to_content: &HashMap<Document, Option<DocumentContent>>,
) -> (
    HashMap<Document, HashMap<String, usize>>,
    HashMap<String, HashMap<Document, usize>>,
) {
    let mut _document_to_absolute_term_frequency_map: HashMap<Document, HashMap<String, usize>> =
        HashMap::new();
    let mut _term_to_document_frequency_map: HashMap<String, HashMap<Document, usize>> =
        HashMap::new();

    for document_path in document_path_to_content.keys() {
        let content_form_document: Vec<char> =
            content_from_document(document_path, document_path_to_content);
        let terms_in_content: Vec<String> = terms_in_content(content_form_document);
        // term_frequency is a HashMap that stores term frequencies for the current document.
        // the values are counts of how many times each term appears in the current document.
        let mut term_counting_map: HashMap<String, usize> = HashMap::new();
        let term_to_frequency_entry: &mut HashMap<String, usize> =
            _document_to_absolute_term_frequency_map
                .entry(document_path.clone())
                .or_default();
        for term in terms_in_content {
            let document_to_frequency_entry: &mut HashMap<Document, usize> =
                _term_to_document_frequency_map
                    .entry(term.to_uppercase())
                    .or_default();

            let count: &mut usize = term_counting_map.entry(term.clone()).or_insert(0);
            *count += 1;

            term_to_frequency_entry.insert(term.to_uppercase(), *count);
            document_to_frequency_entry.insert(document_path.clone(), *count);
        }
    }
    (
        _document_to_absolute_term_frequency_map,
        _term_to_document_frequency_map,
    )
}

// Term frequency, tf(t,d), is the relative frequency of term t within document d,
// i.e., the number of times that term t appears in document d divided by the total number of terms in document d
fn document_to_term_frequency(
    document_to_term_count_map: &HashMap<Document, HashMap<String, usize>>,
    document_to_total_term_count_map: &HashMap<Document, usize>,
) -> HashMap<Document, HashMap<String, f64>> {
    let mut document_to_term_frequency_map: HashMap<Document, HashMap<String, f64>> =
        HashMap::new();

    for (file, term_to_term_count_map) in document_to_term_count_map {
        if !term_to_term_count_map.is_empty() {
            // the total number of terms in document
            let total_term_count_in_document: f64 =
                *document_to_total_term_count_map.get(file).unwrap() as f64;

            // numerator = the number of times that term t occurs in document d
            // denominator = the total number of terms in document d
            for (term, term_count_in_document) in term_to_term_count_map {
                let relative_term_frequency: f64 =
                    (*term_count_in_document as f64) / total_term_count_in_document;

                document_to_term_frequency_map
                    .entry(file.clone())
                    .or_default()
                    .insert(term.clone(), relative_term_frequency);
            }
        }
    }

    document_to_term_frequency_map
}

fn document_to_sorted_terms(
    document_to_relative_term_frequency: &HashMap<Document, HashMap<String, f64>>,
) -> HashMap<Document, Vec<&String>> {
    let mut document_to_sorted_terms: HashMap<Document, Vec<&String>> = HashMap::new();

    document_to_relative_term_frequency
        .keys()
        .for_each(|document_name| {
            let term_to_relative_term_frequency: &HashMap<String, f64> =
                document_to_relative_term_frequency
                    .get(document_name)
                    .unwrap();
            let mut terms: Vec<&String> = term_to_relative_term_frequency.keys().collect();

            terms.sort_by(|&term_2, &term_1| {
                term_to_relative_term_frequency
                    .get(term_1)
                    .unwrap()
                    .partial_cmp(term_to_relative_term_frequency.get(term_2).unwrap())
                    .unwrap()
            });

            document_to_sorted_terms.insert(document_name.clone(), terms);
        });

    document_to_sorted_terms
}

fn count_unique_documents(
    term_to_document_frequency_map: &HashMap<String, HashMap<Document, usize>>,
) -> usize {
    let mut unique_paths = HashSet::new();

    for (_, inner_map) in term_to_document_frequency_map.iter() {
        for (document_path, _) in inner_map.iter() {
            unique_paths.insert(document_path);
        }
    }

    unique_paths.len()
}

fn inverse_document_frequency(
    term_to_document_frequency_map: &HashMap<String, HashMap<Document, usize>>,
    term: String,
) -> f64 {
    let term: String = term.to_uppercase();

    let number_of_documents = count_unique_documents(term_to_document_frequency_map);
    println!("Number of documents: {}", number_of_documents);

    let mut number_of_documents_with_term: usize = 0;

    if let Some(document_frequency_map) = term_to_document_frequency_map.get(&term) {
        number_of_documents_with_term = document_frequency_map
            .iter()
            .filter(|(_, &count)| count > 0)
            .count();
    }
    println!(
        "Number of documents with term: {} {}",
        term, number_of_documents_with_term
    );

    let idf: f64 =
        (number_of_documents as f64 / (1.0 + number_of_documents_with_term as f64)).log10();

    println!("IDF inverse_document_frequency: {}", idf);
    idf
}

fn term_frequency_inverse_document_frequency(
    term: String,
    document: Document,
    document_to_abolut_term_frequency_map: &HashMap<Document, HashMap<String, usize>>,
    document_to_absolut_term_count_map: &HashMap<Document, usize>,
    term_to_document_frequency_map: HashMap<String, HashMap<Document, usize>>,
) -> f64 {
    let tf_map: HashMap<Document, HashMap<String, f64>> = document_to_term_frequency(
        document_to_abolut_term_frequency_map,
        document_to_absolut_term_count_map,
    );
    let idf: f64 = inverse_document_frequency(&term_to_document_frequency_map, term.clone());

    println!("TF map: {:?}", tf_map);
    println!("IDF: {}", idf);

    let tf: &f64 = tf_map.get(&document).unwrap().get(&term).unwrap();

    println!("TF: {}", tf);
    println!("TF * IDF: {}", tf * idf);

    tf * idf
}

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

    // Count occurrences of each term in each document
    let (document_to_absolute_term_frequency_map, term_to_document_frequency_map): (
        HashMap<Document, HashMap<String, usize>>,
        HashMap<String, HashMap<Document, usize>>,
    ) = document_or_term_to_absolute_frequency(&document_to_content_map);

    // Calculate total number of terms for each document
    let document_to_term_count_map: HashMap<Document, usize> =
        document_to_term_count(&document_to_absolute_term_frequency_map);

    // Calculate relative term frequency for each term in each document
    let document_to_relative_term_frequency_map: HashMap<Document, HashMap<String, f64>> =
        document_to_term_frequency(
            &document_to_absolute_term_frequency_map,
            &document_to_term_count_map,
        );

    // Sort terms based on their relative frequency
    let document_to_sorted_terms_map: HashMap<Document, Vec<&String>> =
        document_to_sorted_terms(&document_to_relative_term_frequency_map);

    // Display results
    for (document_name, terms) in &document_to_sorted_terms_map {
        for term in terms {
            println!(
                "File {}: Term '{}' relative term frequency {}",
                document_name.path,
                term,
                document_to_relative_term_frequency_map
                    .get(document_name)
                    .unwrap()
                    .get(*term)
                    .unwrap()
            );
        }
    }

    term_frequency_inverse_document_frequency(
        "is".to_string(),
        Document {
            path: String::from("docs.gl/gl2"),
        },
        &document_to_absolute_term_frequency_map,
        &document_to_term_count_map,
        term_to_document_frequency_map,
    );
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_document_to_occurrences_per_term_map() {
        let submission_document: Document = Document {
            path: String::from("submission.txt"),
        };

        // given:
        let mut mock_document_contents: HashMap<Document, Option<DocumentContent>> = HashMap::new();
        mock_document_contents.insert(
            submission_document.clone(),
            Some(DocumentContent {
                content: String::from("test document content"),
            }),
        );

        // when:
        let (result, _): (
            HashMap<Document, HashMap<String, usize>>,
            HashMap<String, HashMap<Document, usize>>,
        ) = document_or_term_to_absolute_frequency(&mock_document_contents);

        // then:
        let mut expected_result: HashMap<Document, HashMap<String, usize>> = HashMap::new();
        let mut term_count: HashMap<String, usize> = HashMap::new();
        term_count.insert(String::from("TEST"), 1);
        term_count.insert(String::from("DOCUMENT"), 1);
        term_count.insert(String::from("CONTENT"), 1);
        expected_result.insert(submission_document, term_count);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_document_to_total_term_count_map() {
        let submission_document: Document = Document {
            path: String::from("submission.txt"),
        };

        // given:
        let mut mock_term_counts: HashMap<String, usize> = HashMap::new();
        mock_term_counts.insert(String::from("test"), 2);
        mock_term_counts.insert(String::from("file"), 1);
        mock_term_counts.insert(String::from("content"), 3);

        let mut mock_document_to_term_count: HashMap<Document, HashMap<String, usize>> =
            HashMap::new();
        mock_document_to_term_count.insert(submission_document.clone(), mock_term_counts);

        // when:
        let result: HashMap<Document, usize> = document_to_term_count(&mock_document_to_term_count);

        // then:
        let mut expected_result: HashMap<Document, usize> = HashMap::new();
        expected_result.insert(submission_document, 6);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_document_to_relative_term_frequency_map() {
        let document_path: Document = Document {
            path: String::from("file.txt"),
        };

        // given:
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        term_counts.insert(String::from("term1"), 1);
        term_counts.insert(String::from("term2"), 2);
        term_counts.insert(String::from("term3"), 3);

        let mut document_to_occurrences: HashMap<Document, HashMap<String, usize>> = HashMap::new();
        document_to_occurrences.insert(document_path.clone(), term_counts);

        let mut total_term_counts: HashMap<Document, usize> = HashMap::new();
        total_term_counts.insert(document_path.clone(), 6);

        // when:
        let result: HashMap<Document, HashMap<String, f64>> =
            document_to_term_frequency(&document_to_occurrences, &total_term_counts);

        // then:
        let mut expected_term_frequencies: HashMap<String, f64> = HashMap::new();
        expected_term_frequencies.insert(String::from("term1"), 1.0 / 6.0);
        expected_term_frequencies.insert(String::from("term2"), 2.0 / 6.0);
        expected_term_frequencies.insert(String::from("term3"), 3.0 / 6.0);

        let mut expected_result: HashMap<Document, HashMap<String, f64>> = HashMap::new();
        expected_result.insert(document_path, expected_term_frequencies);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_document_to_sorted_terms_map() {
        let document_path: Document = Document {
            path: String::from("file.txt"),
        };

        // given:
        let mut term_frequencies: HashMap<String, f64> = HashMap::new();
        let term1 = String::from("term1");
        let term2 = String::from("term2");
        let term3 = String::from("term3");
        term_frequencies.insert(term1.clone(), 0.3);
        term_frequencies.insert(term2.clone(), 0.2);
        term_frequencies.insert(term3.clone(), 0.1);

        let mut document_to_term_frequency: HashMap<Document, HashMap<String, f64>> =
            HashMap::new();
        document_to_term_frequency.insert(document_path.clone(), term_frequencies);

        // when:
        let result: HashMap<Document, Vec<&String>> =
            document_to_sorted_terms(&document_to_term_frequency);

        // then:
        let expected_sorted_terms = vec![&term1, &term2, &term3];
        let mut expected_result: HashMap<Document, Vec<&String>> = HashMap::new();
        expected_result.insert(document_path, expected_sorted_terms);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_count_unique_documents() {
        let document_1 = Document {
            path: String::from("file.txt"),
        };

        let document_2 = Document {
            path: String::from("file2.txt"),
        };

        let mut document_to_term_frequency_map_1: HashMap<Document, usize> = HashMap::new();
        document_to_term_frequency_map_1.insert(document_1.clone(), 10);
        document_to_term_frequency_map_1.insert(document_2.clone(), 100);

        let mut document_to_term_frequency_map_2: HashMap<Document, usize> = HashMap::new();
        document_to_term_frequency_map_2.insert(document_1.clone(), 11);
        document_to_term_frequency_map_2.insert(document_2.clone(), 110);

        let mut term_to_document_frequency_map: HashMap<String, HashMap<Document, usize>> =
            HashMap::new();

        term_to_document_frequency_map
            .insert(String::from("term_1"), document_to_term_frequency_map_1);
        term_to_document_frequency_map
            .insert(String::from("term_2"), document_to_term_frequency_map_2);

        let count: usize = count_unique_documents(&term_to_document_frequency_map);

        assert_eq!(count, 2);
    }
    #[test]
    fn test_inverse_document_frequency() {
        let document_1: Document = Document {
            path: String::from("document_1.txt"),
        };
        let document_2: Document = Document {
            path: String::from("document_2.txt"),
        };

        // given:
        let mut term_to_document_frequency_map: HashMap<String, HashMap<Document, usize>> =
            HashMap::new();

        let mut term1_document_to_frequency: HashMap<Document, usize> = HashMap::new();
        let mut term2_document_to_frequency: HashMap<Document, usize> = HashMap::new();
        let mut term3_document_to_frequency: HashMap<Document, usize> = HashMap::new();
        // Number of documents with term_1: 2
        term1_document_to_frequency.insert(document_1.clone(), 1);
        term1_document_to_frequency.insert(document_2.clone(), 1);

        // Number of documents with term_2: 1
        term2_document_to_frequency.insert(document_1.clone(), 2);

        // Number of documents with term_3: 1
        term3_document_to_frequency.insert(document_1.clone(), 3);

        let term1 = String::from("TERM1");
        let term2 = String::from("TERM2");
        let term3 = String::from("TERM3");
        term_to_document_frequency_map.insert(term1.clone(), term1_document_to_frequency);
        term_to_document_frequency_map.insert(term2.clone(), term2_document_to_frequency);
        term_to_document_frequency_map.insert(term3.clone(), term3_document_to_frequency);

        // when:
        //     (number_of_documents as f64) / (1.0 + number_of_documents_with_term as f64).log10()
        //     [2 / (1.0 + 2)].log10() = [2 / 3.0)].log10()
        let idf_term_1: f64 = inverse_document_frequency(&term_to_document_frequency_map, term1);

        //     (number_of_documents as f64) / (1.0 + number_of_documents_with_term as f64).log10()
        //     [2 / (1.0 + 1)].log10() = [2 / 2.0)].log10()
        let idf_term_2: f64 = inverse_document_frequency(&term_to_document_frequency_map, term2);

        //     (number_of_documents as f64) / (1.0 + number_of_documents_with_term as f64).log10()
        //     [2 / (1.0 + 1)].log10() = [2 / 2.0)].log10()
        let idf_term_3: f64 = inverse_document_frequency(&term_to_document_frequency_map, term3);

        // then:
        assert_eq!(idf_term_1, (2.0 / 3f64).log10());
        assert_eq!(idf_term_2, (2.0 / 2f64).log10());
        assert_eq!(idf_term_3, (2.0 / 2f64).log10());
    }

    #[test]
    fn test_term_frequency_inverse_document_frequency() {
        let document_1: Document = Document {
            path: String::from("document_1.txt"),
        };

        // given:
        let term_1 = String::from("TERM_1");

        let mut term_frequency_map: HashMap<String, usize> = HashMap::new();
        let mut document_to_term_count_map: HashMap<Document, usize> = HashMap::new();

        let mut document_to_abolute_term_count_map: HashMap<Document, usize> = HashMap::new();
        let mut document_to_term_frequency_map: HashMap<Document, HashMap<String, usize>> =
            HashMap::new();
        let mut term_to_document_frequency_map: HashMap<String, HashMap<Document, usize>> =
            HashMap::new();

        term_frequency_map.insert(term_1.clone(), 1);
        document_to_abolute_term_count_map.insert(document_1.clone(), 1);
        document_to_term_frequency_map.insert(document_1.clone(), term_frequency_map);

        document_to_term_count_map.insert(document_1.clone(), 1);
        term_to_document_frequency_map.insert(term_1.clone(), document_to_term_count_map);

        // when:
        let result: f64 = term_frequency_inverse_document_frequency(
            term_1.clone(),
            document_1.clone(),
            &document_to_term_frequency_map,
            &document_to_abolute_term_count_map,
            term_to_document_frequency_map,
        );

        // (number_of_documents as f64 / (1.0 + number_of_documents_with_term as f64)).log10()
        let idf: f64 = (1.0 / (1.0 + 1f64)).log10();

        //     // numerator = the number of times that term t occurs in document d
        //     // denominator = the total number of terms in document d
        //     for (term, term_count_in_document) in term_to_term_count_map {
        //         let relative_term_frequency: f64 =
        //             (*term_count_in_document as f64) / total_term_count_in_document;
        //           tf =>   1                          /     1
        let tf: f64 = 1.0;
        assert_eq!(result, tf * idf);
    }
}
