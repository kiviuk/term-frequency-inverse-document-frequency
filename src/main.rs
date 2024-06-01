/*
Search Engines, one of the pioneering and most prevalent applications of the TF-IDF metric, use
it as part of their core algorithms to match a user's query to potential results. It forms an
essential part of the process that helps determine which documents are most relevant to a
specific query.

Here's a more detailed explanation:

1. Tokenization and Text Preprocessing: When a new webpage is encountered by a search engine's
   crawler, the content gets tokenized (split into words), and preprocessed (lowercased, stemmed,
   etc). Common words (a, the, is, etc.), called stop words, are often removed as part of the
   preprocessing step.
2. Term Frequency (TF) Calculation: Search engines calculate Term Frequency, which measures how
   frequently a term occurs in a document. If a word occurs often in a document, the TF will be
   higher. This is based on the assumption that a document talks more about certain topics and the
   terms associated with that topic will occur more frequently.
3. Inverse Document Frequency (IDF) Calculation: Next, engines calculate the Inverse Document
   Frequency for each word in their collection of documents (the web). This calculates how
   important a word is to understand the context. Words that occur in many documents from the
   collection are less informative and therefore have less IDF score.
4. TF-IDF Calculation: Each word in a document will have a TF-IDF score which is a multiplication
   of TF and IDF. So, words that occur often in a single document but not in many documents in the
   collection will have a high TF-IDF score.
5. Query Processing: When a user types a search query, the search engine processes the query in the
   same way it processes its documents (tokenizing and preprocessing). It then calculates a TF-IDF
   score for the query words.
6. When a search query is made, the search engine does not calculate the TF-IDF score for the query
   words based on a specific document. Instead, it uses the Inverse Document Frequency (IDF) part
   of the TF-IDF score, which it had pre-calculated for all the words across all documents (webpages)
   in its index.

Here's a simple explanation of how it works:
1. Inverse Document Frequency (IDF): This is a measure of how much information a given word
   provides. IDF is obtained by dividing the total number of documents (N) by the number of
   documents containing the word (n), and then applying logarithm to the result. This signifies the
   rareness of a term. IDF(term) = log_e(Total number of documents / Number of documents with term in
   it)
2. The IDF value reduces the weight of common words and increases the weight of words that are not
   common. The IDF values are pre-calculated and stored by the search engine for all words in the
   documents it had indexed.
3. When a search query is made, each word in the query is considered as a separate "mini document".
4. The Term Frequency (TF) of a word in the query is usually 1 (as most words appear only once in the
   query). For TF-IDF calculation, TF is the term frequency in the query and the IDF is the same as
   what was calculated previously and stored. Each term in the query would thus have a TF-IDF score
   which is a product of its Term Frequency (TF) in the query and its pre-calculated Inverse Document
   Frequency (IDF).
6. Document Ranking: The search engine then ranks the documents based on the cosine similarity of
   the TF-IDF vectors of the query and the documents. The cosine similarity measures the cosine of
   the angle between two vectors. This angle will be smaller for vectors that point roughly in the
   same direction, meaning that the query and the document share many terms with high TF-IDF weights.
   The documents are then returned to the user in the order of their relevance.
7. On-going updates: Search engines constantly update their database, as new webpages are created,
   old ones are deleted and existing ones updated. So, the TF-IDF calculation is an ongoing process
in the backend.  score for the query words.
*/
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::fs::{read_dir, DirEntry, File};
use std::io;
use std::io::{BufReader, Error, ErrorKind};
use xml::reader::EventReader;
use xml::reader::XmlEvent as ReaderEvent;
#[macro_use]
extern crate maplit;

#[derive(Ord, PartialOrd, Debug, Clone, PartialEq, Eq, Hash)]
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

fn document_to_count(
    document_to_term_to_count_map: &HashMap<Document, HashMap<String, usize>>,
) -> HashMap<Document, usize> {
    let mut totals: HashMap<Document, usize> = HashMap::new();
    for (document_name, term_to_count) in document_to_term_to_count_map {
        let term_count: usize = term_to_count.values().sum();
        totals.insert(document_name.clone(), term_count);
    }
    totals
}

type DocToTermCountMap = HashMap<Document, HashMap<String, usize>>;
type TermToDocCountMap = HashMap<String, HashMap<Document, usize>>;

fn document_and_term_to_count(
    document_to_content: &HashMap<Document, Option<DocumentContent>>,
) -> (DocToTermCountMap, TermToDocCountMap) {
    let mut _document_to_term_to_count_map: HashMap<Document, HashMap<String, usize>> =
        HashMap::new();
    let mut _term_to_document_to_count_map: HashMap<String, HashMap<Document, usize>> =
        HashMap::new();

    for document_path in document_to_content.keys() {
        let content_form_document: Vec<char> =
            content_from_document(document_path, document_to_content);
        let terms_in_content: Vec<String> = terms_in_content(content_form_document);
        // term_frequency is a HashMap that stores term frequencies for the current document.
        // the values are counts of how many times each term appears in the current document.
        let mut term_counting_map: HashMap<String, usize> = HashMap::new();
        let term_to_count_entry: &mut HashMap<String, usize> = _document_to_term_to_count_map
            .entry(document_path.clone())
            .or_default();
        for term in terms_in_content {
            let document_to_count_entry: &mut HashMap<Document, usize> =
                _term_to_document_to_count_map
                    .entry(term.clone())
                    .or_default();

            let count: &mut usize = term_counting_map.entry(term.clone()).or_insert(0);
            *count += 1;

            term_to_count_entry.insert(term, *count);
            document_to_count_entry.insert(document_path.clone(), *count);
        }
    }
    (
        _document_to_term_to_count_map,
        _term_to_document_to_count_map,
    )
}

// Term frequency, tf(t,d), is the relative frequency of term t within document d,
// i.e., the number of times that term t appears in document d divided by the total number of terms in document d
fn document_to_term_to_tf(
    document_to_term_to_count_map: &HashMap<Document, HashMap<String, usize>>,
    document_to_total_term_count_map: &HashMap<Document, usize>,
) -> HashMap<Document, HashMap<String, f64>> {
    let mut document_to_term_to_tf_map: HashMap<Document, HashMap<String, f64>> = HashMap::new();

    for (document, term_to_term_count_map) in document_to_term_to_count_map {
        if !term_to_term_count_map.is_empty() {
            // the total number of terms in document
            let total_term_count_in_document: f64 =
                *document_to_total_term_count_map.get(document).unwrap() as f64;

            // term frequency:
            // numerator = the number of times that term t occurs in document d
            // denominator = the total number of terms in document d
            for (term, term_count_in_document) in term_to_term_count_map {
                let tf: f64 = (*term_count_in_document as f64) / total_term_count_in_document;

                document_to_term_to_tf_map
                    .entry(document.clone())
                    .or_default()
                    .insert(term.clone(), tf);
            }
        }
    }

    document_to_term_to_tf_map
}

fn document_to_sorted_terms(
    document_to_term_to_frequency_map: &HashMap<Document, HashMap<String, f64>>,
) -> HashMap<Document, Vec<&String>> {
    let mut document_to_sorted_terms: HashMap<Document, Vec<&String>> = HashMap::new();

    document_to_term_to_frequency_map
        .keys()
        .for_each(|document_name| {
            let term_to_relative_term_frequency: &HashMap<String, f64> =
                document_to_term_to_frequency_map
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
    term_to_document_to_count_map: &HashMap<String, HashMap<Document, usize>>,
) -> usize {
    let mut unique_paths = HashSet::new();

    for (_, inner_map) in term_to_document_to_count_map.iter() {
        for (document_path, _) in inner_map.iter() {
            unique_paths.insert(document_path);
        }
    }

    unique_paths.len()
}

fn document_to_term_to_idf(
    document_to_term_to_count: &HashMap<Document, HashMap<String, usize>>,
    term_to_document_to_count_map: &HashMap<String, HashMap<Document, usize>>,
) -> HashMap<Document, HashMap<String, f64>> {
    let number_of_documents = document_to_term_to_count.keys().count();

    // println!("Number of documents: {}", number_of_documents);

    // collect idf for each term
    let term_to_idf_map: HashMap<String, f64> = term_to_document_to_count_map
        .keys()
        .map(|term| {
            let idf = inverse_document_frequency(
                number_of_documents,
                term_to_document_to_count_map,
                term.clone(),
            );
            (term.clone(), idf)
        })
        .collect();

    // associate term idf with each document that contains the term
    let document_to_term_to_idf_map: HashMap<Document, HashMap<String, f64>> =
        document_to_term_to_count
            .iter()
            .map(|(document, term_to_count_map)| {
                let filtered_term_to_idf_map: HashMap<String, f64> = term_to_count_map
                    .iter()
                    .filter_map(|(term, count)| {
                        if *count > 0 {
                            term_to_idf_map.get(term).map(|&idf| (term.clone(), idf))
                        } else {
                            None
                        }
                    })
                    .collect();

                (document.clone(), filtered_term_to_idf_map)
            })
            .collect();
    document_to_term_to_idf_map
}

fn inverse_document_frequency(
    number_of_documents: usize,
    term_to_document_to_count_map: &HashMap<String, HashMap<Document, usize>>,
    term: String,
) -> f64 {
    // println!("Number of documents: {}", number_of_documents);

    // println!(
    //     "Term to document to count: {:?}",
    //     term_to_document_to_count_map
    // );

    let number_of_documents_with_term =
        number_of_documents_with_term(term_to_document_to_count_map, &term);

    let idf: f64 = if number_of_documents_with_term != 0 {
        (number_of_documents as f64 / (0.0 + number_of_documents_with_term as f64)).log10()
    } else {
        0f64
    };

    // println!(
    //     "IDF inverse_document_frequency: term - {} - {} = {}",
    //     term, number_of_documents_with_term, idf
    // );

    idf
}

fn number_of_documents_with_term(
    term_to_document_to_count_map: &HashMap<String, HashMap<Document, usize>>,
    term: &String,
) -> usize {
    if let Some(document_frequency_map) = term_to_document_to_count_map.get(term) {
        document_frequency_map
            .iter()
            .filter(|(_, &count)| count > 0)
            .count()
    } else {
        0
    }
}

fn term_frequency_inverse_document_frequency(
    term: String,
    document: Document,
    document_to_term_to_count_map: &HashMap<Document, HashMap<String, usize>>,
    document_to_total_term_count_map: &HashMap<Document, usize>,
    term_to_document_to_count_map: HashMap<String, HashMap<Document, usize>>,
) -> f64 {
    let tf_map: HashMap<Document, HashMap<String, f64>> = document_to_term_to_tf(
        document_to_term_to_count_map,
        document_to_total_term_count_map,
    );

    let number_of_documents = document_to_term_to_count_map.keys().count();

    let idf: f64 = inverse_document_frequency(
        number_of_documents,
        &term_to_document_to_count_map,
        term.clone(),
    );

    // println!("TF map: {:?}", tf_map);
    // println!("IDF: {}", idf);

    let tf: &f64 = tf_map.get(&document).unwrap().get(&term).unwrap();

    // println!("TF: {}", tf);
    // println!("TF * IDF: {}", tf * idf);

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
    let (document_to_term_to_count_map, _): (
        HashMap<Document, HashMap<String, usize>>,
        HashMap<String, HashMap<Document, usize>>,
    ) = document_and_term_to_count(&document_to_content_map);

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

fn process_documents(
    documents: &[(Document, DocumentContent)],
) -> (DocToTermCountMap, TermToDocCountMap) {
    // Your function implementation here...
    let mut document_to_term_to_count_map: HashMap<Document, HashMap<String, usize>> = hashmap! {};
    let mut term_to_document_to_count_map: HashMap<String, HashMap<Document, usize>> = hashmap! {};

    for (document, doc_content) in documents.iter() {
        let words: Vec<&str> = doc_content.content.split_whitespace().collect();

        let mut term_to_count_map: HashMap<String, usize> = HashMap::new();
        for word in words {
            *term_to_count_map.entry(word.to_string()).or_insert(0) += 1;
        }

        document_to_term_to_count_map.insert(document.clone(), term_to_count_map.clone());

        for (term, count) in term_to_count_map.iter() {
            let entry: &mut HashMap<Document, usize> = term_to_document_to_count_map
                .entry(term.clone())
                .or_default();

            let new_count = entry.get(document).unwrap_or(&0) + *count;

            entry.insert(document.clone(), new_count);
        }
    }

    (document_to_term_to_count_map, term_to_document_to_count_map)
}

#[cfg(test)]
mod tests {

    use super::count_unique_documents;
    use super::document_and_term_to_count;
    use super::document_to_count;
    use super::document_to_sorted_terms;
    use super::document_to_term_to_idf;
    use super::document_to_term_to_tf;
    use super::inverse_document_frequency;
    use super::number_of_documents_with_term;
    use super::process_documents;
    use super::term_frequency_inverse_document_frequency;
    use super::DocToTermCountMap;
    use super::Document;
    use super::DocumentContent;
    use super::TermToDocCountMap;
    use pretty_assertions::assert_eq;
    use std::collections::HashMap;

    #[test]
    fn test_document_or_term_to_count() {
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
        let (document_to_term_to_count_map, _): (DocToTermCountMap, TermToDocCountMap) =
            document_and_term_to_count(&mock_document_contents);

        // then:
        let mut expected_result: HashMap<Document, HashMap<String, usize>> = HashMap::new();
        let mut term_count: HashMap<String, usize> = HashMap::new();
        term_count.insert(String::from("test"), 1);
        term_count.insert(String::from("document"), 1);
        term_count.insert(String::from("content"), 1);
        expected_result.insert(submission_document, term_count);

        assert_eq!(document_to_term_to_count_map, expected_result);
    }

    #[test]
    fn test_document_to_term_count() {
        let submission_document: Document = Document {
            path: String::from("submission.txt"),
        };

        // given:
        let mut mock_term_counts: HashMap<String, usize> = HashMap::new();
        mock_term_counts.insert(String::from("test"), 2);
        mock_term_counts.insert(String::from("document"), 1);
        mock_term_counts.insert(String::from("content"), 3);

        let mut mock_document_to_term_to_count: HashMap<Document, HashMap<String, usize>> =
            HashMap::new();
        mock_document_to_term_to_count.insert(submission_document.clone(), mock_term_counts);

        // when:
        let result: HashMap<Document, usize> = document_to_count(&mock_document_to_term_to_count);

        // then:
        let mut expected_result: HashMap<Document, usize> = HashMap::new();
        expected_result.insert(submission_document, 6);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_document_to_term_frequency() {
        let document_path: Document = Document {
            path: String::from("document.txt"),
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
        let term_to_document_to_frequency_map: HashMap<Document, HashMap<String, f64>> =
            document_to_term_to_tf(&document_to_occurrences, &total_term_counts);

        // then:
        let mut expected_term_to_frequency_map: HashMap<String, f64> = HashMap::new();
        expected_term_to_frequency_map.insert(String::from("term1"), 1.0 / 6.0);
        expected_term_to_frequency_map.insert(String::from("term2"), 2.0 / 6.0);
        expected_term_to_frequency_map.insert(String::from("term3"), 3.0 / 6.0);

        let mut expected_document_to_term_to_frequency_map: HashMap<
            Document,
            HashMap<String, f64>,
        > = HashMap::new();
        expected_document_to_term_to_frequency_map
            .insert(document_path, expected_term_to_frequency_map);

        assert_eq!(
            term_to_document_to_frequency_map,
            expected_document_to_term_to_frequency_map
        );
    }

    #[test]
    fn test_document_to_sorted_terms() {
        let document_path: Document = Document {
            path: String::from("document.txt"),
        };

        // given:
        let mut term_to_frequencies_map: HashMap<String, f64> = HashMap::new();
        let term1 = String::from("term1");
        let term2 = String::from("term2");
        let term3 = String::from("term3");
        term_to_frequencies_map.insert(term1.clone(), 0.3);
        term_to_frequencies_map.insert(term2.clone(), 0.2);
        term_to_frequencies_map.insert(term3.clone(), 0.1);

        let mut document_to_term_tofrequency_map: HashMap<Document, HashMap<String, f64>> =
            HashMap::new();
        document_to_term_tofrequency_map.insert(document_path.clone(), term_to_frequencies_map);

        // when:
        let document_to_sorted_terms_map: HashMap<Document, Vec<&String>> =
            document_to_sorted_terms(&document_to_term_tofrequency_map);

        // then:
        let expected_sorted_map = vec![&term1, &term2, &term3];
        let mut expected: HashMap<Document, Vec<&String>> = HashMap::new();
        expected.insert(document_path, expected_sorted_map);

        assert_eq!(document_to_sorted_terms_map, expected);
    }

    #[test]
    fn test_count_unique_documents() {
        let document_1: Document = Document {
            path: String::from("document_1.txt"),
        };

        let document_2: Document = Document {
            path: String::from("document_2.txt"),
        };

        let mut document_to_term_to_frequency_map_1: HashMap<Document, usize> = HashMap::new();
        document_to_term_to_frequency_map_1.insert(document_1.clone(), 10);
        document_to_term_to_frequency_map_1.insert(document_2.clone(), 100);

        let mut document_to_term_to_frequency_map_2: HashMap<Document, usize> = HashMap::new();
        document_to_term_to_frequency_map_2.insert(document_1.clone(), 11);
        document_to_term_to_frequency_map_2.insert(document_2.clone(), 110);

        let mut term_to_document_to_frequency_map: HashMap<String, HashMap<Document, usize>> =
            HashMap::new();

        term_to_document_to_frequency_map
            .insert(String::from("term_1"), document_to_term_to_frequency_map_1);
        term_to_document_to_frequency_map
            .insert(String::from("term_2"), document_to_term_to_frequency_map_2);

        // when:
        let number_of_unique_documents: usize =
            count_unique_documents(&term_to_document_to_frequency_map);

        // then:
        assert_eq!(number_of_unique_documents, 2);
    }
    #[test]
    fn test_inverse_document_frequency() {
        let document_1: Document = Document {
            path: String::from("document_1.txt"),
        };
        let document_2: Document = Document {
            path: String::from("document_2.txt"),
        };

        let number_of_documents: usize = 2;

        // given:
        let mut term_to_document_to_count_map: HashMap<String, HashMap<Document, usize>> =
            HashMap::new();

        let mut term1_document_to_frequency_map: HashMap<Document, usize> = HashMap::new();
        let mut term2_document_to_frequency_map: HashMap<Document, usize> = HashMap::new();
        let mut term3_document_to_frequency_map: HashMap<Document, usize> = HashMap::new();
        // Number of documents with term_1: 2
        term1_document_to_frequency_map.insert(document_1.clone(), 1);
        term1_document_to_frequency_map.insert(document_2.clone(), 1);

        // Number of documents with term_2: 1
        term2_document_to_frequency_map.insert(document_1.clone(), 2);

        // Number of documents with term_3: 1
        term3_document_to_frequency_map.insert(document_1.clone(), 3);

        let term1 = String::from("TERM1");
        let term2 = String::from("TERM2");
        let term3 = String::from("TERM3");
        term_to_document_to_count_map.insert(term1.clone(), term1_document_to_frequency_map);
        term_to_document_to_count_map.insert(term2.clone(), term2_document_to_frequency_map);
        term_to_document_to_count_map.insert(term3.clone(), term3_document_to_frequency_map);

        // when:
        //     [number_of_documents as f64 / number_of_documents_with_term as f64].log10()
        //     [2 / (2)].log10() = [1.0].log10()
        let idf_term_1: f64 =
            inverse_document_frequency(number_of_documents, &term_to_document_to_count_map, term1);

        //     [2 / (1)].log10() = [2.0].log10()
        let idf_term_2: f64 =
            inverse_document_frequency(number_of_documents, &term_to_document_to_count_map, term2);

        //     [2 / (1)].log10() = [2.0].log10()
        let idf_term_3: f64 =
            inverse_document_frequency(number_of_documents, &term_to_document_to_count_map, term3);

        // then:
        assert_eq!(idf_term_1, (2.0 / 2f64).log10()); // 0
        assert_eq!(idf_term_2, (2.0 / 1f64).log10()); // 0.30102999
        assert_eq!(idf_term_3, (2.0 / 1f64).log10()); // 0.30102999
    }

    #[test]
    fn test_document_to_term_to_idf() {
        let content_1: DocumentContent = DocumentContent {
            content: "the cats are in the house".to_string(),
        };
        let content_2: DocumentContent = DocumentContent {
            content: "the dogs are in the house and outside".to_string(),
        };
        let content_3: DocumentContent = DocumentContent {
            content: "the cats and dogs are friends".to_string(),
        };

        let documents: [(Document, DocumentContent); 3] = [
            (
                Document {
                    path: "Doc1".to_string(),
                },
                content_1,
            ),
            (
                Document {
                    path: "Doc2".to_string(),
                },
                content_2,
            ),
            (
                Document {
                    path: "Doc3".to_string(),
                },
                content_3,
            ),
        ];

        let (document_to_term_to_count_map, term_to_document_to_count_map): (
            DocToTermCountMap,
            TermToDocCountMap,
        ) = process_documents(&documents);

        let doc1 = Document {
            path: "Doc1".to_string(),
        };
        let doc2 = Document {
            path: "Doc2".to_string(),
        };
        let doc3 = Document {
            path: "Doc3".to_string(),
        };

        let doc1_map = hashmap! {
            "cats".to_string() => 0.17609125905568124,
            "the".to_string() => 0.0,
            "in".to_string() => 0.17609125905568124,
            "house".to_string() => 0.17609125905568124,
            "are".to_string() => 0.0,
        };

        let doc2_map = hashmap! {
            "dogs".to_string() => 0.17609125905568124,
            "and".to_string() => 0.17609125905568124,
            "are".to_string() => 0.0,
            "in".to_string() => 0.17609125905568124,
            "the".to_string() => 0.0,
            "house".to_string() => 0.17609125905568124,
            "outside".to_string() => 0.47712125471966244,
        };

        let doc3_map = hashmap! {
            "cats".to_string() => 0.17609125905568124,
            "and".to_string() => 0.17609125905568124,
            "the".to_string() => 0.0,
            "dogs".to_string() => 0.17609125905568124,
            "are".to_string() => 0.0,
            "friends".to_string() => 0.47712125471966244,
        };

        let docs_map: HashMap<Document, HashMap<String, f64>> = hashmap! {
            doc1.clone() => doc1_map,
            doc2.clone() => doc2_map,
            doc3.clone() => doc3_map,
        };

        // when:
        let result_document_to_term_to_idf: HashMap<Document, HashMap<String, f64>> =
            document_to_term_to_idf(
                &document_to_term_to_count_map,
                &term_to_document_to_count_map,
            );

        result_document_to_term_to_idf
            .iter()
            .for_each(|(result_document, result_term_to_idf)| {
                // println!("Doc: {}", result_document.path);

                let expected_term_to_idf: &HashMap<String, f64> =
                    docs_map.get(result_document).unwrap();

                expected_term_to_idf
                    .iter()
                    .for_each(|(expected_term, expected_idf)| {
                        let result_idf: &f64 = result_term_to_idf.get(expected_term).unwrap();
                        // println!(
                        //     "{} => {} => {} => {}",
                        //     result_document.path, expected_term, expected_idf, result_idf
                        // );
                        assert_eq!(result_idf, expected_idf);
                    });
            });
    }

    #[test]
    fn test_number_of_documents_with_term() {
        // given:
        let term_to_document_to_count_map: HashMap<String, HashMap<Document, usize>> = hashmap! {
            "and".to_string() => hashmap!{
                Document { path: "Doc2".to_string() } => 1,
                Document { path: "Doc3".to_string() } => 1,
            },
            "dogs".to_string() => hashmap!{
                Document { path: "Doc2".to_string() } => 1,
                Document { path: "Doc3".to_string() } => 1,
            },
            "outside".to_string() => hashmap!{
                Document { path: "Doc2".to_string() } => 1,
            },
            "in".to_string() => hashmap!{
                Document { path: "Doc1".to_string() } => 1,
                Document { path: "Doc2".to_string() } => 1,
            },
            "cats".to_string() => hashmap!{
                Document { path: "Doc1".to_string() } => 1,
                Document { path: "Doc3".to_string() } => 1,
            },
            "the".to_string() => hashmap!{
                Document { path: "Doc1".to_string() } => 2,
                Document { path: "Doc2".to_string() } => 2,
                Document { path: "Doc3".to_string() } => 1,
            },
            "house".to_string() => hashmap!{
                Document { path: "Doc1".to_string() } => 1,
                Document { path: "Doc2".to_string() } => 1,
            },
            "friends".to_string() => hashmap!{
                Document { path: "Doc3".to_string() } => 1,
            },
            "are".to_string() => hashmap!{
                Document { path: "Doc1".to_string() } => 1,
                Document { path: "Doc2".to_string() } => 1,
                Document { path: "Doc3".to_string() } => 1,
            },
        };

        // when:
        let the: usize =
            number_of_documents_with_term(&term_to_document_to_count_map, &"the".to_string());
        let and: usize =
            number_of_documents_with_term(&term_to_document_to_count_map, &"and".to_string());
        let are: usize =
            number_of_documents_with_term(&term_to_document_to_count_map, &"are".to_string());
        let unknown: usize =
            number_of_documents_with_term(&term_to_document_to_count_map, &"unknown".to_string());

        // then:
        assert_eq!(the, 3);
        assert_eq!(and, 2);
        assert_eq!(are, 3);
        assert_eq!(unknown, 0);
    }

    #[test]
    fn test_term_frequency_inverse_document_frequency() {
        let document_1: Document = Document {
            path: String::from("document_1.txt"),
        };

        // given:
        let term_1: String = String::from("TERM_1");

        let term_count_map: HashMap<String, usize> = hashmap! { term_1.clone() => 1 };

        let document_to_total_term_count_map: HashMap<Document, usize> =
            hashmap! { document_1.clone() => 1 };

        let document_to_term_to_count_map: HashMap<Document, HashMap<String, usize>> =
            hashmap! { document_1.clone() => term_count_map };

        let document_to_count_map: HashMap<Document, usize> = hashmap! { document_1.clone() => 1 };

        let term_to_document_to_count_map: HashMap<String, HashMap<Document, usize>> =
            hashmap! { term_1.clone() => document_to_count_map };

        // when:
        let result: f64 = term_frequency_inverse_document_frequency(
            term_1.clone(),
            document_1.clone(),
            &document_to_term_to_count_map,
            &document_to_total_term_count_map,
            term_to_document_to_count_map,
        );

        // (number_of_documents as f64 / (number_of_documents_with_term as f64)).log10()
        let idf: f64 = (1.0 / 1f64).log10();

        //     // numerator = the number of times that term t occurs in document d
        //     // denominator = the total number of terms in document d
        //     for (term, term_count_in_document) in term_to_term_count_map {
        //         let relative_term_frequency: f64 =
        //             (*term_count_in_document as f64) / total_term_count_in_document;
        //           tf =>   1                          /     1
        let tf: f64 = 1.0;
        assert_eq!(result, tf * idf);
    }

    #[test]
    fn test_term_frequency_inverse_document_frequency_2() {
        let doc_1: Document = Document {
            path: String::from("document_1.txt"),
        };
        let doc_2: Document = Document {
            path: String::from("document_2.txt"),
        };
        let doc_3: Document = Document {
            path: String::from("document_3.txt"),
        };

        let document_to_content: HashMap<Document, Option<DocumentContent>> = hashmap! {
            doc_1.clone() => Some(DocumentContent {
                content: String::from("the cats are in the house"),
            }),
            doc_2.clone() => Some(DocumentContent {
                content: String::from("the dogs are in the house and outside"),
            }),
            doc_3.clone() => Some(DocumentContent {
                content: String::from("the cats and dogs are friends"),
            }),
        };

        // Count occurrences of each term in each document
        let (document_to_term_to_count_map, _): (DocToTermCountMap, TermToDocCountMap) =
            document_and_term_to_count(&document_to_content);

        // Calculate total number of terms for each document
        let document_to_total_term_count_map: HashMap<Document, usize> =
            document_to_count(&document_to_term_to_count_map);

        // when:
        // Calculate term frequency for each term in each document
        let document_to_term_to_tf_map: HashMap<Document, HashMap<String, f64>> =
            document_to_term_to_tf(
                &document_to_term_to_count_map,
                &document_to_total_term_count_map,
            );

        // println!(
        //     "DOC1: {:?}",
        //     document_to_term_to_tf_map.get(&doc_1).unwrap()
        // );
        // println!(
        //     "DOC2: {:?}",
        //     document_to_term_to_tf_map.get(&doc_2).unwrap()
        // );
        // println!(
        //     "DOC3: {:?}",
        //     document_to_term_to_tf_map.get(&doc_3).unwrap()
        // );
        //

        // DOC1
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_1)
                .unwrap()
                .get("the")
                .unwrap(),
            2f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_1)
                .unwrap()
                .get("cats")
                .unwrap(),
            1f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_1)
                .unwrap()
                .get("are")
                .unwrap(),
            1f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_1)
                .unwrap()
                .get("in")
                .unwrap(),
            1f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_1)
                .unwrap()
                .get("house")
                .unwrap(),
            1f64 / 6f64
        );

        // DOC2
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_2)
                .unwrap()
                .get("the")
                .unwrap(),
            2f64 / 8f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_2)
                .unwrap()
                .get("dogs")
                .unwrap(),
            1f64 / 8f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_2)
                .unwrap()
                .get("are")
                .unwrap(),
            1f64 / 8f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_2)
                .unwrap()
                .get("in")
                .unwrap(),
            1f64 / 8f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_2)
                .unwrap()
                .get("house")
                .unwrap(),
            1f64 / 8f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_2)
                .unwrap()
                .get("and")
                .unwrap(),
            1f64 / 8f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_2)
                .unwrap()
                .get("outside")
                .unwrap(),
            1f64 / 8f64
        );

        // DOC3
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_3)
                .unwrap()
                .get("the")
                .unwrap(),
            1f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_3)
                .unwrap()
                .get("dogs")
                .unwrap(),
            1f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_3)
                .unwrap()
                .get("and")
                .unwrap(),
            1f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_3)
                .unwrap()
                .get("cats")
                .unwrap(),
            1f64 / 6f64
        );
        assert_eq!(
            *document_to_term_to_tf_map
                .get(&doc_3)
                .unwrap()
                .get("friends")
                .unwrap(),
            1f64 / 6f64
        );
    }

    #[test]
    fn test_tf_idf() {
        let content_1: DocumentContent = DocumentContent {
            content: "the cats are in the house".to_string(),
        };
        let content_2: DocumentContent = DocumentContent {
            content: "the dogs are in the house and outside".to_string(),
        };
        let content_3: DocumentContent = DocumentContent {
            content: "the cats and dogs are friends".to_string(),
        };

        let documents: [(Document, DocumentContent); 3] = [
            (
                Document {
                    path: "Doc1".to_string(),
                },
                content_1,
            ),
            (
                Document {
                    path: "Doc2".to_string(),
                },
                content_2,
            ),
            (
                Document {
                    path: "Doc3".to_string(),
                },
                content_3,
            ),
        ];

        let (document_to_term_to_count_map, term_to_document_to_count_map): (
            DocToTermCountMap,
            TermToDocCountMap,
        ) = process_documents(&documents);

        // Calculate total number of terms for each document
        let document_to_total_term_count_map: HashMap<Document, usize> =
            document_to_count(&document_to_term_to_count_map);
        // when:
        // Calculate term frequency for each term in each document
        let document_to_term_to_tf_map: HashMap<Document, HashMap<String, f64>> =
            document_to_term_to_tf(
                &document_to_term_to_count_map,
                &document_to_total_term_count_map,
            );
        let result_document_to_term_to_idf: HashMap<Document, HashMap<String, f64>> =
            document_to_term_to_idf(
                &document_to_term_to_count_map,
                &term_to_document_to_count_map,
            );
    }
}

