
use clap::{Arg, Command};
use std::path::PathBuf;

pub(crate) fn parse_args(args: Vec<String>) -> Vec<String> {
    args
}

fn main() {
    let matches = Command::new("My File Tool")
        .version("1.0")
        .author("Your Name")
        .about("Indexes and searches files")
        .subcommand(
            Command::new("index")
                .about("Indexes files in specified paths")
                .arg(
                    Arg::new("path")
                        .long("path")
                        .value_name("DIRECTORY")
                        .help("Directory path to index")
                        .num_args(1..)
                ),
        )
        .subcommand(
            Command::new("search")
                .about("Searches for a string in indexed files")
                .arg(
                    Arg::new("query")
                        .help("The string to search for")
                        .required(true)
                        .index(1),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("index", index_matches)) => {
            if let Some(paths) = index_matches.get_many::<String>("path") {
                for path in paths {
                    let path = PathBuf::from(path);
                    if let Some(path_str) = path.to_str() { // Store string here
                        let recursive = path_str.ends_with(":r");
                        let clean_path = if recursive {
                            path.with_file_name(path_str.trim_end_matches(":r"))
                        } else {
                            path
                        };
                        println!("Indexing path: {:?}, Recursive: {}", clean_path, recursive);
                        // Add your indexing logic here
                    }
                }
            }
        }
        Some(("search", search_matches)) => {
            if let Some(query) = search_matches.get_one::<String>("query") {
                println!("Searching for: {}", query);
                // Add your search logic here
            }
        }
        _ => println!("Please use a valid subcommand. Use --help for more information."),
    }
}