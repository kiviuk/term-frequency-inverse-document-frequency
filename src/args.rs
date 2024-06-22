#[derive(Debug)]
pub struct Path(String);

#[derive(Debug)]
pub struct Terms(Vec<String>);

#[derive(Debug)]
pub(crate) enum Command {
    Index(Path),
    Search(Terms),
    Error(String),
}

pub(crate) fn parse_args(args: Vec<String>) -> Vec<String> {
    args
}

fn parse_index_path(args: Vec<String>) -> Command {
    if args.len() != 3 {
        return Command::Error("Invalid number of arguments".to_string());
    }

    match args[1].as_str() {
        "index" =>
            Command::Index(Path(args[2].to_string())),
        _ => Command::Error("Not an index command".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_index_path() {

        let args: Vec<String> = vec!["my_program".to_string(), "index".to_string(), "/path/to/files".to_string()];
        let command: Command = parse_index_path(args);

        println!("{:?}", command);

        match command {
            Command::Index(Path(path)) => assert_eq!(path, "/path/to/files"),
            _ => assert!(false),
        }
    }

    // index "path"
    // search "term1 term2..."
    #[test]
    fn test_parse_args() {
        let args: Vec<String> = parse_args(["index /path/to/files".to_string(), "search term1 term2".to_string()].to_vec());
        assert!(!args.is_empty());
    }
}