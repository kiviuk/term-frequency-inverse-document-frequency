use std::fs::File;
use std::io;
use xml::reader::{EventReader, XmlEvent};
use std::io::BufReader;
use xml::reader::XmlEvent as ReaderEvent;


// fn read_entire_xml_file(file_path: &str) -> io::Result<String> {
//
//     let file: File = File::open(file_path)
//         .expect(&format!("Error: Something went wrong reading the file '{}'", file_path));
//
//     let file: BufReader<File> = BufReader::new(file);
//
//     let parser: EventReader<BufReader<File>> = EventReader::new(file);
//
//     // create a buffer to hold the characters
//     let mut buffer = String::new();
//
//     for event in parser {
//         match event {
//             Ok(ReaderEvent::Characters(s)) => {
//                 buffer.push_str(&s);
//             }
//             _ => {}
//         }
//     }
//
//     Ok(buffer)
// }

fn main() {
    let file_path = "docs.gl/gl4/glClear.xhtml";

    let file: File = File::open(file_path)
        .expect(&format!("Error: Something went wrong reading the file '{}'", file_path));

    let file: BufReader<File> = BufReader::new(file);

    let parser: EventReader<BufReader<File>> = EventReader::new(file);

    // create a buffer to hold the characters
    let mut buffer = String::new();

    for event in parser {
        match event {
            Ok(ReaderEvent::Characters(s)) => {
                buffer.push_str(&s);
            }
            _ => {}
        }
    }

    println!("{}", buffer);
}