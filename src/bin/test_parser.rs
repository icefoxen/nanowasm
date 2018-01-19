//! A start on a parser/runner for the wast test format.  :|
//!
//! What we're going to have to do is parse it, pull out the actual code,
//! assemble it (if necessary),
//!
//! The test script syntax is described here: https://github.com/WebAssembly/spec/blob/master/interpreter/README.md#scripts
//! though maybe not entirely correctly.

extern crate clap;
#[macro_use]
extern crate nom;

use std::env;
use std::fs;
use std::io::Read;

use clap::{App, Arg};

#[derive(Debug, Clone, PartialEq)]
enum Ast {
    Symbol(String),
    Number(String),
    String(String),
    List(Vec<Ast>)
}

named!(parens(&str) -> Ast,
       map!(
            ws!(
                delimited!(tag!("("),
                           expr,
                           tag!(")"))),
           Ast::List
       ));


// TODO: Does not yet parse strings with escaped quotes or other characters!  Aieeee.
// Turns out I couldn't figure out how to do this in Rivet either.
named!(string(&str) -> Ast,
       map!(
           delimited!(tag!("\""), is_not_s!("\""), tag!("\"")),
           |x| Ast::String(x.to_owned())));

named!(number(&str) -> Ast, map!(ws!(nom::digit), |x| Ast::Number(x.to_owned())));

named!(symbol(&str) -> Ast, map!(ws!(nom::alphanumeric), |x| Ast::Symbol(x.to_owned())));

named!(expr(&str) -> Vec<Ast>, many0!(alt!(number | symbol | string | parens)));

fn main() {
    println!("Args are: {:?}", env::args());

    // Parse inputs
    let matches = App::new("nanowasm")
        .version("0.1")
        .about("A standalone WebAssembly interpreter in Rust.")
        .arg(Arg::with_name("file").required(true))
        .get_matches();

    let input_file = matches
        .value_of("file")
        .expect("file argument is required; should never happen.");

    println!("Input file is {}", input_file);
    let mut v = vec![];
    let mut f = fs::File::open(input_file).unwrap();
    f.read_to_end(&mut v).unwrap();
    let s = String::from_utf8(v).unwrap();
    println!("String: {}", s);
}

#[cfg(test)]
mod tests {
    use nom::IResult;
    use super::*;
    #[test]
    fn test_parse() {
        let s = "()";
        let res = parens("(foo bar bop (rawr hi there \"joe\" 10))");
        println!("{:?}", res);
        assert!(false);
    }
}
