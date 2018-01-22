extern crate clap;
extern crate nanowasm;
extern crate parity_wasm;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::env;

use clap::{App, Arg};
use nanowasm::types::*;
use nanowasm::loader::*;
use nanowasm::interpreter::*;


// Structs for loading wasm test suite JSON specification files.

#[derive(Deserialize, Debug)]
pub struct RuntimeValue {
    #[serde(rename = "type")]
    pub value_type: String,
    pub value: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Action {
    #[serde(rename = "invoke")]
    Invoke { field: String, args: Vec<RuntimeValue> }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Command {
    #[serde(rename = "module")]
    Module { line: u64, filename: String },
    #[serde(rename = "assert_return")]
    AssertReturn { 
        line: u64, 
        action: Action,
        expected: Vec<RuntimeValue>,
    },
    #[serde(rename = "assert_trap")]
    AssertTrap {
        line: u64,
        action: Action,
        text: String,
    },
}

#[derive(Deserialize, Debug)]
pub struct Spec {
    pub source_filename: String,
    pub commands: Vec<Command>,
}



fn main() {
    println!("Args are: {:?}", env::args());

    // Parse inputs
    let matches = App::new("nanowasm")
        .version("0.1")
        .about("A standalone WebAssembly interpreter in Rust.")
        .arg(Arg::with_name("file").required(true))
        .arg(Arg::with_name("run-all-exports")
             .long("run-all-exports")
             .help("TODO")
             // TODO: Fix wabt's test suite so it doesn't give this more than once. :/
             .multiple(true))
        .arg(Arg::with_name("Enable threads")
             .long("enable-threads")
             .help("TODO"))
        .arg(Arg::with_name("verbose")
             .short("v")
             .long("verbose")
             .help("Enable verbose output"))
        .get_matches();

    let input_file = matches
        .value_of("file")
        .expect("file argument is required; should never happen.");

    println!("Input file is {}", input_file);

    let module = parity_wasm::deserialize_file(input_file).unwrap();
    let mut mod_instance = LoadedModule::new("fib", module);
    mod_instance.validate();
    let mut interp = Interpreter::new().with_module(mod_instance);

    let start_addr = FunctionAddress::new(1);
    interp.run(start_addr, &vec![Value::I32(30)]);
    println!("{:#?}", interp);
}
