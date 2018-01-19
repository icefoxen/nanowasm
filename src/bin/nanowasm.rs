extern crate clap;
extern crate nanowasm;
extern crate parity_wasm;

use std::env;

use clap::{App, Arg};
use nanowasm::types::*;
use nanowasm::loader::*;
use nanowasm::interpreter::*;

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
