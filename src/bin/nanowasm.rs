extern crate clap;
extern crate parity_wasm;
extern crate nanowasm;

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
