extern crate clap;
extern crate nanowasm;
extern crate parity_wasm;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate wabt;

use std::env;
use std::fs::File;
use std::path;

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
    Invoke {
        module: Option<String>,
        field: String,
        args: Vec<RuntimeValue>,
    },
    #[serde(rename = "get")]
    Get {
        module: Option<String>,
        field: String,
    }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Command {
    #[serde(rename = "module")]
    Module {
        line: u64,
        name: Option<String>,
        filename: String
    },
    #[serde(rename = "assert_return")]
    AssertReturn {
        line: u64,
        action: Action,
        expected: Vec<RuntimeValue>,
    },
    #[serde(rename = "assert_return_canonical_nan")]
    AssertReturnCanonicalNan {
        line: u64,
        action: Action,
    },
    #[serde(rename = "assert_return_arithmetic_nan")]
    AssertReturnArithmeticNan {
        line: u64,
        action: Action,
    },
    #[serde(rename = "assert_trap")]
    AssertTrap {
        line: u64,
        action: Action,
        text: String,
    },
    #[serde(rename = "assert_invalid")]
    AssertInvalid {
        line: u64,
        filename: String,
        text: String,
    },
    #[serde(rename = "assert_malformed")]
    AssertMalformed {
        line: u64,
        filename: String,
        text: String,
    },
    #[serde(rename = "assert_uninstantiable")]
    AssertUninstantiable {
        line: u64,
        filename: String,
        text: String,
    },
    #[serde(rename = "assert_exhaustion")]
    AssertExhaustion {
        line: u64,
        action: Action,
    },
    #[serde(rename = "assert_unlinkable")]
    AssertUnlinkable {
        line: u64,
        filename: String,
        text: String,
    },
    #[serde(rename = "register")]
    Register {
        line: u64,
        name: Option<String>,
        #[serde(rename = "as")]
        as_name: String,
    },
    #[serde(rename = "action")]
    Action {
        line: u64,
        action: Action,
    },
}

#[derive(Deserialize, Debug)]
pub struct Spec {
    pub source_filename: String,
    pub commands: Vec<Command>,
}

fn run_spec(spec: &Spec, file_dir: &path::Path) -> Result<(), ()> {
    let mut current_module = None;
    for c in &spec.commands {
        println!("Command {:?}", c);
        match *c {
            Command::Module { ref filename, .. } => {
                // Annoyingly, lots of assertions don't have an actual module name
                // attached to them, just use the most recently loaded one.  I guess
                // that prevents us from having to re-load the module multiple times
                // though...
                let mut file_path: path::PathBuf = file_dir.into();
                file_path.push(filename);
                print!("Loading module from file {}: ", filename);
                // TODO: Make it return errors properly.
                let module = parity_wasm::deserialize_file(&file_path)
                    .unwrap();
                current_module = Some(module.clone());
                print!("Loaded ");

                let loaded_module = LoadedModule::new(filename, module).unwrap();
                let validated_module = loaded_module.validate();
                let mut _interp = Interpreter::new().with_module(validated_module)
                    .expect("Could not initialize module for test");

                print!("Instantiated ");
                println!("Ok.");

            },
            Command::AssertInvalid { ref filename, ref text, line: _line } => {
                let mut file_path: path::PathBuf = file_dir.into();
                file_path.push(filename);
                let module = parity_wasm::deserialize_file(&file_path).unwrap();                    
                match LoadedModule::new(filename, module) {
                    Ok(loaded_module) => {
                        let validated_module = loaded_module.validate();
                    },
                    Err(Error::Invalid{ref reason, ..}) => {
                        assert!(reason.contains(text), "Expected an ErrorInvalid with text '{}', instead got {}", text, reason);
                    },
                    Err(e) => {
                        panic!("AssertInvalid: Should have gotten an Error::Invalid, instead got: {:?}", e);
                    }
                }


            },
            Command::AssertUninstantiable { .. } | 
            Command::AssertExhaustion { .. } |
            Command::AssertUnlinkable { .. } |
            Command::AssertReturn { .. } |
            Command::AssertReturnCanonicalNan { .. } |
            Command::AssertReturnArithmeticNan { .. } |
            Command::AssertTrap { .. } | 
            Command::AssertMalformed { .. } => {
                println!("TODO: Need to test this assertion case: {:?}", c);
            },
            Command::Register { .. } => (),
            Command::Action { .. } => (),
        }
    }
    Ok(())
}


fn main() {
    // Parse inputs
    let matches = App::new("nanowasm")
        .version("0.1")
        .about("A standalone WebAssembly interpreter in Rust.")
        .arg(Arg::with_name("load-test")
             .long("load-test")
             .help("Specify a json test specification to load."))
        .arg(Arg::with_name("file").required(true))
        .arg(Arg::with_name("verbose")
             .short("v")
             .long("verbose")
             .help("Enable verbose output (TODO: Make it enable verbose output)"))
        .get_matches();

    let input_file = matches
        .value_of("file")
        .expect("file argument is required; should never happen.");

    println!("Input file is {}", input_file);

    if matches.is_present("load-test") {
        let mut f = File::open(input_file)
            .expect("Failed to open JSON file");
        let file_dir = path::Path::new(input_file)
            .parent()
            .unwrap_or(path::Path::new("./"));
        let spec: Spec = serde_json::from_reader(&mut f)
            .expect("Failed to parse JSON file");
        run_spec(&spec, file_dir)
            .expect("Spec test failed");
    } else {
        let module = parity_wasm::deserialize_file(input_file).unwrap();
        let loaded_module = LoadedModule::new(input_file, module).unwrap();
        let validated_module = loaded_module.validate();
        let mut interp = Interpreter::new().with_module(validated_module)
            .expect("Could not initialize module");
        
        let start_addr = FunctionAddress::new(1);
        interp.run(start_addr, &vec![Value::I32(30)]);
        println!("{:#?}", interp);
    }
}
