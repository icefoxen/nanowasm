extern crate clap;
extern crate parity_wasm;

use std::sync::Arc;
use clap::{Arg, App};
use std::collections::HashMap;
use std::borrow::Cow;
use std::sync::Weak;

use parity_wasm::interpreter::{CallerContext, Error, RuntimeValue, UserFunctionExecutor,
UserFunctionDescriptor, ModuleInstanceInterface, UserDefinedElements, ModuleInstance};

use parity_wasm::interpreter;
use parity_wasm::elements;

struct MyExecutor;

impl UserFunctionExecutor for MyExecutor {
    fn execute(
        &mut self,
        name: &str,
        context: CallerContext,
    ) -> Result<Option<RuntimeValue>, Error> {
        match name {
            "add" => {
                // fn add(a: u32, b: u32) -> u32
                let b = context.value_stack.pop_as::<u32>()?;
                let a = context.value_stack.pop_as::<u32>()?;
                let sum = a + b;
                Ok(Some(RuntimeValue::I32(sum as i32)))
            }
            "printi" => {
                // fn printi(a: u32)
                let a = context.value_stack.pop_as::<u32>()?;
                println!("Value is: {}", a);
                Ok(None)
            }
            "foo" => {
                // fn foo()
                println!("Foo");
                Ok(None)
            }


            _ => Err(Error::Trap("not implemented".into()).into()),
        }
    }
}

fn main() {
    // Parse inputs
    let matches = App::new("nanowasm")
        .version("0.1")
        .about("A standalone WebAssembly interpreter in Rust.")
        .arg(Arg::with_name("file")
             .required(true))
        .get_matches();

    let input_file = matches.value_of("file").expect("file argument is required; should never happen.");

    println!("Input file is {}", input_file);

    // Create add-in module
    let elts = UserDefinedElements {
        globals: HashMap::new(),
        functions: Cow::Owned(vec![
            UserFunctionDescriptor::Heap("foo".to_owned(), vec![], None),
            UserFunctionDescriptor::Heap("printi".to_owned(), vec![elements::ValueType::I32], None),
        ]),
        executor: Some(MyExecutor),
    };

    let empty_module = ModuleInstance::new(
        Weak::new(),
        "foo_module".to_owned(),
        parity_wasm::elements::Module::new(vec![])
    ).unwrap();
    let native = interpreter::native_module(Arc::new(empty_module), elts).unwrap();


    // Load and instantiate given file
    let module = parity_wasm::deserialize_file(input_file).unwrap();
    assert!(module.code_section().is_some());

    let m = interpreter::ProgramInstance::new();
    let foo_instance = m.insert_loaded_module("foo_module", native).unwrap();
    
    let inc_instance = m.add_module("inc", module, None)
        .expect("Failed to instantiate module loaded from file");
    
    //let start_res = inc_instance.run_start_function().unwrap();
    //println!("Result of start function: {:?}", start_res);
}
