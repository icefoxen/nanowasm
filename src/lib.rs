//! An attempt at creating an embeddable wasm interpreter that's
//! bloody easier to use and debug than the `parity-wasm` one.

extern crate parity_wasm;

use parity_wasm::elements;

use std::collections::HashMap;


/// A wasm program consisting of multiple modules that
/// have been loaded, validated and are ready to execute.
pub struct Program {
    modules: HashMap<String, ModuleInstance>,
}

impl Program {
    fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    fn with_module(mut self, name: &str, module: ModuleInstance) -> Self {
        self.modules.insert(name.to_owned(), module);
        self
    }
}

/// A type signature for a function type
pub struct FuncType {
    params: Vec<elements::ValueType>,
    return_type: Option<elements::ValueType>,
}

pub struct Func {
}

pub struct Table {
}

pub struct Memory {
}

pub struct Global {
}

/// A loaded wasm module
pub struct ModuleInstance {
    types: Vec<FuncType>,
    funcs: Vec<Func>,
    /// wasm 1.0 defines only a single table
    tables: Table,
    /// wasm 1.0 defines only a single memory.
    mem: Memory,
    globals: Vec<Global>,
    
}

impl ModuleInstance {
    fn new(module: elements::Module) -> Self {
        Self {
            types: vec![],
            funcs: vec![],
            tables: Table {},
            mem: Memory {},
            globals: vec![],
        }
    }
}

/// An interpreter which runs a particular program.
pub struct Interpreter {
    value_stack: Vec<parity_wasm::RuntimeValue>,
    program: Program,
}

impl Interpreter {
    fn new(program: Program) -> Self {
        Self {
            value_stack: vec![],
            program: program,
        }
    }
}

#[cfg(test)]
mod tests {
    use parity_wasm;
    use super::*;
    
    #[test]
    fn test_create() {
        let input_file = "test_programs/inc.wasm";
        let module = parity_wasm::deserialize_file(input_file).unwrap();
        let mod_instance = ModuleInstance::new(module);
        let program = Program::new()
            .with_module("inc", mod_instance);
        let interpreter = Interpreter::new(program);
    }
}
