//! An attempt at creating an embeddable wasm interpreter that's
//! bloody easier to use and debug than the `parity-wasm` one.

extern crate parity_wasm;

use parity_wasm::elements;

use std::collections::HashMap;


/// A type signature for a function type
#[derive(Debug, Clone)]
pub struct FuncType {
    params: Vec<elements::ValueType>,
    return_type: Option<elements::ValueType>,
}

#[derive(Debug, Clone)]
pub struct Func {
}

#[derive(Debug, Clone)]
pub struct Table {
}

#[derive(Debug, Clone)]
pub struct Memory {
}

#[derive(Debug, Clone)]
pub struct Global {
}

/// A loaded wasm module
#[derive(Debug, Clone)]
pub struct ModuleInstance {
    /// Function type vector
    types: Vec<FuncType>,
    /// Function value vector
    funcs: Vec<Func>,
    /// wasm 1.0 defines only a single table
    tables: Table,
    /// wasm 1.0 defines only a single memory.
    mem: Memory,
    globals: Vec<Global>,

    validated: bool,
}

impl ModuleInstance {
    /// Instantiates and initializes a new module.
    /// Does NOT run the start function though!
    fn new(module: elements::Module) -> Self {
        assert_eq!(module.version(), 1);
        
        if let Some(code) = module.code_section() {
            println!("Code: {:?}", code.bodies());
        }

        if let Some(types) = module.type_section() {
            println!("Types: {:?}", types.types());
        }

        if let Some(functions) = module.function_section() {
            let funcs = functions.entries().iter()
                .map(|x| x.type_ref())
                .collect::<Vec<_>>();
            println!("Functions: {:?}", &funcs);
        }

        if let Some(imports) = module.import_section() {
            println!("Imports: {:?}", imports);
        }

        if let Some(exports) = module.export_section() {
            println!("Exports: {:?}", exports);
        }

        if let Some(start) = module.start_section() {
            println!("Start: {:?}", start);
        }

        // TODO: tables, elements, memory, data,
        // globals.

        Self {
            types: vec![],
            funcs: vec![],
            
            tables: Table {},
            mem: Memory {},
            globals: vec![],

            validated: false,
        }
    }

    /// Validates the module: makes sure types are correct,
    /// all the indices into various parts of the module are valid, etc.
    fn validate(&mut self) {
        self.validated = true;
    }
}

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

    /// Builder function to add a loaded and validated module to the
    /// program.
    ///
    /// Essentially, this does the dynamic linking, and should cause
    /// errors to happen if there are invalid/dangling references.
    /// So, you have to load all the modules in order of dependencies.
    ///
    /// We could load all the modules in arbitrary order, then validate+link
    /// them at the end, but meh.
    fn with_module(mut self, name: &str, module: ModuleInstance) -> Self {
        assert!(module.validated);
        self.modules.insert(name.to_owned(), module);
        self
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
        let mut mod_instance = ModuleInstance::new(module);
        mod_instance.validate();
        let program = Program::new()
            .with_module("inc", mod_instance);
        let interpreter = Interpreter::new(program);
        assert!(false);
    }
}
