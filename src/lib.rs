//! An attempt at creating an embeddable wasm interpreter that's
//! bloody easier to use and debug than the `parity-wasm` one.

extern crate parity_wasm;

use parity_wasm::elements;

use std::collections::HashMap;


/// A type signature for a function type, intended to
/// go into the `types` section of a module.
///
/// parity-wasm has `elements::FunctionType` which is basically
/// this but with some extra serialization info we don't
/// need for execution, so we make our own.
/// It also wraps it in `elements::Type`, of which the
/// only member is a `FunctionType`... that might get extended
/// by wasm in the future but for now we just omit it.
#[derive(Debug, Clone)]
pub struct FuncType {
    params: Vec<elements::ValueType>,
    return_type: Option<elements::ValueType>,
}

impl<'a> From<&'a elements::Type> for FuncType {
    fn from(t: &'a elements::Type) -> Self {
        match *t {
            elements::Type::Function(ref ft) => {
                Self {
                    params: ft.params().to_owned(),
                    return_type: ft.return_type(),
                }
            }
        }
    }
}

/// An index into a module's `type` vector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeIdx(usize);

/// An actual value used at runtime.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl Value {

    /// Takes a `ValueType` and returns a new, zero'ed `Value`
    /// of the appropriate type.
    fn default_from_type(t: elements::ValueType) -> Self {
        match t {
            elements::ValueType::I32 => Value::I32(0),
            elements::ValueType::I64 => Value::I64(0),
            elements::ValueType::F32 => Value::F32(0.0),
            elements::ValueType::F64 => Value::F64(0.0),
        }
    }
}


/// Memory for a variable.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VariableSlot {
    variable_type: elements::ValueType,
    value: Value,
}

impl VariableSlot {
    /// Takes a slice of `Local`'s (local variable *specifications*),
    /// and creates a vec of `VariableSlot`'s matching them.
    fn from_locals(locals: &[elements::Local]) -> Vec<VariableSlot> {
        let num_local_slots = locals.iter()
            .map(|x| x.count() as usize)
            .sum();
        let mut v = Vec::with_capacity(num_local_slots);
        for local in locals {
            for i in 0..local.count() {
                let slot = VariableSlot {
                    variable_type: local.value_type(),
                    value: Value::default_from_type(local.value_type()),
                };
                v.push(slot);
            }
        }
        v
    }
}



/// A function ready to be executed.
#[derive(Debug, Clone)]
pub struct Func {
    typeidx: TypeIdx,
    locals: Vec<VariableSlot>,
    body: Vec<elements::Opcode>,
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

        let mut m = Self {
            types: vec![],
            funcs: vec![],
            
            tables: Table {},
            mem: Memory {},
            globals: vec![],

            validated: false,
        };

        // In `wat` a function's type index is declared
        // along with the function, but in the binary `wasm`
        // it's in its own section; the `code` section
        // contains the code and the `function` section
        // contains signature indices.  We join them back
        // together here.
        if let (Some(code), Some(functions)) = (module.code_section(), module.function_section()) {
            assert_eq!(code.bodies().len(), functions.entries().len());
            let converted_funcs = code.bodies().iter()
                .zip(functions.entries())
                .map(|(c, f)| Func {
                    typeidx: TypeIdx(f.type_ref() as usize),
                    locals: VariableSlot::from_locals(c.locals()),
                    body: c.code().elements().to_owned(),
                });
            m.funcs.extend(converted_funcs);
        } else {
            panic!("Code section exists but type section does not, or vice versa!");
        }
        
        if let Some(code) = module.code_section() {
            println!("Code: {:?}", code.bodies());
        }

        if let Some(types) = module.type_section() {
            let functypes = types.types().iter()
                .map(From::from);
            m.types.extend(functypes);
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

        m
    }

    /// Validates the module: makes sure types are correct,
    /// all the indices into various parts of the module are valid, etc.
    fn validate(&mut self) {
        self.validated = true;
    }
}

/// A wasm program consisting of multiple modules that
/// have been loaded, validated and are ready to execute.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
        println!("{:#?}", program);
        let interpreter = Interpreter::new(program);
        assert!(false);
    }
}
