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

/// An index into a module's `function` vector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FuncIdx(usize);

/// A function ready to be executed.
#[derive(Debug, Clone)]
pub struct Func {
    typeidx: TypeIdx,
    locals: Vec<VariableSlot>,
    body: Vec<elements::Opcode>,
}

/// A table.
///
/// Currently, a table is *purely* a mapping of
/// integers to anyfunc's.
///
/// Obviously mainly there for integration with Javascript.
#[derive(Debug, Clone)]
pub struct Table {
    /// Actual data
    data: Vec<FuncIdx>,
    /// Maximum size
    max: Option<u32>,
}

impl Table {
    fn new() -> Self {
        Self {
            data: vec![],
            max: None,
        }
    }

    /// Resizes the underlying storage, zero'ing it in the process.
    /// For a Table it fills it with `FuncIdx(0)`, even
    /// in the case that there IS no function 0.
    fn fill(&mut self, size: u32) {
        let mut v = Vec::with_capacity(size as usize);
        v.resize(size as usize, FuncIdx(0));
        self.data = v;
        self.max = Some(size);
    }

}

/// A structure containing a memory space.
#[derive(Debug, Clone)]
pub struct Memory {
    /// Actual data
    data: Vec<u8>,
    /// Maximum size, in units of 65,536 bytes
    max: Option<u32>,
}

impl Memory {
    const MEMORY_PAGE_SIZE: usize = 65_536;
    
    pub fn new(size: Option<u32>) -> Self {
        let mut mem = Self {
            data: vec![],
            max: None,
        };
        if let Some(size) = size {
            mem.fill(size);
        }
        mem
    }

    /// Resizes the underlying storage, zero'ing it in the process.
    fn fill(&mut self, size: u32) {
        use std::usize;
        let v_size = usize::checked_mul(Self::MEMORY_PAGE_SIZE, size as usize)
            .expect("Tried to allocate memory bigger than usize!");
        let mut v = Vec::with_capacity(v_size);
        v.resize(v_size, 0);
        self.data = v;
        self.max = Some(size);
    }
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
    /// Index of start function, if any.
    start: Option<usize>,
    /// wasm 1.0 defines only a single table,
    /// but we can import multiple of them?
    tables: Table,
    /// wasm 1.0 defines only a single memory.
    mem: Memory,
    globals: Vec<Global>,

    validated: bool,
}

impl ModuleInstance {
    /// Instantiates and initializes a new module.
    /// Does NOT validate or run the start function though!
    fn new(module: elements::Module) -> Self {
        assert_eq!(module.version(), 1);

        let mut m = Self {
            types: vec![],
            funcs: vec![],
            start: None,
            
            tables: Table::new(),
            mem: Memory::new(None),
            globals: vec![],

            validated: false,
        };

        // Allocate types
        if let Some(types) = module.type_section() {
            let functypes = types.types().iter()
                .map(From::from);
            m.types.extend(functypes);
        }


        // Allocate functions
        //
        // In `wat` a function's type index is declared
        // along with the function, but in the binary `wasm`
        // it's in its own section; the `code` section
        // contains the code and the `function` section
        // contains signature indices.  We join them back
        // together here.
        if let (Some(code), Some(functions)) = (module.code_section(), module.function_section()) {
            assert_eq!(code.bodies().len(), functions.entries().len());
            // Evade double-borrow of m here.
            let types = &m.types;
            let converted_funcs = code.bodies().iter()
                .zip(functions.entries())
                .map(|(c, f)| {
                    // Make sure the function signature is a valid type.
                    let type_idx = f.type_ref() as usize;
                    assert!(type_idx < types.len(), "Function refers to a type signature that does not exist!");
                    
                    Func {
                        typeidx: TypeIdx(type_idx),
                        locals: VariableSlot::from_locals(c.locals()),
                        body: c.code().elements().to_owned(),
                    }
                });
            m.funcs.extend(converted_funcs);
        } else {
            panic!("Code section exists but type section does not, or vice versa!");
        }

        // Allocate tables
        if let Some(table) = module.table_section() {
            println!("Table: {:?}", table);
            // currently we can only have one table section with
            // 0 or 1 elements in it, so.
            assert!(table.entries().len() < 2, "More than one memory entry, should never happen!");
            if let Some(table) = table.entries().iter().next() {
                // TODO: As far as I can tell, the memory's minimum size is never used?
                let _min = table.limits().initial();
                let max = table.limits().maximum();

                // TODO: It's apparently valid for a memory to have no max size?
                if let Some(max) = max {
                    m.tables.fill(max);
                }
            }

            if let Some(elements) = module.elements_section() {
                // TODO
                unimplemented!();
            }
        }

        // Allocate memories
        if let Some(memory) = module.memory_section() {
            // currently we can only have one memory section with
            // 0 or 1 elements in it, so.
            assert!(memory.entries().len() < 2, "More than one memory entry, should never happen!");
            if let Some(memory) = memory.entries().iter().next() {
                // TODO: As far as I can tell, the memory's minimum size is never used?
                let _min = memory.limits().initial();
                let max = memory.limits().maximum();

                // TODO: It's apparently valid for a memory to have no max size?
                if let Some(max) = max {
                    m.mem.fill(max);
                }
            }
                 
            if let Some(data) = module.data_section() {
                // TODO
                unimplemented!();
            }
        }


        // Allocate imports
        if let Some(imports) = module.import_section() {
            println!("Imports: {:?}", imports);
        }

        // Allocate exports
        if let Some(exports) = module.export_section() {
            println!("Exports: {:?}", exports);
        }

        // Check for start section
        if let Some(start) = module.start_section() {
            let start = start as usize;
            assert!(start < m.funcs.len(), "Start section references a non-existent function!");
            m.start = Some(start);
        }

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

    /// Make sure that adding a non-validated module to a program fails.
    #[test]
    #[should_panic]
    fn test_validate_failure() {
        let input_file = "test_programs/inc.wasm";
        let module = parity_wasm::deserialize_file(input_file).unwrap();
        let mut mod_instance = ModuleInstance::new(module);
        let program = Program::new()
            .with_module("inc", mod_instance);
    }
    
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
