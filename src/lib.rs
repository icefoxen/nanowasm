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
    mutable: bool,
    variable_type: elements::ValueType,
    value: Value,
    init_code: Vec<elements::Opcode>,
}

/// A loaded wasm module
#[derive(Debug, Clone)]
pub struct LoadedModule {
    /// Module name.  Not technically necessary, but handy.
    name: String,
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

impl LoadedModule {
    /// Instantiates and initializes a new module.
    /// Does NOT validate or run the start function though!
    fn new(name: &str, module: elements::Module) -> Self {
        assert_eq!(module.version(), 1);

        let mut m = Self {
            name: name.to_owned(),
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

        // Allocate globals
        if let Some(globals) = module.global_section() {
            let global_iter = globals.entries().iter()
                .map(|global| {
                let global_type = global.global_type().content_type();
                let mutability = global.global_type().is_mutable();
                let init_code = Vec::from(global.init_expr().code());
                Global {
                    variable_type: global_type,
                    mutable: mutability,
                    value: Value::default_from_type(global_type),
                    // TODO: The init_code must be zero or more `const` instructions
                    // followed by `end`
                    // See https://webassembly.github.io/spec/core/syntax/modules.html#syntax-global
                    init_code: init_code,
                }
            });
            m.globals.extend(global_iter);
        }

        // Allocate imports
        if let Some(imports) = module.import_section() {
            println!("Imports: {:?}", imports);
            unimplemented!();
        }

        // Allocate exports
        if let Some(exports) = module.export_section() {
            println!("Exports: {:?}", exports);
            unimplemented!();
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
// #[derive(Debug, Clone)]
// pub struct Program {
//     modules: Vec<LoadedModule>,
// }

// impl Program {
//     fn new() -> Self {
//         Self {
//             modules: Vec::new(),
//         }
//     }

//     /// Builder function to add a loaded and validated module to the
//     /// program.
//     ///
//     /// Essentially, this does the dynamic linking, and should cause
//     /// errors to happen if there are invalid/dangling references.
//     /// So, you have to load all the modules in order of dependencies.
//     ///
//     /// We could load all the modules in arbitrary order, then validate+link
//     /// them at the end, but meh.
//     fn with_module(mut self, module: LoadedModule) -> Self {
//         assert!(module.validated);
//         self.modules.push(module);
//         self
//     }
// }


/// The context for an executing function.
#[derive(Debug, Clone, Default)]
pub struct StackFrame {
    value_stack:  Vec<Value>,
    labels: Vec<usize>,
    locals: Vec<Value>,
    // This IS an activation record so we don't need
    // to store those separately.
}

impl StackFrame {
    /// Takes a Func and allocates a stack frame for it, then pushes
    /// the given args to its locals.
    fn from_func(func: &Func, functype: &FuncType, args: &[Value]) -> Self {
        // Allocate space for locals+params
        let mut locals = Vec::with_capacity(func.locals.len() + functype.params.len());
        assert_eq!(functype.params.len(), args.len(), "Tried to create stack frame for func with different number of parameters than the type says it takes!");

        // Push params
        locals.extend(args.into_iter());
        // Fill remaining space with 0's
        let iter = functype.params.iter()
            .map(|t| Value::default_from_type(*t));
        locals.extend(iter);

        Self {
            value_stack: vec![],
            labels: vec![],
            locals: locals,
        }
    }
}
#[derive(Debug, Clone)]
pub struct ModuleInstance {

}

/// An interpreter which runs a particular program.
///
/// Per the wasm spec, this contains the **Store**; you can
/// have a validated program that is ready to run, but this
/// has all the runtime state and such from it.
///
/// The WASM spec has a not-immediately-obvious gap in semantics
/// between the environment in which programs are defined, loaded
/// and validated, where all references are *purely module-local*,
/// and the environment in which programs are executed, where all
/// references are *global*; modules are loaded and all their resources
/// are just shoved
/// into the Store.  It distinguishes these environments by using the
/// term "index" to mean an offset into a module-local environment,
/// and "address" to mean an offset into a global environment.
/// See <https://webassembly.github.io/spec/core/exec/runtime.html>
///
/// A module then becomes a **module instance** when ready to execute,
/// which ceases to be a collection of data and becomes a collection
/// of index-to-address mappings.  A **function instance** then is 
/// the original function definition, plus the a reference to the
/// module instance to allow it to resolve its indices to addresses.
#[derive(Debug, Clone)]
pub struct Interpreter {
    stack: Vec<StackFrame>,
    funcs: Vec<Func>,
    tables: Vec<Table>,
    mems: Vec<Memory>,
    globals: Vec<Global>,
    module_instances: Vec<ModuleInstance>,
    modules: HashMap<String, LoadedModule>,
}


/// Function address types
struct FunctionAddress(usize);
struct TableAddress(usize);
struct MemoryAddress(usize);
struct GlobalAddress(usize);

impl Interpreter {
    fn new() -> Self {
        Self {
            stack: vec![],
            funcs: vec![],
            tables: vec![],
            mems: vec![],
            globals: vec![],
            module_instances: vec![],
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
    fn with_module(mut self, module: LoadedModule
) -> Self {
        assert!(module.validated);
        self.modules.insert(module.name.to_owned(), module);
        self
    }

    fn trap() {
        panic!("Trap occured!  Aieee!")
    }

    /// Takes a loaded module, pulls it apart, and shoves all its
    /// parts into the interpreter's Store.  Produces a ModuleInstance
    /// which lets you translate indices referring to module resources
    /// into addresses referring to Store resources.
    fn instantiate(&mut self, module: &LoadedModule) {

    }


    fn run_module_function(&mut self, module: &str, func: FuncIdx, args: &[Value]) {
        // let function = self.funcs.get(func.0)
        //     .expect("Invalid function address, should never happen");
        let function = &self.modules[module].funcs[1];
        let func_type = &self.modules[module].types[function.typeidx.0];
        let frame = StackFrame::from_func(function, &func_type, args);
        println!("Frame is {:?}", frame);
        self.stack.push(frame);
        for op in &function.body {
            println!("Op is {:?}", op);
        }
        self.stack.pop();
    }

    fn run_function(&mut self, func: FunctionAddress, args: &[Value]) {
        // let function = self.funcs.get(func.0)
        //     .expect("Invalid function address, should never happen");
        let function = &self.funcs[0];

        for op in &function.body {
            println!("Op is {:?}", op);
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
        let mut mod_instance = LoadedModule
    ::new("inc", module);
        let interp = Interpreter::new()
            .with_module(mod_instance);
    }
    
    #[test]
    fn test_create() {
        let input_file = "test_programs/inc.wasm";
        let module = parity_wasm::deserialize_file(input_file).unwrap();
        let mut mod_instance = LoadedModule
    ::new("inc", module);
        mod_instance.validate();
        let interp = Interpreter::new()
            .with_module(mod_instance);
        println!("{:#?}", interp);
        // assert!(false);
    }

    #[test]
    fn test_create_fib() {
        let input_file = "test_programs/fib.wasm";
        let module = parity_wasm::deserialize_file(input_file).unwrap();
        let mut mod_instance = LoadedModule
    ::new("fib", module);
        mod_instance.validate();
        let interp = Interpreter::new()
            .with_module(mod_instance);
        println!("{:#?}", interp);
        // assert!(false);
    }

    #[test]
    fn test_run_fib() {
        let input_file = "test_programs/fib.wasm";
        let module = parity_wasm::deserialize_file(input_file).unwrap();
        let mut mod_instance = LoadedModule
    ::new("fib", module);
        mod_instance.validate();
        let mut interp = Interpreter::new()
            .with_module(mod_instance);
            
        interp.run_module_function("fib", FuncIdx(1), &vec![Value::I32(30)]);
        assert!(false);
    }

}
