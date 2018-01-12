//! An attempt at creating an embeddable wasm interpreter that's
//! bloody easier to use and debug than the `parity-wasm` one.

extern crate parity_wasm;

use parity_wasm::elements;

use std::collections::HashMap;

#[cfg(test)]
mod tests;


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

    /// Get the type of the value.
    fn get_type(self) -> elements::ValueType {
        match self {
            Value::I32(_) => elements::ValueType::I32,
            Value::I64(_) => elements::ValueType::I64,
            Value::F32(_) => elements::ValueType::F32,
            Value::F64(_) => elements::ValueType::F64,
        }
    }
}

// parity-wasm is hard to understand but does have some
// pretty nice ideas.

impl From<i32> for Value {
    fn from(num: i32) -> Self {
        Value::I32(num)
    }
}

impl From<i64> for Value {
    fn from(num: i64) -> Self {
        Value::I64(num)
    }
}

impl From<f32> for Value {
    fn from(num: f32) -> Self {
        Value::F32(num)
    }
}

impl From<f64> for Value {
    fn from(num: f64) -> Self {
        Value::F64(num)
    }
}



// /// Memory for a variable.
// #[derive(Debug, Copy, Clone, PartialEq)]
// pub struct VariableSlot {
//     variable_type: elements::ValueType,
//     value: Value,
// }

// impl VariableSlot {
//     /// Takes a slice of `Local`'s (local variable *specifications*),
//     /// and creates a vec of `VariableSlot`'s matching them.
//     fn from_locals(locals: &[elements::Local]) -> Vec<VariableSlot> {
//         let num_local_slots = locals.iter()
//             .map(|x| x.count() as usize)
//             .sum();
//         let mut v = Vec::with_capacity(num_local_slots);
//         for local in locals {
//             for i in 0..local.count() {
//                 let slot = VariableSlot {
//                     variable_type: local.value_type(),
//                     value: Value::default_from_type(local.value_type()),
//                 };
//                 v.push(slot);
//             }
//         }
//         v
//     }
// }

/// Takes a slice of `Local`'s (local variable *specifications*),
/// and creates a vec of their types.
fn types_from_locals(locals: &[elements::Local]) -> Vec<elements::ValueType> {
    let num_local_slots = locals.iter()
        .map(|x| x.count() as usize)
        .sum();
    let mut v = Vec::with_capacity(num_local_slots);
    for local in locals {
        for i in 0..local.count() {
            let t = local.value_type();
            v.push(t);
        }
    }
    v
}

/// An index into a module's `function` vector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FuncIdx(usize);

/// A function ready to be executed.
#[derive(Debug, Clone)]
pub struct Func {
    typeidx: TypeIdx,
    locals: Vec<elements::ValueType>,
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
                        locals: types_from_locals(c.locals()),
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


/// The activation record for an executing function.
#[derive(Debug, Clone, Default)]
pub struct StackFrame {
    value_stack:  Vec<Value>,
    labels: Vec<usize>,
    locals: Vec<Value>,
    /// Where in the current function execution is.
    ip: usize,
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
            ip: 0,
        }
    }

    fn from_func_instance(func: &FuncInstance, args: &[Value]) -> Self {
        // Allocate space for locals+params
        let mut locals = Vec::with_capacity(func.locals.len() + func.functype.params.len());
        assert_eq!(func.functype.params.len(), args.len(), "Tried to create stack frame for func with different number of parameters than the type says it takes!");

        // Push params
        locals.extend(args.into_iter());
        // Fill remaining space with 0's
        let iter = func.functype.params.iter()
            .map(|t| Value::default_from_type(*t));
        locals.extend(iter);

        Self {
            value_stack: vec![],
            labels: vec![],
            locals: locals,
            ip: 0,
        }
    }

    /// Get a local variable in the stack frame by index.
    /// Panics if out of bounds.
    fn get_local(&mut self, idx: usize) -> Value {
        assert!(idx < self.locals.len());
        self.locals[idx]
    }

    /// Set a local variable in the stack frame by index.
    /// Panics if out of bounds or if the type of the new
    /// variable does not match the old one(?).
    fn set_local(&mut self, idx: usize, vl: Value) {
        assert!(idx < self.locals.len());
        assert_eq!(self.locals[idx].get_type(), vl.get_type());
        self.locals[idx] = vl;
    }

    /// Pop the top of the value_stack and returns the value.
    ///
    /// Panics if the stack is empty.
    fn pop(&mut self) -> Value {
        assert!(!self.value_stack.is_empty());
        self.value_stack.pop()
            .unwrap()
    }

    /// Pushes the given value to the top of the value_stack.
    /// Basically just for symmetry with `pop()`.
    fn push(&mut self, vl: Value) {
        self.value_stack.push(vl)
    }

    /// Returns the value from the top of the value_stack
    /// without altering the stack.
    ///
    /// Panics if the stack is empty.
    fn peek(&self) -> Value {
        assert!(!self.value_stack.is_empty());
        *self.value_stack.last()
            .unwrap()
    }
}


/// Function address type; refers to a particular `FuncInstance` in the Store.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FunctionAddress(usize);
/// Table address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TableAddress(usize);
/// Memory address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MemoryAddress(usize);
/// Global address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GlobalAddress(usize);
/// Module instance address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleAddress(usize);


/// Contains all the information needed to execute a function.
#[derive(Debug, Clone)]
pub struct FuncInstance {
    functype: FuncType,
    locals: Vec<elements::ValueType>,
    body: Vec<elements::Opcode>,
    module: ModuleAddress,
}

/// Relates all the local indices to globals, functions etc.
/// from within a module to the global addresses of the Store.
#[derive(Debug, Clone)]
pub struct ModuleInstance {
    functions: Vec<FunctionAddress>,
    table: Option<TableAddress>,
    memory: Option<MemoryAddress>,
    globals: Vec<GlobalAddress>,
    // TODO: Start function!
}

/// All the *mutable* parts of the interpreter state.
/// This slightly wacky structure helps keep borrows from
/// being awful, a little bit.
///
/// Also see: `State`.
#[derive(Debug, Clone, Default)]
pub struct Store {
    stack: Vec<StackFrame>,
    tables: Vec<Table>,
    mems: Vec<Memory>,
    globals: Vec<Global>,
}

/// All the *immutable* parts of the interpreter state.
///
/// Also see: `Store`.
#[derive(Debug, Clone, Default)]
pub struct State {
    funcs: Vec<FuncInstance>,
    module_instances: Vec<ModuleInstance>,
    modules: HashMap<String, LoadedModule>,
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
    store: Store,
    state: State,
}


impl Interpreter {
    fn new() -> Self {
        Self {
            store: Store::default(),
            state: State::default(),
        }
    }

    /// Returns a GlobalAddress from a given index
    fn resolve_global(state: &State, module_addr: ModuleAddress, idx: usize) -> GlobalAddress {
        assert!(module_addr.0 < state.module_instances.len());
        let module_instance = &state.module_instances[module_addr.0];
        assert!(idx < module_instance.globals.len());
        module_instance.globals[idx]
    }
    
    /// Get a global variable by *index*.  Needs a module instance
    /// address to look up the global variable's address.
    /// Panics if out of bounds.
    ///
    /// This is unused since it creates irritating double-borrows.
    fn get_global(globals: &[Global], state: &State, module_addr: ModuleAddress, idx: usize) -> Value {
        let global_addr = Interpreter::resolve_global(state, module_addr, idx);
        globals[global_addr.0].value
    }
    /// Sets a global variable by *index*.  Needs a module instance
    /// address to look up the global variable's address.
    /// Panics if out of bounds or if the type of the new
    /// variable does not match the old one(?).
    fn set_global(globals: &mut [Global], state: &State, module_addr: ModuleAddress, idx: usize, vl: Value) {
        let global_addr = Interpreter::resolve_global(state, module_addr, idx);
        assert!(globals[global_addr.0].mutable);
        assert_eq!(globals[global_addr.0].variable_type, vl.get_type());
        globals[global_addr.0].value = vl;
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
        let module_instance_address = ModuleAddress(self.state.module_instances.len());
        let mut functions = vec![];
        let table = None;
        let memory = None;
        let globals = vec![];
        for func in module.funcs.iter() {
            let address = FunctionAddress(self.state.funcs.len());
            let functype = module.types[func.typeidx.0].clone(); 
            let instance = FuncInstance {
                functype: functype,
                locals: func.locals.clone(),
                body: func.body.clone(),
                module: module_instance_address,
            };
            println!("Function: {:?}", instance);
            self.state.funcs.push(instance);
            functions.push(address);
        }
        let inst = ModuleInstance {
            functions,
            table,
            memory,
            globals,
        };
        self.state.modules.insert(module.name.to_owned(), module);
        self.state.module_instances.push(inst);
        self
    }

    fn trap() {
        panic!("Trap occured!  Aieee!")
    }

    /*
    /// Takes a loaded module, pulls it apart, and shoves all its
    /// parts into the interpreter's Store.  Produces a ModuleInstance
    /// which lets you translate indices referring to module resources
    /// into addresses referring to Store resources.
    fn instantiate(&mut self, module: &LoadedModule) {

    }
*/


    // fn run_module_function(&mut self, module: &str, func: FuncIdx, args: &[Value]) {
    //     // let function = self.funcs.get(func.0)
    //     //     .expect("Invalid function address, should never happen");
    //     let function = &self.modules[module].funcs[1];
    //     let func_type = &self.modules[module].types[function.typeidx.0];
    //     let frame = StackFrame::from_func(function, &func_type, args);
    //     println!("Frame is {:?}", frame);
    //     self.stack.push(frame);
    //     for op in &function.body {
    //         println!("Op is {:?}", op);
    //     }
    //     self.stack.pop();
    // }

    /// Actually do the interpretation of the given function, assuming
    /// that a stack frame already exists for it with args and locals 
    /// and such
    fn exec(&mut self, func: FunctionAddress) {
        let func = &self.state.funcs[func.0];
        let frame = self.store.stack.last_mut().expect("No stack frame, should be impossible");
        use elements::Opcode::*;
        use std::usize;
        loop {
            let op = &func.body[frame.ip];

            println!("Op: {:?}", op);
            match *op {
                Unreachable => unreachable!(),
                Nop => (),
                Block(blocktype) => (),
                Loop(blocktype) => (),
                If(blocktype) => (),
                Else => (),
                End => break,
                Br(i) => (),
                BrIf(i) => (),
                BrTable(ref v, i) => (),
                Return => (),
                Call(i) => (), //Interpreter::exec_call(self, i),
                CallIndirect(i, b) => (),
                Drop => {
                    frame.pop();
                },
                Select => (),
                GetLocal(i) => {
                    let i = i as usize;
                    let vl = frame.get_local(i as usize);
                    frame.value_stack.push(vl);
                },
                SetLocal(i) => {
                    let i = i as usize;
                    let vl = frame.pop();
                    frame.set_local(i, vl);
                },
                TeeLocal(i) => {
                    let i = i as usize;
                    let vl = frame.peek();
                    frame.set_local(i, vl);
                },
                GetGlobal(i) => {
                    let i = i as usize;
                    let vl = Interpreter::get_global(&self.store.globals, &self.state, func.module, i);
                    frame.push(vl);
                },
                SetGlobal(i) => {
                    let i = i as usize;
                    let vl = frame.pop();
                    Interpreter::set_global(&mut self.store.globals, &self.state, func.module, i, vl);
                },
                I32Load(i1, i2) => (),
                I64Load(i1, i2) => (),
                F32Load(i1, i2) => (),
                F64Load(i1, i2) => (),
                I32Load8S(i1, i2) => (),
                I32Load8U(i1, i2) => (),
                I32Load16S(i1, i2) => (),
                I32Load16U(i1, i2) => (),
                I64Load8S(i1, i2) => (),
                I64Load8U(i1, i2) => (),
                I64Load16S(i1, i2) => (),
                I64Load16U(i1, i2) => (),
                I64Load32S(i1, i2) => (),
                I64Load32U(i1, i2) => (),
                I32Store(i1, i2) => (),
                I64Store(i1, i2) => (),
                F32Store(i1, i2) => (),
                F64Store(i1, i2) => (),
                I32Store8(i1, i2) => (),
                I32Store16(i1, i2) => (),
                I64Store8(i1, i2) => (),
                I64Store16(i1, i2) => (),
                I64Store32(i1, i2) => (),
                CurrentMemory(b) => (),
                GrowMemory(b) => (),
                I32Const(i) => Interpreter::exec_const(frame, i.into()),
                I64Const(l) => Interpreter::exec_const(frame, l.into()),
                // Why oh why are these represented as u32 and u64?
                // Because this is the serialized representation, sigh.
                // TODO: Fix this somehow so we don't have to keep encoding/
                // decoding floats but just check them once?
                // Even though from_bits() should be basically free...
                // BUGGO: This is technically incorrect because a signaling NaN
                // *may* slip through from_bits(), and WebAssembly currently
                // does not support signaling NaN's.
                // See https://webassembly.github.io/spec/core/exec/numerics.html#floating-point-operations
                F32Const(i) => {
                    use std::f32;
                    Interpreter::exec_const(frame, Value::from(f32::from_bits(i)))
                }
                F64Const(l) => {
                    use std::f64;
                    Interpreter::exec_const(frame, Value::from(f64::from_bits(l)))
                },
                I32Eqz => (),
                I32Eq => (),
                I32Ne => (),
                I32LtS => (),
                I32LtU => (),
                I32GtS => (),
                I32GtU => (),
                I32LeS => (),
                I32LeU => (),
                I32GeS => (),
                I32GeU => (),
                I64Eqz => (),
                I64Eq => (),
                I64Ne => (),
                I64LtS => (),
                I64LtU => (),
                I64GtS => (),
                I64GtU => (),
                I64LeS => (),
                I64LeU => (),
                I64GeS => (),
                I64GeU => (),
                F32Eq => (),
                F32Ne => (),
                F32Lt => (),
                F32Gt => (),
                F32Le => (),
                F32Ge => (),
                F64Eq => (),
                F64Ne => (),
                F64Lt => (),
                F64Gt => (),
                F64Le => (),
                F64Ge => (),
                I32Clz => (),
                I32Ctz => (),
                I32Popcnt => (),
                I32Add => (),
                I32Sub => (),
                I32Mul => (),
                I32DivS => (),
                I32DivU => (),
                I32RemS => (),
                I32RemU => (),
                I32And => (),
                I32Or => (),
                I32Xor => (),
                I32Shl => (),
                I32ShrS => (),
                I32ShrU => (),
                I32Rotl => (),
                I32Rotr => (),
                I64Clz => (),
                I64Ctz => (),
                I64Popcnt => (),
                I64Add => (),
                I64Sub => (),
                I64Mul => (),
                I64DivS => (),
                I64DivU => (),
                I64RemS => (),
                I64RemU => (),
                I64And => (),
                I64Or => (),
                I64Xor => (),
                I64Shl => (),
                I64ShrS => (),
                I64ShrU => (),
                I64Rotl => (),
                I64Rotr => (),
                F32Abs => (),
                F32Neg => (),
                F32Ceil => (),
                F32Floor => (),
                F32Trunc => (),
                F32Nearest => (),
                F32Sqrt => (),
                F32Add => (),
                F32Sub => (),
                F32Mul => (),
                F32Div => (),
                F32Min => (),
                F32Max => (),
                F32Copysign => (),
                F64Abs => (),
                F64Neg => (),
                F64Ceil => (),
                F64Floor => (),
                F64Trunc => (),
                F64Nearest => (),
                F64Sqrt => (),
                F64Add => (),
                F64Sub => (),
                F64Mul => (),
                F64Div => (),
                F64Min => (),
                F64Max => (),
                F64Copysign => (),
                I32WarpI64 => (),
                I32TruncSF32 => (),
                I32TruncUF32 => (),
                I32TruncSF64 => (),
                I32TruncUF64 => (),
                I64ExtendSI32 => (),
                I64ExtendUI32 => (),
                I64TruncSF32 => (),
                I64TruncUF32 => (),
                I64TruncSF64 => (),
                I64TruncUF64 => (),
                F32ConvertSI32 => (),
                F32ConvertUI32 => (),
                F32ConvertSI64 => (),
                F32ConvertUI64 => (),
                F32DemoteF64 => (),
                F64ConvertSI32 => (),
                F64ConvertUI32 => (),
                F64ConvertSI64 => (),
                F64ConvertUI64 => (),
                F64PromoteF32 => (),
                I32ReinterpretF32 => (),
                I64ReinterpretF64 => (),
                F32ReinterpretI32 => (),
                F64ReinterpretI64 => (),
            }
            frame.ip += 1;
        }
    }

    fn run_function(&mut self, func: FunctionAddress, args: &[Value]) {
        let frame = StackFrame::from_func_instance(&self.state.funcs[func.0], args);
        println!("Frame is {:?}", frame);

        self.store.stack.push(frame);
        self.exec(func);
        self.store.stack.pop();
    }

    fn exec_const(frame: &mut StackFrame, vl: Value) {
        frame.value_stack.push(vl);
    }

    fn exec_call(&mut self, i: u32) {
    }

}

