//! The actual run-time interpreter.

use std;
use std::collections::HashMap;

use byteorder;
use byteorder::ByteOrder;
use parity_wasm::elements;

use types::*;
use util::*;
use loader::*;

/// A label pointing to a particular block: just an instruction index.
/// The label index is implicit in the labels stack; label 0 is always
/// the **top** of the stack.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BlockLabel(usize);

/// The activation record for an executing function.
#[derive(Debug, Clone, Default)]
struct StackFrame {
    /// Value stack; "the stack".
    value_stack: Vec<Value>,
    /// Current label stack.
    labels: Vec<BlockLabel>,
    /// Local variables.
    locals: Vec<Value>,
    /// Where in the current function execution is.
    ip: usize,
}

impl StackFrame {
    /// Takes a FuncInstance and allocates a stack frame for it, then pushes
    /// the given args to its locals.
    fn from_func_instance(func: &FuncInstance, args: &[Value]) -> Self {
        // Allocate space for locals+params
        let locals_size = func.locals.len() + func.functype.params.len();
        let mut locals = Vec::with_capacity(locals_size);
        assert_eq!(func.functype.params.len(), args.len(), "Tried to create stack frame for func with different number of parameters than the type says it takes!");

        // Push params
        locals.extend(args.into_iter());
        // Fill remaining space with 0's
        let iter = func.functype
            .params
            .iter()
            .map(|t| Value::default_from_type(*t));
        locals.extend(iter);

        Self {
            value_stack: Vec::with_capacity(8),
            labels: Vec::with_capacity(8),
            locals: locals,
            ip: 0,
        }
    }

    /// Push a new BlockLabel to the label stack.
    fn push_label(&mut self, ip: BlockLabel) {
        self.labels.push(ip);
    }

    /// Pops to the given label index and returns
    /// the BlockLabel of the destination instruction index.
    /// Passing it 0 jumps to the first containing label, etc.
    ///
    /// Panics if an invalid/too large index is given.
    fn pop_label(&mut self, label_idx: usize) -> BlockLabel {
        let i = 0;
        while i < label_idx {
            self.labels.pop();
        }
        self.labels.pop().unwrap()
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
        self.value_stack.pop().unwrap()
    }

    /// Pops the top of the value_stack and returns the value as a number.
    ///
    /// Panics if the stack is empty or the Value is not the right
    /// numeric type.
    fn pop_as<T>(&mut self) -> T
    where
        T: From<Value>,
    {
        self.pop().into()
    }

    /// Pops the top two values of the value_stack and returns them
    /// cast into the given types.
    ///
    /// The top of the stack is the second value returned, the first
    /// is one down from the top.
    ///
    /// Panics if the stack is empty or either of the Value's is not the right
    /// numeric type.
    fn pop2_as<T1, T2>(&mut self) -> (T1, T2)
    where
        T1: From<Value>,
        T2: From<Value>,
    {
        let a = self.pop().into();
        let b = self.pop().into();
        (b, a)
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
        *self.value_stack.last().unwrap()
    }
}


/// Function address type; refers to a particular `FuncInstance` in the Store.
/// All `*Address` types are indices into the runtime `Store`; contrast with `*Idx`
/// types which are indices into a particular module's local namespace.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FunctionAddress(pub usize);
/// Table address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TableAddress(pub usize);
/// Memory address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryAddress(pub usize);
/// Global address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalAddress(pub usize);
/// Module instance address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleAddress(pub usize);

/// For forward jumps (if, block) we need to know where to jump TO.
/// Serialized wasm doesn't store this information explicitly,
/// and searching for it mid-execution is a wasteful PITA,
/// so we find it ahead of time and then store it when the
/// function is instantiated.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct JumpTarget {
    block_start_instruction: usize,
    block_end_instruction: usize,
    /// Only used for if/else statements.
    else_instruction: usize,
}

/// Contains all the runtime information needed to execute a function
#[derive(Debug, Clone)]
struct FuncInstance {
    functype: FuncType,
    locals: Vec<elements::ValueType>,
    body: FuncBody,
    module: ModuleAddress,
    /// A vec of jump targets sorted by source instruction,
    /// so we can just binary-search in it.  A HashMap would
    /// work too, but I suspect this is faster?  And is trivial
    /// to construct, so.
    jump_table: Vec<JumpTarget>,
}

impl FuncInstance {
    /// Iterate through a function's body and construct the jump table for it.
    /// If we find a block or if instruction, the target is the matching end instruction.
    ///
    /// Panics on invalid (improperly nested) blocks.
    fn compute_jump_table(body: &FuncBody) -> Vec<JumpTarget> {
        match *body {
            FuncBody::Opcodes(ref opcodes) => {
                use parity_wasm::elements::Opcode::*;
                // TODO: I would be sort of happier recursively walking a sequence, OCaml
                // style, but oh well.
                let mut offset = 0;
                let mut accm = vec![];
                while offset < opcodes.len() {
                    let op = &opcodes[offset];
                    // println!("Computing jump table: {}, {:?}", offset, op);
                    match *op {
                        Block(_) | If(_) => {
                            offset = FuncInstance::find_block_close(opcodes, offset, &mut accm);
                        }
                        _ => (),
                    }
                    offset += 1;
                }
                accm
            }
            FuncBody::HostFunction(_) => vec![],
        }
    }

    /// Recursively walk through opcodes starting from the given offset, and
    /// accumulate jump targets into the given vec.  This way we only have
    /// to walk the function once.
    ///
    /// Returns the last instruction index of the block, so you can start
    /// there and go on to find the next block.
    fn find_block_close(
        body: &[elements::Opcode],
        start_offset: usize,
        accm: &mut Vec<JumpTarget>,
    ) -> usize {
        use std::usize;
        use parity_wasm::elements::Opcode::*;
        let mut offset = start_offset;
        // TODO: Potentially invalid here, but, okay.
        let mut else_offset = usize::MAX;
        loop {
            let op = &body[offset];
            match *op {
                End => {
                    // Found matching end, yay.
                    let jt = JumpTarget {
                        block_start_instruction: start_offset,
                        block_end_instruction: offset,
                        else_instruction: else_offset,
                    };
                    accm.push(jt);
                    return offset;
                }
                // TODO: Only valid within If blocks,
                // but we don't check for that...
                // Easy to do just by having an argument to this function
                // that says whether or not we're in an If, and panic if we
                // find an invalid Else.
                Else => {
                    else_offset = offset;
                }
                // Opening another block, recurse.
                // BUGGO TODO: This stack overflows, figure out why.
                /*
                If(_) => {
                    println!("IF happening!");
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();
                    offset = FuncInstance::find_block_close(body, offset, accm);
                },
*/
                Block(_)=> {
                    offset = FuncInstance::find_block_close(body, offset, accm);
                }
                _ => (),
            }
            offset += 1;
            assert!(offset < body.len(), "Unclosed block, should never happen!");
        }
    }
}

/// While a LoadedModule contains the specification of a module
/// in a convenient form, this is a runtime structure that contains
/// the relationship between module-local indices and global addresses.
/// So it relates its own local address space to the address space of the
/// `Store`
///
/// It also contains a bit of module-local data, mainly type vectors, that
/// don't need to be in the Store since they're never communicated between
/// modules.
#[derive(Debug, Clone)]
struct ModuleInstance {
    name: String,
    // These might be somewhat redundant with the
    // LoadedModule's
    exported_functions: Vec<Export<FuncIdx>>,
    exported_tables: Option<Export<()>>,
    exported_memories: Option<Export<()>>,
    exported_globals: Vec<Export<GlobalIdx>>,

    types: Vec<FuncType>,
    functions: Vec<FunctionAddress>,
    table: Option<TableAddress>,
    memory: Option<MemoryAddress>,
    globals: Vec<GlobalAddress>,
    start: Option<FunctionAddress>,
}

impl ModuleInstance {
    /// Takes a loaded-but-not-instantiated module and a slice of other modules
    /// loaded before it, and checks to see whether the module's imports are
    /// all provided by other modules.
    fn resolve_imports(
        &mut self,
        module: &LoadedModule,
        other_modules: &[ModuleInstance],
    ) -> Result<(), Error> {
        // Breaking imports/exports apart into separate arrays by type makes
        // life somewhat easier; instead of having a big for loop that checks
        // whether it exists and whether the types match, we just have to check
        // for existence in the appropriate array.
        // TODO: Validate the External memory/table/function/etc junk more

        // Turns out you CAN create closures that capture a function's environment and return them.
        // oy.
        fn generate_not_found_error<'a, T>(import: &'a Import<T>, module: &'a LoadedModule) -> Box<Fn() -> Error + 'a> {
            Box::new(move || Error::ModuleNotFound {
                module: import.module_name.clone(),
                dependent_module: module.name.clone()
            })
        }

        // Same as previous function, just a different error type.
        fn generate_not_exported_error<'a>(name: &'a str, import: &'a str, dependent_module: &'a str, typ: &'a str) -> Box<Fn() -> Error + 'a> {
            Box::new(move || Error::NotExported {
                name: name.to_owned(),
                module: import.to_owned(),
                dependent_module: dependent_module.to_owned(),
                typ: typ.to_owned(),
            })
        }
        
        for import in &module.imported_functions {
            let target_module = other_modules
                .iter()
                .find(|m| import.module_name == m.name)
                .ok_or_else(&*generate_not_found_error(import, module))?;

            let export_idx = target_module
                .exported_functions
                .iter()
                .position(|e| e.name == import.field_name)
                .ok_or_else(&*generate_not_exported_error(
                    &target_module.name,
                    &import.field_name,
                    "function",
                    &module.name
                ))?;

            // TODO: Assert that the import and export types match

            let addr = target_module.functions[export_idx];
            self.functions.push(addr);
        }

        for import in &module.imported_tables {
            let target_module = other_modules
                .iter()
                .find(|m| import.module_name == m.name)
                .ok_or_else(&*generate_not_found_error(import, module))?;

            let export = target_module
                .exported_tables
                .iter()
                .find(|e| e.name == import.field_name)
                .ok_or_else(&*generate_not_exported_error(
                    &target_module.name,
                    &import.field_name,
                    "table",
                    &module.name
                ))?;

            // TODO: The "unwrap" here and for Memory
            // forms our ghetto error-checking;
            // since we can only have one memory or table,
            // the index is irrelevant.
            let addr = target_module.table.unwrap();
            self.table = Some(addr);
        }

        for import in &module.imported_memories {
            let target_module = other_modules
                .iter()
                .find(|m| import.module_name == m.name)
                .ok_or_else(&*generate_not_found_error(import, module))?;

            let export = target_module
                .exported_memories
                .iter()
                .find(|e| e.name == import.field_name)
                .ok_or_else(&*generate_not_exported_error(
                    &target_module.name,
                    &import.field_name,
                    "memory",
                    &module.name
                ))?;

            // TODO: The "unwrap" here and for Memory
            // forms our ghetto error-checking;
            // since we can only have one memory or table,
            // the index is irrelevant.
            let addr = target_module.memory.unwrap();
            self.memory = Some(addr);
        }

        for import in &module.imported_globals {
            let target_module = other_modules
                .iter()
                .find(|m| import.module_name == m.name)
                .ok_or_else(&*generate_not_found_error(import, module))?;

            let export = target_module
                .exported_globals
                .iter()
                .find(|e| e.name == import.field_name)
                .ok_or_else(&*generate_not_exported_error(
                    &target_module.name,
                    &import.field_name,
                    "global",
                    &module.name
                ))?;

            let addr = target_module.globals[export.value.0];
            self.globals.push(addr);
        }

        self.exported_functions = module.exported_functions.clone();
        self.exported_tables = module.exported_tables.clone();
        self.exported_memories = module.exported_memories.clone();
        self.exported_globals = module.exported_globals.clone();

        Ok(())
    }
}

/// All the *mutable* parts of the interpreter state.
/// This slightly wacky structure helps keep borrows from
/// being awful, a little bit.
///
/// Also see: `State`.
#[derive(Debug, Clone, Default)]
pub struct Store {
    tables: Vec<Table>,
    mems: Vec<Memory>,
    globals: Vec<Global>,
    // We don't have explicit StackFrame's in the Store for Reasons.
    // Borrowing reasons.  Namely, a function needs
    // a mut reference to its StackFrame, naturally.
    // but it also has to be able to push new StackFrame's
    // to the stack when a new function is called, and so
    // will mutate the vec it has a reference
    // into.  *We* know that it will never do anything
    // to invalidate its own StackFrame, but Rust doesn't.
    // So instead we basically just use Rust's stack and
    // have each wasm `Call` instruction allocate a new
    // StackFrame and pass it to the thing it's calling.
    // I feel like this may cause problems with potential
    // threading applications somewhere down the line
    // (see Python), but for now oh well.
    // Trivially gotten around with unsafe, if we want to.
    // stack: Vec<StackFrame>,
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
/// Per the wasm spec, this contains the **Store**, defined as all the
/// runtime data for a collection of modules: memory's, tables, globals,
/// and stack.  In this implementation, stack frames are locals in the
/// `exec()` method, not an explicit structure field, because otherwise
/// borrowing gets tricky.  We essentially use the Rust stack instead
/// of constructing a separate one.
///
/// The WASM spec has a not-immediately-obvious separation in semantics
/// between the environment in which programs are defined, loaded
/// and validated, where all references are *purely module-local*,
/// and the environment in which programs are executed, where most
/// references are *global*; modules are loaded and all their resources
/// are just shoved
/// into the Store.  It distinguishes these environments by using the
/// term "index" to mean an offset into a module-local environment,
/// and "address" to mean an offset into a global environment.
/// See <https://webassembly.github.io/spec/core/exec/runtime.html>
///
/// A module thus becomes a **module instance** when ready to execute,
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
    pub fn new() -> Self {
        // Don't know if there's a better place to put this, or a less annoying way of doing it
        // while still making it always visible, but this is fine for now.
        #[cfg(target_endian = "big")]
        eprintln!("WARNING: Running on big-endian target architecture!  Results are *not* guarenteed to be correct!");

        Self {
            store: Store::default(),
            state: State::default(),
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
    /// them at the end, but that's a PITA.  Also unnecessary since each module
    /// can only depend on modules that are already loaded; you can't have
    /// circular dependencies.
    ///
    /// This DOES run the module's start function, which potentially
    /// takes forever, soooooo.  That may not be what we want.
    /// However it IS what the spec prescribes, so!
    pub fn with_module(mut self, module: ValidatedModule) -> Result<Self, Error> {
        let module: LoadedModule = module.into_inner();
        let module_instance_address = ModuleAddress(self.state.module_instances.len());

        // We MUST load imports first because they consume the first indices
        // before all local definitions.
        // "Every import defines an index in the respective index space. In each
        // index space, the indices of imports go before the first index of any
        // definition contained in the module itself."

        let types = module.types.clone();
        let name = module.name.clone();
        let mut inst = ModuleInstance {
            name: name,
            types: types,
            exported_functions: vec![],
            exported_tables: None,
            exported_memories: None,
            exported_globals: vec![],
            functions: vec![],
            table: None,
            memory: None,
            globals: vec![],
            start: None,
        };
        inst.resolve_imports(&module, &self.state.module_instances)?;

        for func in module.funcs.iter() {
            let address = FunctionAddress(self.state.funcs.len());
            let functype = module.types[func.typeidx.0].clone();
            let instance = FuncInstance {
                functype: functype,
                locals: func.locals.clone(),
                body: func.body.clone(),
                module: module_instance_address,
                jump_table: FuncInstance::compute_jump_table(&func.body),
            };
            //println!("Created function instance: {:?}", instance);
            self.state.funcs.push(instance);
            inst.functions.push(address);
        }

        // If the module has a memory, clone it, initialize it, shove
        // it into the store, and return the address of it.  Otherwise,
        // return None.
        inst.memory = if let Some(mut memory) = module.mem.clone() {
            // It's sorta meaningless to define a memory when we already
            // import one, since we can only have one.
            if inst.memory.is_some() {
                return Err(Error::Invalid {
                    module: module.name.clone(),
                    reason: "Memory was imported but we also define a local one".to_owned(),
                });
            }
            
            let store = &mut self.store;
            for &(ref offset_expr, ref val) in &module.mem_initializers {
                let offset_value = Interpreter::eval_constexpr(&offset_expr, store).unwrap();
                // TODO: This will panic on failure;
                // replacing it with TryFrom may be apropos.  But annoying.
                let offset_i: u32 = offset_value.into();
                memory.initialize(offset_i, &val)?
            }
            let mem_addr = MemoryAddress(store.mems.len());
            store.mems.push(memory);
            Some(mem_addr)
        } else {
            None
        };

        // Like memories, if the module has a table, clone it, initialize it, shove
        // it into the store, and return the address of it.  Otherwise,
        // return None.
        inst.table = if let Some(mut table) = module.tables.clone() {
            // Same as memory's above; meaningless to define a table if we
            // import one.
            if inst.table.is_some() {
                return Err(Error::Invalid {
                    module: module.name.clone(),
                    reason: "Table was imported but we also define a local one".to_owned(),
                });
            }

            table.initialize(&module.table_initializers)?;
            let table_addr = TableAddress(self.store.tables.len());
            self.store.tables.push(table);
            Some(table_addr)
        } else {
            None
        };

        // This has to be in its own block 'cause we borrow `module`
        // and don't clone all of it.
        inst.globals = {
            // Borrow this so we don't have wacky borrowing problems
            // associated with `self` in a closure and whatever.
            let store = &mut self.store;
            // Create an iterator of initialized Global values
            let initialized_globals = module
                .globals
                .iter()
                .map(|&(ref global, ref init)| {
                    let mut g = global.clone();
                    let init_value =
                        Interpreter::eval_constexpr(init, store)?;
                    println!("Initializing global {:?} to {:?}", g, init_value);
                    g.initialize(init_value);
                    Ok(g)
                })
                .collect::<Result<Vec<_>,_>>()?;

            // Get the address of the next Global slot,
            // shove all the initialized Global's into it,
            // and then get the address again, and that's the
            // mapping for our GlobalAddress's for this module.
            let global_addr_start = store.globals.len();
            store.globals.extend(initialized_globals);
            let global_addr_end = store.globals.len();
            (global_addr_start..global_addr_end)
                .map(GlobalAddress)
                .collect()
        };

        // Start function.
        // TODO: Need to handle index-out-of-bound here
        inst.start = module.start
            .map(|start_idx| inst.functions[start_idx.0]);
        //println!("Instance start function: {:?}, module start function: {:?}", inst.start, module.start);
        // Save it for later too.
        let start_function = inst.start;

        // Great, instance is created, add it to the State
        self.state.modules.insert(module.name.to_owned(), module);
        self.state.module_instances.push(inst);

        // Run start function.
        if let Some(function_addr) = start_function {
            Interpreter::exec(&mut self.store, &self.state, function_addr, &[]);
        }

        Ok(self)
    }

    /// Evaluates the constexpr in the current context.
    /// This is a PITA 'cause a constexpr might be `get_global`, but hey.
    fn eval_constexpr(expr: &ConstExpr, store: &Store) -> Result<Value, Error> {
        // I have no damn idea why a constexpr is defined to be a sequence
        // when it seems to only ever actually use the last value.
        let expr = expr.0.last()
            .expect("Expected non-empty constexpr, got empty one; should never happen");
        match *expr {
            ConstOpcode::I32Const(v) => Ok(Value::I32(v)),
            ConstOpcode::I64Const(v) => Ok(Value::I64(v)),
            ConstOpcode::F32Const(v) => Ok(Value::F32(v)),
            ConstOpcode::F64Const(v) => Ok(Value::F64(v)),
            ConstOpcode::GetGlobal(i) => unimplemented!(),
        }
    }

    /// Returns a GlobalAddress from a given index
    fn resolve_global(state: &State, module_addr: ModuleAddress, idx: GlobalIdx) -> GlobalAddress {
        assert!(module_addr.0 < state.module_instances.len());
        let module_instance = &state.module_instances[module_addr.0];
        assert!(idx.0 < module_instance.globals.len());
        module_instance.globals[idx.0]
    }

    /// Returns a FunctionAddress from a given index
    fn resolve_function(
        state: &State,
        module_addr: ModuleAddress,
        idx: FuncIdx,
    ) -> FunctionAddress {
        assert!(module_addr.0 < state.module_instances.len());
        let module_instance = &state.module_instances[module_addr.0];
        assert!(idx.0 < module_instance.functions.len());
        module_instance.functions[idx.0]
    }

    /// Returns a reference to a `FuncType` from a given index
    ///
    /// This is somewhat asymmetric with everything else, but there is no
    /// explicit "type address" type described in wasm, since types are completely
    /// local to modules.
    fn resolve_type(state: &State, module_addr: ModuleAddress, idx: TypeIdx) -> &FuncType {
        assert!(module_addr.0 < state.module_instances.len());
        let module_instance = &state.module_instances[module_addr.0];
        assert!(idx.0 < module_instance.types.len());
        &module_instance.types[idx.0]
    }

    /// Returns a MemoryAddress for the Memory of a given ModuleInstance.
    /// Modules can currently only have one Memory, so it's pretty easy.
    fn resolve_memory(state: &State, module_addr: ModuleAddress) -> MemoryAddress {
        assert!(module_addr.0 < state.module_instances.len());
        let module_instance = &state.module_instances[module_addr.0];
        assert!(module_instance.memory.is_some());
        module_instance.memory.unwrap()
    }

    /// Returns a TableAddress for the Table of a given ModuleInstance.
    /// Modules can currently only have one Table, so it's pretty easy.
    fn resolve_table(state: &State, module_addr: ModuleAddress) -> TableAddress {
        assert!(module_addr.0 < state.module_instances.len());
        let module_instance = &state.module_instances[module_addr.0];
        assert!(module_instance.table.is_some());
        module_instance.table.unwrap()
    }

    /// Get a global variable by *index*.  Needs a module instance
    /// address to look up the global variable's address.
    /// Panics if out of bounds.
    ///
    /// This is unused since it creates irritating double-borrows.
    fn get_global(
        globals: &[Global],
        state: &State,
        module_addr: ModuleAddress,
        idx: GlobalIdx,
    ) -> Value {
        let global_addr = Interpreter::resolve_global(state, module_addr, idx);
        globals[global_addr.0].value
    }

    /// Sets a global variable by *index*.  Needs a module instance
    /// address to look up the global variable's address.
    /// Panics if out of bounds or if the type of the new
    /// variable does not match the old one(?).
    fn set_global(
        globals: &mut [Global],
        state: &State,
        module_addr: ModuleAddress,
        idx: GlobalIdx,
        vl: Value,
    ) {
        let global_addr = Interpreter::resolve_global(state, module_addr, idx);
        assert!(globals[global_addr.0].mutable);
        assert_eq!(globals[global_addr.0].variable_type, vl.get_type());
        globals[global_addr.0].value = vl;
    }

    /// Assigns a value to the given `memory` with the given function.
    fn set_memory_with<F, N>(
        mems: &mut [Memory],
        state: &State,
        module_addr: ModuleAddress,
        offset: usize,
        f: F,
        vl: N,
    ) where
        F: Fn(&mut [u8], N),
    {
        let memory_address = Interpreter::resolve_memory(state, module_addr);
        let mem = &mut mems[memory_address.0];
        assert!(offset + std::mem::size_of::<N>() < mem.data.len());
        f(&mut mem.data[offset..], vl)
    }

    /// Reads data from a slice of the given `memory` with the given function
    fn get_memory_with<F, N>(
        mems: &[Memory],
        state: &State,
        module_addr: ModuleAddress,
        offset: usize,
        f: F,
    ) -> N
    where
        F: Fn(&[u8]) -> N,
    {
        let memory_address = Interpreter::resolve_memory(state, module_addr);
        let mem = &mems[memory_address.0];
        assert!(offset + std::mem::size_of::<N>() < mem.data.len());
        f(&mem.data[offset..])
    }

    fn trap() {
        panic!("Trap occured!  Aieee!")
    }

    fn exec_const(frame: &mut StackFrame, vl: Value) {
        frame.push(vl);
    }

    /// Executes a load instruction, using the given function to
    /// convert the memory's `&[u8]` into the given Value type.
    fn exec_load<F, N>(
        frame: &mut StackFrame,
        store: &mut Store,
        state: &State,
        module: ModuleAddress,
        offset: u32,
        func: F,
    ) where
        F: Fn(&[u8]) -> N,
        N: Into<Value>,
    {
        let address = frame.pop_as::<i32>();
        // TODO: Should trap on runtime errors!
        // ...though now I'm wondering how to implement trap...
        // it's kinda gotta be a return value from a function,
        // or MAYBE a flag that gets set on the interpreter state or such...
        let effective_address = address.checked_add(offset as i32)
            .expect("Address of load instruction wrapped; this is an error!");
        let mem_contents = Interpreter::get_memory_with(
            &mut store.mems,
            &state,
            module,
            effective_address as usize,
            func,
        ).into();
        frame.push(mem_contents);
    }

    /// Executes a load instruction, using the given function to
    /// convert the memory's `&[u8]` into the the SourceN type,
    /// then sign-extending it (based on whether it's signed or unsigned)
    /// into DestN.
    fn exec_load_extend<F, SourceN, DestN>(
        frame: &mut StackFrame,
        store: &mut Store,
        state: &State,
        module: ModuleAddress,
        offset: u32,
        func: F,
    ) where
        F: Fn(&[u8]) -> SourceN,
        SourceN: Extend<DestN>,
        DestN: Into<Value>,
    {
        let address = frame.pop_as::<i32>();
        // TODO: Should trap on error.
        let effective_address = address.checked_add(offset as i32)
            .expect("Address of load instruction wrapped; this is an error!");
        let mem_contents = Interpreter::get_memory_with(
            &mut store.mems,
            &state,
            module,
            effective_address as usize,
            func,
        ).extend()
            .into();
        frame.push(mem_contents);
    }

    /// Executes a store instruction, using the given function to
    /// write the Value type into the memory's `&mut [u8]`
    fn exec_store<F, N>(
        frame: &mut StackFrame,
        store: &mut Store,
        state: &State,
        module: ModuleAddress,
        offset: u32,
        func: F,
    ) where
        F: Fn(&mut [u8], N),
        N: From<Value>,
    {
        let vl = frame.pop_as::<N>();
        let address = frame.pop_as::<i32>();
        // TODO: Trap on error.
        let effective_address = address.checked_add(offset as i32)
            .expect("Address of load instruction wrapped; this is an error!");
        let effective_address = address + offset as i32;
        Interpreter::set_memory_with(
            &mut store.mems,
            &state,
            module,
            effective_address as usize,
            func,
            vl,
        );
    }

    /// Wraps/truncates the the Value on the stack from the given SourceN type
    /// to the DestN type, then stores it in memory.
    fn exec_store_wrap<F, SourceN, DestN>(
        frame: &mut StackFrame,
        store: &mut Store,
        state: &State,
        module: ModuleAddress,
        offset: u32,
        func: F,
    ) where
        F: Fn(&mut [u8], DestN),
        SourceN: From<Value> + Wrap<DestN>,
    {
        let vl: DestN = frame.pop_as::<SourceN>().wrap();
        let address = frame.pop_as::<i32>();
        // TODO: Trap on error.
        let effective_address = address.checked_add(offset as i32)
            .expect("Address of load instruction wrapped; this is an error!");
        Interpreter::set_memory_with(
            &mut store.mems,
            &state,
            module,
            effective_address as usize,
            func,
            vl,
        );
    }

    /// Helper function for running binary operations that pop
    /// two values from the stack and push one result
    fn exec_binop<T1, T2, Res, F>(frame: &mut StackFrame, op: F)
    where
        T1: From<Value>,
        T2: From<Value>,
        Res: Into<Value>,
        F: Fn(T1, T2) -> Res,
    {
        let (a, b) = frame.pop2_as::<T1, T2>();
        frame.push(op(a, b).into());
    }

    /// Helper function for running binary operations that pop
    /// two values from the stack and push one result
    fn exec_uniop<T, Res, F>(frame: &mut StackFrame, op: F)
    where
        T: From<Value>,
        Res: Into<Value>,
        F: Fn(T) -> Res,
    {
        let a = frame.pop_as::<T>();
        frame.push(op(a).into());
    }

    /// Helper function for running a function call.
    fn exec_call(
        frame: &mut StackFrame,
        store: &mut Store,
        state: &State,
        function_addr: FunctionAddress,
    ) {
        // Typecheck and get appropriate arguments off the stack to pass
        // to the called function.
        let f = &state.funcs[function_addr.0];
        let return_val = {
            assert!(f.functype.params.len() <= frame.value_stack.len());
            let params_slice = if f.functype.params.len() == 0 {
                &[]
            } else {
                let params_end = frame.value_stack.len();
                let params_start = params_end - f.functype.params.len();
                let params_slice = &frame.value_stack[params_start..params_end];
                for (param, desired_type) in params_slice.iter().zip(&f.functype.params) {
                    assert_eq!(param.get_type(), *desired_type);
                }
                params_slice
            };
            // Recurse into `exec()`, which creates a new stack frame.
            Interpreter::exec(store, state, function_addr, params_slice)
        };

        // Because a function call must actually pop values off the stack,
        // we have to remove the values that were passed to the function in
        // `params_slice`
        // TODO: Might be easier to just slice them off directly, since they get
        // copied anyway?
        let new_stack_len = frame.value_stack.len() - f.functype.params.len();
        frame.value_stack.truncate(new_stack_len);

        // Great, now check that the return value matches the stated
        // return type, and push it to the values stack.
        let return_type = return_val.map(|v| v.get_type());
        assert_eq!(return_type, f.functype.return_type);
        if let Some(v) = return_val {
            frame.value_stack.push(v);
        }
    }

    /// Actually do the interpretation of the given function, creating
    /// a stack frame for it.  Returns the return value of the function,
    /// if any.
    pub fn exec(
        store: &mut Store,
        state: &State,
        func: FunctionAddress,
        args: &[Value],
    ) -> Option<Value> {
        let func = &state.funcs[func.0];
        // println!("Params: {:?}, args: {:?}", func.functype.params, args);
        let frame = &mut StackFrame::from_func_instance(func, args);
        match func.body {
            FuncBody::HostFunction(ref f) => {
                (*f)(&mut frame.value_stack);
            }
            FuncBody::Opcodes(ref opcodes) => {
                use parity_wasm::elements::Opcode::*;
                use std::usize;
                loop {
                    if frame.ip == opcodes.len() {
                        break;
                    }
                    let op = &opcodes[frame.ip];

                    // println!("Frame: {:?}", frame);
                    // println!("Op: {:?}", op);
                    match *op {
                        Unreachable => panic!("Unreachable?"),
                        Nop => (),
                        Block(_blocktype) => {
                            // TODO: Verify blocktype
                            let jump_target_idx = func.jump_table
                                .binary_search_by(|jt| jt.block_start_instruction.cmp(&frame.ip))
                                .expect("Cannot find matching jump table for block statement");
                            let jump_target = &func.jump_table[jump_target_idx];
                            frame.push_label(BlockLabel(jump_target.block_end_instruction));
                        }
                        Loop(_blocktype) => {
                            // TODO: Verify blocktype
                            // Instruction index to jump to on branch or such.
                            let end_idx = frame.ip + 1;
                            frame.push_label(BlockLabel(end_idx));
                        }
                        If(_blocktype) => {
                            // TODO: Verify blocktype
                            let vl = frame.pop_as::<i32>();
                            let jump_target_idx = func.jump_table
                                .binary_search_by(|jt| jt.block_start_instruction.cmp(&frame.ip))
                                .expect("Cannot find matching jump table for if statement");
                            let jump_target = &func.jump_table[jump_target_idx];
                            frame.push_label(BlockLabel(jump_target.block_end_instruction));
                            if vl != 0 {
                                // continue
                            } else {
                                // Jump to instruction after the else section
                                frame.ip = jump_target.else_instruction + 1;
                            }
                        }
                        Else => {
                            // Done with if part of the statement,
                            // skip to (just after) the end.
                            let target_ip = frame.pop_label(0);
                            frame.ip = target_ip.0 + 1;
                        }
                        End => {
                            // Done with whatever block we're in
                            // OR, we are at the end of the function and must return;
                            // if so, popping the label is NOT what we want 'cause we
                            // have no labels.
                            // TODO: This may still be incorrect.
                            if frame.ip != opcodes.len() - 1 {
                                frame.pop_label(0);
                            } // else we're at the end of the function, do nothing
                        }
                        Br(i) => {
                            let target_ip = frame.pop_label(i as usize);
                            frame.ip = target_ip.0;
                        }
                        BrIf(i) => {
                            let i = i as usize;
                            let vl = frame.pop_as::<i32>();
                            if vl != 0 {
                                let target_ip = frame.pop_label(i);
                                frame.ip = target_ip.0;
                            }
                        }
                        BrTable(ref v, i) => {
                            // TODO: Double-check this is correct, I don't fully
                            // understand its goals.  It's a computed jump into
                            // a list of labels, but, needs verification.
                            let i = i as usize;
                            let vl = frame.pop_as::<i32>() as usize;
                            let target_label = if vl < v.len() { v[vl] as usize } else { i };
                            let target_ip = frame.pop_label(target_label);
                            frame.ip = target_ip.0;
                        }
                        Return => {
                            break;
                        }
                        Call(i) => {
                            let i = i as usize;
                            let function_addr =
                                Interpreter::resolve_function(state, func.module, FuncIdx(i));
                            Interpreter::exec_call(frame, store, state, function_addr);
                        }
                        CallIndirect(x, _) => {
                            // Okay, x is the expected type signature of the function we
                            // are trying to call.
                            // So we pop an i32 i from the stack, use that to index into
                            // the table to get a function index, then call that
                            // function
                            let x = x as usize;
                            let func_type =
                                Interpreter::resolve_type(state, func.module, TypeIdx(x));
                            let i = frame.pop_as::<u32>() as usize;
                            let table_addr = Interpreter::resolve_table(state, func.module);
                            let function_index = {
                                let table = &store.tables[table_addr.0];
                                table.data[i]
                            };
                            let function_addr =
                                Interpreter::resolve_function(state, func.module, function_index);
                            // Make sure that the function we've actually retrieved has the same signature as the
                            // type we want.
                            assert_eq!(&state.funcs[function_addr.0].functype, func_type);
                            Interpreter::exec_call(frame, store, state, function_addr);
                        }
                        Drop => {
                            frame.pop();
                        }
                        Select => {
                            let selector = frame.pop_as::<i32>();
                            let v2 = frame.pop();
                            let v1 = frame.pop();
                            if selector != 0 {
                                frame.push(v1);
                            } else {
                                frame.push(v2);
                            }
                        }
                        GetLocal(i) => {
                            let i = i as usize;
                            let vl = frame.get_local(i as usize);
                            frame.push(vl);
                        }
                        SetLocal(i) => {
                            let i = i as usize;
                            let vl = frame.pop();
                            frame.set_local(i, vl);
                        }
                        TeeLocal(i) => {
                            let i = i as usize;
                            let vl = frame.peek();
                            frame.set_local(i, vl);
                        }
                        GetGlobal(i) => {
                            let i = i as usize;
                            let vl = Interpreter::get_global(
                                &store.globals,
                                &state,
                                func.module,
                                GlobalIdx(i),
                            );
                            frame.push(vl);
                        }
                        SetGlobal(i) => {
                            let i = i as usize;
                            let vl = frame.pop();
                            Interpreter::set_global(
                                &mut store.globals,
                                &state,
                                func.module,
                                GlobalIdx(i),
                                vl,
                            );
                        }
                        I32Load(offset, _align) => {
                            Interpreter::exec_load(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_i32,
                            );
                        }
                        I64Load(offset, _align) => {
                            Interpreter::exec_load(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_i64,
                            );
                        }
                        F32Load(offset, _align) => {
                            Interpreter::exec_load(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_f32,
                            );
                        }
                        F64Load(offset, _align) => {
                            Interpreter::exec_load(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_f64,
                            );
                        }
                        I32Load8S(offset, _align) => {
                            Interpreter::exec_load_extend::<_, i8, i32>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                |mem| mem[0] as i8,
                            );
                        }
                        I32Load8U(offset, _align) => {
                            Interpreter::exec_load_extend::<_, u8, i32>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                |mem| mem[0] as u8,
                            );
                        }
                        I32Load16S(offset, _align) => {
                            Interpreter::exec_load_extend::<_, i16, i32>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_i16,
                            );
                        }
                        I32Load16U(offset, _align) => {
                            Interpreter::exec_load_extend::<_, u16, i32>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_u16,
                            );
                        }
                        I64Load8S(offset, _align) => {
                            Interpreter::exec_load_extend::<_, i8, i64>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                |mem| mem[0] as i8,
                            );
                        }
                        I64Load8U(offset, _align) => {
                            Interpreter::exec_load_extend::<_, u8, i64>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                |mem| mem[0] as u8,
                            );
                        }
                        I64Load16S(offset, _align) => {
                            Interpreter::exec_load_extend::<_, i16, i64>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_i16,
                            );
                        }
                        I64Load16U(offset, _align) => {
                            Interpreter::exec_load_extend::<_, u16, i64>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_u16,
                            );
                        }
                        I64Load32S(offset, _align) => {
                            Interpreter::exec_load_extend::<_, i32, i64>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_i32,
                            );
                        }
                        I64Load32U(offset, _align) => {
                            Interpreter::exec_load_extend::<_, u32, i64>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::read_u32,
                            );
                        }
                        I32Store(offset, _align) => {
                            Interpreter::exec_store(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::write_i32,
                            );
                        }
                        I64Store(offset, _align) => {
                            Interpreter::exec_store(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::write_i64,
                            );
                        }
                        F32Store(offset, _align) => {
                            Interpreter::exec_store(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::write_f32,
                            );
                        }
                        F64Store(offset, _align) => {
                            Interpreter::exec_store(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::write_f64,
                            );
                        }
                        I32Store8(offset, _align) => {
                            // `byteorder` doesn't have write_i8 since it's a bit redundant,
                            // so we make our own.
                            Interpreter::exec_store_wrap::<_, i32, i8>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                |mem, x| mem[0] = x as u8,
                            );
                        }
                        I32Store16(offset, _align) => {
                            Interpreter::exec_store_wrap::<_, i32, i16>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::write_i16,
                            );
                        }
                        I64Store8(offset, _align) => {
                            Interpreter::exec_store_wrap::<_, i64, i8>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                |mem, x| mem[0] = x as u8,
                            );
                        }
                        I64Store16(offset, _align) => {
                            Interpreter::exec_store_wrap::<_, i64, i16>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::write_i16,
                            );
                        }
                        I64Store32(offset, _align) => {
                            Interpreter::exec_store_wrap::<_, i64, i32>(
                                frame,
                                store,
                                state,
                                func.module,
                                offset,
                                byteorder::LittleEndian::write_i32,
                            );
                        }
                        CurrentMemory(_) => {
                            let module_addr = func.module;
                            let memory_addr = Interpreter::resolve_memory(state, module_addr);
                            let mem = &store.mems[memory_addr.0];
                            frame.push(mem.len().into())
                        }
                        GrowMemory(_) => {
                            let size_delta = frame.pop_as::<i32>();
                            let module_addr = func.module;
                            let memory_addr = Interpreter::resolve_memory(state, module_addr);
                            let mem = &mut store.mems[memory_addr.0];
                            let prev_size = mem.len();
                            // TODO: We should return -1 if enough memory cannot be allocated.
                            mem.resize(size_delta);
                            frame.push(prev_size.into());
                        }
                        I32Const(i) => Interpreter::exec_const(frame, i.into()),
                        I64Const(l) => Interpreter::exec_const(frame, l.into()),
                        // Why oh why are these floats represented as u32 and u64?
                        // Because this is the serialized representation, sigh.
                        F32Const(i) => {
                            Interpreter::exec_const(frame, Value::from(u32_to_f32(i)));
                        }
                        F64Const(l) => {
                            Interpreter::exec_const(frame, Value::from(u64_to_f64(l)));
                        }
                        I32Eqz => {
                            Interpreter::exec_uniop::<i32, bool, _>(frame, |x| i32::eq(&x, &0));
                        }
                        I32Eq => {
                            Interpreter::exec_binop(frame, |x: i32, y: i32| i32::eq(&x, &y));
                        }
                        I32Ne => {
                            Interpreter::exec_binop(frame, |x: i32, y: i32| i32::ne(&x, &y));
                        }
                        I32LtS => {
                            Interpreter::exec_binop(frame, |x, y| i32::lt(&x, &y));
                        }
                        I32LtU => {
                            Interpreter::exec_binop(frame, |x, y| u32::lt(&x, &y));
                        }
                        I32GtS => {
                            Interpreter::exec_binop(frame, |x, y| i32::gt(&x, &y));
                        }
                        I32GtU => {
                            Interpreter::exec_binop(frame, |x, y| u32::gt(&x, &y));
                        }
                        I32LeS => {
                            Interpreter::exec_binop(frame, |x, y| i32::le(&x, &y));
                        }
                        I32LeU => {
                            Interpreter::exec_binop(frame, |x, y| u32::le(&x, &y));
                        }
                        I32GeS => {
                            Interpreter::exec_binop(frame, |x, y| i32::ge(&x, &y));
                        }
                        I32GeU => {
                            Interpreter::exec_binop(frame, |x, y| u32::ge(&x, &y));
                        }
                        I64Eqz => {
                            Interpreter::exec_uniop::<i64, bool, _>(frame, |x| i64::eq(&x, &0));
                        }
                        I64Eq => {
                            Interpreter::exec_binop(frame, |x: i64, y: i64| i64::eq(&x, &y));
                        }
                        I64Ne => {
                            Interpreter::exec_binop(frame, |x: i64, y: i64| i64::ne(&x, &y));
                        }
                        I64LtS => {
                            Interpreter::exec_binop(frame, |x, y| i64::lt(&x, &y));
                        }
                        I64LtU => {
                            Interpreter::exec_binop(frame, |x, y| u64::lt(&x, &y));
                        }
                        I64GtS => {
                            Interpreter::exec_binop(frame, |x, y| i64::gt(&x, &y));
                        }
                        I64GtU => {
                            Interpreter::exec_binop(frame, |x, y| u64::gt(&x, &y));
                        }
                        I64LeS => {
                            Interpreter::exec_binop(frame, |x, y| i64::le(&x, &y));
                        }
                        I64LeU => {
                            Interpreter::exec_binop(frame, |x, y| u64::le(&x, &y));
                        }
                        I64GeS => {
                            Interpreter::exec_binop(frame, |x, y| i64::ge(&x, &y));
                        }
                        I64GeU => {
                            Interpreter::exec_binop(frame, |x, y| u64::ge(&x, &y));
                        }
                        F32Eq => {
                            Interpreter::exec_binop(frame, |x: f32, y: f32| f32::eq(&x, &y));
                        }
                        F32Ne => {
                            Interpreter::exec_binop(frame, |x: f32, y: f32| f32::ne(&x, &y));
                        }
                        F32Lt => {
                            Interpreter::exec_binop(frame, |x, y| f32::lt(&x, &y));
                        }
                        F32Gt => {
                            Interpreter::exec_binop(frame, |x, y| f32::gt(&x, &y));
                        }
                        F32Le => {
                            Interpreter::exec_binop(frame, |x, y| f32::le(&x, &y));
                        }
                        F32Ge => {
                            Interpreter::exec_binop(frame, |x, y| f32::ge(&x, &y));
                        }
                        F64Eq => {
                            Interpreter::exec_binop(frame, |x: f64, y: f64| f64::eq(&x, &y));
                        }
                        F64Ne => {
                            Interpreter::exec_binop(frame, |x: f64, y: f64| f64::ne(&x, &y));
                        }
                        F64Lt => {
                            Interpreter::exec_binop(frame, |x, y| f64::lt(&x, &y));
                        }
                        F64Gt => {
                            Interpreter::exec_binop(frame, |x, y| f64::gt(&x, &y));
                        }
                        F64Le => {
                            Interpreter::exec_binop(frame, |x, y| f64::le(&x, &y));
                        }
                        F64Ge => {
                            Interpreter::exec_binop(frame, |x, y| f64::ge(&x, &y));
                        }
                        I32Clz => {
                            Interpreter::exec_uniop(frame, i32::leading_zeros);
                        }
                        I32Ctz => {
                            Interpreter::exec_uniop(frame, i32::trailing_zeros);
                        }
                        I32Popcnt => {
                            Interpreter::exec_uniop(frame, i32::count_zeros);
                        }
                        I32Add => {
                            Interpreter::exec_binop(frame, i32::wrapping_add);
                        }
                        I32Sub => {
                            Interpreter::exec_binop(frame, i32::wrapping_sub);
                        }
                        I32Mul => {
                            Interpreter::exec_binop(frame, i32::wrapping_mul);
                        }
                        I32DivS => {
                            Interpreter::exec_binop(frame, i32::wrapping_div);
                        }
                        I32DivU => {
                            Interpreter::exec_binop(frame, u32::wrapping_div);
                        }
                        I32RemS => {
                            Interpreter::exec_binop(frame, i32::wrapping_rem);
                        }
                        I32RemU => {
                            Interpreter::exec_binop(frame, u32::wrapping_rem);
                        }
                        I32And => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i32, i32, _, _>(frame, i32::bitand);
                        }
                        I32Or => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i32, i32, _, _>(frame, i32::bitor);
                        }
                        I32Xor => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i32, i32, _, _>(frame, i32::bitxor);
                        }
                        I32Shl => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i32, i32, _, _>(frame, i32::shl);
                        }
                        I32ShrS => {
                            Interpreter::exec_binop::<i32, u32, _, _>(frame, i32::wrapping_shr);
                        }
                        I32ShrU => {
                            Interpreter::exec_binop::<u32, u32, _, _>(frame, u32::wrapping_shr);
                        }
                        I32Rotl => {
                            Interpreter::exec_binop(frame, i32::rotate_left);
                        }
                        I32Rotr => {
                            Interpreter::exec_binop(frame, i32::rotate_right);
                        }
                        I64Clz => {
                            Interpreter::exec_uniop(frame, i64::leading_zeros);
                        }
                        I64Ctz => {
                            Interpreter::exec_uniop(frame, i64::trailing_zeros);
                        }
                        I64Popcnt => {
                            Interpreter::exec_uniop(frame, i64::count_zeros);
                        }
                        I64Add => {
                            Interpreter::exec_binop(frame, i64::wrapping_add);
                        }
                        I64Sub => {
                            Interpreter::exec_binop(frame, i64::wrapping_sub);
                        }
                        I64Mul => {
                            Interpreter::exec_binop(frame, i64::wrapping_mul);
                        }
                        I64DivS => {
                            Interpreter::exec_binop(frame, i64::wrapping_div);
                        }
                        I64DivU => {
                            Interpreter::exec_binop(frame, u64::wrapping_div);
                        }
                        I64RemS => {
                            Interpreter::exec_binop(frame, i64::wrapping_rem);
                        }
                        I64RemU => {
                            Interpreter::exec_binop(frame, u64::wrapping_rem);
                        }
                        I64And => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i64, i64, _, _>(frame, i64::bitand);
                        }
                        I64Or => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i64, i64, _, _>(frame, i64::bitor);
                        }
                        I64Xor => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i64, i64, _, _>(frame, i64::bitxor);
                        }
                        I64Shl => {
                            use std::ops::*;
                            Interpreter::exec_binop::<i64, i64, _, _>(frame, i64::shl);
                        }
                        I64ShrS => {
                            Interpreter::exec_binop::<i64, u32, _, _>(frame, i64::wrapping_shr);
                        }
                        I64ShrU => {
                            Interpreter::exec_binop::<u64, u32, _, _>(frame, u64::wrapping_shr);
                        }
                        I64Rotl => {
                            Interpreter::exec_binop::<i64, u32, _, _>(frame, i64::rotate_left);
                        }
                        I64Rotr => {
                            Interpreter::exec_binop::<i64, u32, _, _>(frame, i64::rotate_right);
                        }
                        F32Abs => {
                            Interpreter::exec_uniop::<f32, _, _>(frame, f32::abs);
                        }
                        F32Neg => {
                            use std::ops::Neg;
                            Interpreter::exec_uniop::<f32, _, _>(frame, Neg::neg);
                        }
                        F32Ceil => {
                            Interpreter::exec_uniop::<f32, _, _>(frame, f32::ceil);
                        }
                        F32Floor => {
                            Interpreter::exec_uniop::<f32, _, _>(frame, f32::floor);
                        }
                        F32Trunc => {
                            Interpreter::exec_uniop::<f32, _, _>(frame, f32::trunc);
                        }
                        F32Nearest => {
                            // TODO: Double-check rounding behavior is correct
                            Interpreter::exec_uniop::<f32, _, _>(frame, f32::round);
                        }
                        F32Sqrt => {
                            Interpreter::exec_uniop::<f32, _, _>(frame, f32::sqrt);
                        }
                        F32Add => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::add);
                        }
                        F32Sub => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::sub);
                        }
                        F32Mul => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::mul);
                        }
                        F32Div => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::div);
                        }
                        F32Min => {
                            Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::min);
                        }
                        F32Max => {
                            Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::max);
                        }
                        F32Copysign => {
                            Interpreter::exec_binop::<f32, f32, _, _>(frame, copysign);
                        }
                        F64Abs => {
                            Interpreter::exec_uniop::<f64, _, _>(frame, f64::abs);
                        }
                        F64Neg => {
                            use std::ops::Neg;
                            Interpreter::exec_uniop::<f64, _, _>(frame, Neg::neg);
                        }
                        F64Ceil => {
                            Interpreter::exec_uniop::<f64, _, _>(frame, f64::ceil);
                        }
                        F64Floor => {
                            Interpreter::exec_uniop::<f64, _, _>(frame, f64::floor);
                        }
                        F64Trunc => {
                            Interpreter::exec_uniop::<f64, _, _>(frame, f64::trunc);
                        }
                        F64Nearest => {
                            // TODO: Double-check rounding behavior is correct
                            Interpreter::exec_uniop::<f64, _, _>(frame, f64::round);
                        }
                        F64Sqrt => {
                            Interpreter::exec_uniop::<f64, _, _>(frame, f64::sqrt);
                        }
                        F64Add => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::add);
                        }
                        F64Sub => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::sub);
                        }
                        F64Mul => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::mul);
                        }
                        F64Div => {
                            use std::ops::*;
                            Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::div);
                        }
                        F64Min => {
                            Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::min);
                        }
                        F64Max => {
                            Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::max);
                        }
                        F64Copysign => {
                            Interpreter::exec_binop::<f64, f64, _, _>(frame, copysign);
                        }
                        I32WrapI64 => {
                            Interpreter::exec_uniop::<i64, i32, _>(frame, Wrap::wrap);
                        }
                        I32TruncSF32 => {
                            Interpreter::exec_uniop::<f32, i32, _>(frame, truncate_to_int);
                        }
                        I32TruncUF32 => {
                            // TODO: Verify signedness works here
                            Interpreter::exec_uniop::<f32, u32, _>(frame, truncate_to_int);
                        }
                        I32TruncSF64 => {
                            Interpreter::exec_uniop::<f64, i32, _>(frame, truncate_to_int);
                        }
                        I32TruncUF64 => {
                            // TODO: Verify signedness
                            Interpreter::exec_uniop::<f64, u32, _>(frame, truncate_to_int);
                        }
                        I64ExtendSI32 => {
                            Interpreter::exec_uniop::<i32, i64, _>(frame, From::from);
                        }
                        I64ExtendUI32 => {
                            Interpreter::exec_uniop::<u32, i64, _>(frame, From::from);
                        }
                        I64TruncSF32 => {
                            Interpreter::exec_uniop::<f32, i64, _>(frame, truncate_to_int);
                        }
                        I64TruncUF32 => {
                            Interpreter::exec_uniop::<f32, u64, _>(frame, truncate_to_int);
                        }
                        I64TruncSF64 => {
                            Interpreter::exec_uniop::<f64, i64, _>(frame, truncate_to_int);
                        }
                        I64TruncUF64 => {
                            Interpreter::exec_uniop::<f64, u64, _>(frame, truncate_to_int);
                        }
                        F32ConvertSI32 => {
                            Interpreter::exec_uniop::<f32, i32, _>(frame, round_to_int);
                        }
                        F32ConvertUI32 => {
                            Interpreter::exec_uniop::<f32, u32, _>(frame, round_to_int);
                        }
                        F32ConvertSI64 => {
                            Interpreter::exec_uniop::<f32, i64, _>(frame, round_to_int);
                        }
                        F32ConvertUI64 => {
                            Interpreter::exec_uniop::<f32, u64, _>(frame, round_to_int);
                        }
                        F32DemoteF64 => {
                            Interpreter::exec_uniop::<f64, _, _>(frame, |f| f as f32);
                        }
                        F64ConvertSI32 => {
                            Interpreter::exec_uniop::<f64, i32, _>(frame, round_to_int);
                        }
                        F64ConvertUI32 => {
                            Interpreter::exec_uniop::<f64, u32, _>(frame, round_to_int);
                        }
                        F64ConvertSI64 => {
                            Interpreter::exec_uniop::<f64, i64, _>(frame, round_to_int);
                        }
                        F64ConvertUI64 => {
                            Interpreter::exec_uniop::<f64, u64, _>(frame, round_to_int);
                        }
                        F64PromoteF32 => {
                            Interpreter::exec_uniop::<f32, _, _>(frame, f64::from);
                        }
                        I32ReinterpretF32 => {
                            // TODO: Check that this is going the correct direction,
                            // i32 -> f32
                            Interpreter::exec_uniop(frame, f32::from_bits);
                        }
                        I64ReinterpretF64 => {
                            // TODO: Check that this is going the correct direction,
                            // i64 -> f64
                            Interpreter::exec_uniop(frame, f64::from_bits);
                        }
                        F32ReinterpretI32 => {
                            // TODO: Check that this is going the correct direction,
                            // f32 -> i32
                            Interpreter::exec_uniop(frame, f32::to_bits);
                        }
                        F64ReinterpretI64 => {
                            // TODO: Check that this is going the correct direction,
                            // f64 -> i64
                            Interpreter::exec_uniop(frame, f64::to_bits);
                        }
                    }
                    frame.ip += 1;
                }
            }
        }
        // Return the function's return value (if any).
        println!("Value stack is: {:?}", frame.value_stack);
        let return_type = frame.value_stack.last().map(|vl| vl.get_type());
        assert_eq!(return_type, func.functype.return_type);
        frame.value_stack.last().cloned()
    }

    /// A nice shortcut to run `exec()` with appropriate values.
    pub fn run(&mut self, func: FunctionAddress, args: &[Value]) -> Option<Value> {
        let state = &self.state;
        let store = &mut self.store;
        Interpreter::exec(store, state, func, args)
    }

    /// Looks up a function with the given name
    /// and executes it with the given arguments.
    /// Returns the function's return value, if any.
    pub fn run_export(
        &mut self,
        module_name: &str,
        func_name: &str,
        args: &[Value],
    ) -> Result<Option<Value>, Error> {
        let function_addr = {
            // TODO: Probably some duplication with ModuleInstance::resolve_imports()
            // but argh.
            let target_module = self.state
                .module_instances
                .iter()
                .find(|m| module_name == m.name)
                .ok_or(Error::ModuleNotFound {
                    module: module_name.to_owned(),
                    dependent_module: "<Interpreter::run_export()>".to_owned(),
                })?;
            //println!("target module: {:#?}", target_module);
            let function_idx = target_module
                .exported_functions
                .iter()
                .find(|funcs| {
                    // println!("Searching for {}, got {}", func_name, funcs.name);
                    funcs.name == func_name
                })
                .ok_or(Error::NotExported {
                    module: module_name.to_owned(),
                    name: func_name.to_owned(),
                    typ: "function".to_owned(),
                    dependent_module: "<Interpreter::run_export()>".to_owned(),
                })?;
            target_module.functions[function_idx.value.0]
        };
        Ok(self.run(function_addr, args))
    }
}
