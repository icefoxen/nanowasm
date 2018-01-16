//! An attempt at creating an embeddable wasm interpreter that's
//! bloody easier to use and debug than the `parity-wasm` one.

extern crate byteorder;
extern crate num;
extern crate parity_wasm;

use byteorder::ByteOrder;
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
#[derive(Debug, Clone, PartialEq)]
pub struct FuncType {
    params: Vec<elements::ValueType>,
    return_type: Option<elements::ValueType>,
}

impl<'a> From<&'a elements::Type> for FuncType {
    fn from(t: &'a elements::Type) -> Self {
        match *t {
            elements::Type::Function(ref ft) => Self {
                params: ft.params().to_owned(),
                return_type: ft.return_type(),
            },
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

impl From<Value> for i32 {
    fn from(vl: Value) -> i32 {
        match vl {
            Value::I32(i) => i,
            _ => panic!("Unwrap value failed"),
        }
    }
}

impl From<Value> for i64 {
    fn from(vl: Value) -> i64 {
        match vl {
            Value::I64(i) => i,
            _ => panic!("Unwrap value failed"),
        }
    }
}

impl From<Value> for u32 {
    fn from(vl: Value) -> u32 {
        match vl {
            Value::I32(i) => i as u32,
            _ => panic!("Unwrap value failed"),
        }
    }
}

impl From<Value> for u64 {
    fn from(vl: Value) -> u64 {
        match vl {
            Value::I64(i) => i as u64,
            _ => panic!("Unwrap value failed"),
        }
    }
}

impl From<Value> for f32 {
    fn from(vl: Value) -> f32 {
        match vl {
            Value::F32(i) => i,
            _ => panic!("Unwrap value failed"),
        }
    }
}

impl From<Value> for f64 {
    fn from(vl: Value) -> f64 {
        match vl {
            Value::F64(i) => i,
            _ => panic!("Unwrap value failed"),
        }
    }
}

impl From<Value> for bool {
    fn from(vl: Value) -> bool {
        match vl {
            Value::I32(i) => i != 0,
            _ => panic!("Unwrap value failed"),
        }
    }
}

// Grrrr I think these are impossible.  x_X
// impl<'a, T> From<&'a Value> for T where T: From<Value> {
//     fn from(vl: &'a Value) -> T {
//         (*vl).into()
//     }
// }

// impl<'a> From<Value> for &'a T where T: From<Value> {
//     fn from(vl: Value) -> &'a T {
//         &vl.into()
//     }
// }

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

impl From<u32> for Value {
    fn from(num: u32) -> Self {
        Value::I32(num as i32)
    }
}

impl From<u64> for Value {
    fn from(num: u64) -> Self {
        Value::I64(num as i64)
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

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        if b {
            Value::I32(1)
        } else {
            Value::I32(0)
        }
    }
}

/// A trait to say "you can convert X to this type, wrapping numbers".
/// aka, just `as`.
/// Like From but doesn't promise to preserve all data.
/// This should already exist, dammit.
/// Num crate is working on it: <https://github.com/rust-num/num/issues/183>
trait Wrap<T> {
    fn wrap(self) -> T;
}

/// Stolen wholesale from parity-wasm
/// `src/interpreter/value.rs`
macro_rules! impl_wrap_into {
	($from: ident, $into: ident) => {
		impl Wrap<$into> for $from {
			fn wrap(self) -> $into {
				self as $into
			}
		}
	}
}

impl_wrap_into!(i32, i8);
impl_wrap_into!(i32, i16);
impl_wrap_into!(i64, i8);
impl_wrap_into!(i64, i16);
impl_wrap_into!(i64, i32);
impl_wrap_into!(i64, f32);
impl_wrap_into!(u64, f32);

// Misc numerical functions here.

/// fcopysign:
///
/// If z1 and z2 have the same sign, then return z1
/// Else return z1 with negated sign.
fn copysign<T>(f1: T, f2: T) -> T
    where T: num::Float {
    // TODO: Probably doesn't work properly with NaN's?
    if f1.is_sign_positive() && f2.is_sign_positive()
        || f1.is_sign_negative() && f2.is_sign_negative() {
            f1
        } else {
            f1.neg()
        }
}

/// Truncates the float into the given integer type.
fn truncate_to_int<From, To>(f1: From) -> To
    where From: num::Float,
          To: num::NumCast {
    // TODO: Needs more correctness checking!  Inf, NaN, etc.
    // Also not sure NumCast is the right thing to use here but it seems to work.
    To::from(f1.trunc()).unwrap()
}

/// Rounds the float into the given integer type.
fn round_to_int<From, To>(f1: From) -> To
    where From: num::Float,
          To: num::NumCast {
    // TODO: Needs more correctness checking!  Inf, NaN, etc.
    // Also needs to verify that the rounding behavior is correct.
    // Also not sure NumCast is the right thing to use here but it seems to work.
    To::from(f1.round()).unwrap()
}


/// Convert one type to another by extending with leading zeroes
/// or one's (depending on destination type)
pub trait Extend<T> {
	/// Convert one type to another by extending with leading zeroes.
	fn extend(self) -> T;
}

/// Also stolen from parity-wasm
macro_rules! impl_extend_into {
	($from: ident, $into: ident) => {
		impl Extend<$into> for $from {
			fn extend(self) -> $into {
				self as $into
			}
		}
	}
}

impl_extend_into!(i8, i32);
impl_extend_into!(u8, i32);
impl_extend_into!(i16, i32);
impl_extend_into!(u16, i32);
impl_extend_into!(i8, i64);
impl_extend_into!(u8, i64);
impl_extend_into!(i16, i64);
impl_extend_into!(u16, i64);
impl_extend_into!(i32, i64);
impl_extend_into!(u32, i64);
impl_extend_into!(u32, u64);
impl_extend_into!(i32, f32);
impl_extend_into!(i32, f64);
impl_extend_into!(u32, f32);
impl_extend_into!(u32, f64);
impl_extend_into!(i64, f64);
impl_extend_into!(u64, f64);
impl_extend_into!(f32, f64);

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
    let num_local_slots = locals.iter().map(|x| x.count() as usize).sum();
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
    ///
    /// BUGGO: Table values are allowed to be uninitialized, apparently.
    fn fill(&mut self, size: u32) {
        self.data.resize(size as usize, FuncIdx(std::usize::MAX));
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
    const PAGE_SIZE: usize = 65_536;

    pub fn new(size: Option<u32>) -> Self {
        let mut mem = Self {
            data: vec![],
            max: None,
        };
        if let Some(size) = size {
            // TODO: Should always be true; use one of num's try_convert
            // methods or such instead.
            assert!(size < std::i32::MAX as u32);
            mem.resize(size as i32);
        }
        mem
    }

    /// The length of the allocated storage, in pages.
    fn len(&self) -> u32 {
        (self.data.len() / Self::PAGE_SIZE) as u32
    }

    /// Resizes the memory by the given delta, in units of `Memory::PAGE_SIZE`.
    /// That is, if delta is positive, the memory will grow, if negative it will shrink.
    /// Newly allocated memory is zero'd.
    fn resize(&mut self, delta: i32) {
        use std::usize;
        let delta_bytes = i32::checked_mul(Self::PAGE_SIZE as i32, delta)
            .expect("Asked for more memory than can fit in an i32?");
        // This assert should always be true if we only ever allocate mem with
        // this function, buuuuuut...
        assert!(self.data.len() < std::i32::MAX as usize);
        let new_size = self.data.len() as i32 + delta_bytes;
        self.data.resize(new_size as usize, 0);
        // BUGGO: Augh, the max size semantics here are awful, fix them.
        //self.max = Some(size);
    }
}

/// An index into a module's `Globals` vector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalIdx(usize);


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
            let functypes = types.types().iter().map(From::from);
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
            let converted_funcs = code.bodies().iter().zip(functions.entries()).map(|(c, f)| {
                // Make sure the function signature is a valid type.
                let type_idx = f.type_ref() as usize;
                assert!(
                    type_idx < types.len(),
                    "Function refers to a type signature that does not exist!"
                );

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
            assert!(
                table.entries().len() < 2,
                "More than one memory entry, should never happen!"
            );
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
            assert!(
                memory.entries().len() < 2,
                "More than one memory entry, should never happen!"
            );
            if let Some(memory) = memory.entries().iter().next() {
                // TODO: As far as I can tell, the memory's minimum size is never used?
                let _min = memory.limits().initial();
                let max = memory.limits().maximum();

                // TODO: It's apparently valid for a memory to have no max size?
                if let Some(max) = max {
                    m.mem.resize(max as i32);
                }
            }

            if let Some(data) = module.data_section() {
                // TODO
                unimplemented!();
            }
        }

        // Allocate globals
        if let Some(globals) = module.global_section() {
            let global_iter = globals.entries().iter().map(|global| {
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
            assert!(
                start < m.funcs.len(),
                "Start section references a non-existent function!"
            );
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

/// A label to a particular block: just an instruction index.
/// The label index is implicit in the labels stack; label 0 is always
/// the top of the stack.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BlockLabel(usize);

/// The activation record for an executing function.
#[derive(Debug, Clone, Default)]
pub struct StackFrame {
    value_stack: Vec<Value>,
    labels: Vec<BlockLabel>,
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
        let iter = functype.params.iter().map(|t| Value::default_from_type(*t));
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
        let iter = func.functype
            .params
            .iter()
            .map(|t| Value::default_from_type(*t));
        locals.extend(iter);

        Self {
            value_stack: vec![],
            labels: vec![],
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
    /// Panics if the stack is empty or the Value is not the right
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FunctionAddress(usize);
/// Table address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TableAddress(usize);
/// Memory address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryAddress(usize);
/// Global address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalAddress(usize);
/// Module instance address type
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleAddress(usize);


/// For forward jumps (if, block) we need to know where to jump TO.
/// Serialized wasm doesn't store this information explicitly,
/// and searching for it mid-execution is a wasteful PITA,
/// so we find it ahead of time and then store it when the
/// function is instantiated.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct JumpTarget {
    block_start_instruction: usize,
    block_end_instruction: usize,
    /// Only used for if/else statements.
    else_instruction: usize,
}

/// Contains all the information needed to execute a function.
#[derive(Debug, Clone)]
pub struct FuncInstance {
    functype: FuncType,
    locals: Vec<elements::ValueType>,
    body: Vec<elements::Opcode>,
    module: ModuleAddress,
    /// A vec of jump targets sorted by source instruction,
    /// so we can just binary-search in it.  A HashMap would
    /// work too, but I suspect this is faster?  And is trivial
    /// to construct, so.
    jump_table: Vec<JumpTarget>,
}

impl FuncInstance {
    /// Iterate through a function's body and construct the jump table for it.
    /// If we find a block instruction, the target is the matching end instruction.
    ///
    /// Panics on invalid (improperly nested) blocks.
    fn compute_jump_table(body: &[elements::Opcode]) -> Vec<JumpTarget> {
        use elements::Opcode::*;
        // TODO: I would be sort of happier properly walking a sequence, OCaml
        // style, but oh well.
        let mut offset = 0;
        let mut accm = vec![];
        while offset < body.len() {
            let op = &body[offset];
            println!("Computing jump table: {}, {:?}", offset, op);
            match *op {
                Block(_) => {
                    offset = FuncInstance::find_block_close(body, offset, &mut accm);
                }
                If(_) => {
                    offset = FuncInstance::find_block_close(body, offset, &mut accm);
                }
                _ => (),
            }
            offset += 1;
        }
        accm
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
        use elements::Opcode::*;
        use std::usize;
        let mut offset = start_offset;
        // TODO: Potentially invalid here, but, okay.
        let mut else_offset = usize::MAX;
        loop {
            let op = &body[offset];
            match *op {
                elements::Opcode::End => {
                    // Found matching end, yay.
                    let jt = JumpTarget {
                        block_start_instruction: start_offset,
                        block_end_instruction: offset,
                        else_instruction: else_offset,
                    };
                    accm.push(jt);
                    return offset;
                }
                elements::Opcode::Else => {
                    else_offset = offset;
                }
                // Opening another block, recurse.
                elements::Opcode::Block(_) => {
                    offset = FuncInstance::find_block_close(body, offset, accm);
                }
                _ => (),
            }
            offset += 1;
            assert!(offset < body.len(), "Unclosed block, should never happen!");
        }
        unreachable!();
    }
}

/// Relates all the local indices to globals, functions etc.
/// from within a module to the global addresses of the Store.
#[derive(Debug, Clone)]
pub struct ModuleInstance {
    types: Vec<FuncType>,
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
    pub fn new() -> Self {
        Self {
            store: Store::default(),
            state: State::default(),
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
    fn resolve_function(state: &State, module_addr: ModuleAddress, idx: FuncIdx) -> FunctionAddress {
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
    fn set_memory_with<F, N>(mems: &mut [Memory], state: &State, module_addr: ModuleAddress, offset: usize, f: F, vl: N) where F: Fn(&mut [u8], N) {
        let memory_address = Interpreter::resolve_memory(state, module_addr);
        let mem = &mut mems[memory_address.0];
        assert!(offset + std::mem::size_of::<N>() < mem.data.len());
        f(&mut mem.data[offset..], vl)
    }

    /// Reads data from a slice of the given `memory` with the given function
    fn get_memory_with<F, N>(mems: &[Memory], state: &State, module_addr: ModuleAddress, offset: usize, f: F) -> N where F: Fn(&[u8]) -> N {
        let memory_address = Interpreter::resolve_memory(state, module_addr);
        let mem = &mems[memory_address.0];
        assert!(offset + std::mem::size_of::<N>() < mem.data.len());
        f(&mem.data[offset..])
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
    pub fn with_module(mut self, module: LoadedModule) -> Self {
        assert!(module.validated);
        let module_instance_address = ModuleAddress(self.state.module_instances.len());
        let mut functions = vec![];
        let table = None;
        let memory = None;
        let globals = vec![];
        let types = module.types.clone();
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
            println!("Function: {:?}", instance);
            self.state.funcs.push(instance);
            functions.push(address);
        }
        // TODO: table, memory, globals
        // all also need to come from the module!
        let inst = ModuleInstance {
            types,
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

    fn exec_const(frame: &mut StackFrame, vl: Value) {
        frame.push(vl);
    }

    /// Executes a load instruction, using the given function to 
    /// convert the memory's `&[u8]` into the given Value type.
    fn exec_load<F, N>(frame: &mut StackFrame, store: &mut Store, state: &State, module: ModuleAddress, offset: u32, func: F)
        where F: Fn(&[u8]) -> N,
              N: Into<Value> {
        let address = frame.pop_as::<i32>();
        let effective_address = address + offset as i32;
        let mem_contents = Interpreter::get_memory_with(
            &mut store.mems, &state, module, 
            effective_address as usize, 
            func).into();
        frame.push(mem_contents);
    }


    /// Executes a load instruction, using the given function to 
    /// convert the memory's `&[u8]` into the the SourceN type,
    /// then sign-extending it (based on whether it's signed or unsigned)
    /// into DestN.
    fn exec_load_extend<F, SourceN, DestN>(frame: &mut StackFrame, store: &mut Store, state: &State, module: ModuleAddress, offset: u32, func: F)
        where F: Fn(&[u8]) -> SourceN,
              SourceN: Extend<DestN>,
              DestN: Into<Value> {
        let address = frame.pop_as::<i32>();
        let effective_address = address + offset as i32;
        let mem_contents = Interpreter::get_memory_with(
            &mut store.mems, &state, module, 
            effective_address as usize, 
            func).extend().into();
        frame.push(mem_contents);
    }


    /// Executes a store instruction, using the given function to 
    /// write the Value type into the memory's `&mut [u8]`
    fn exec_store<F, N>(frame: &mut StackFrame, store: &mut Store, state: &State, module: ModuleAddress, offset: u32, func: F)
        where F: Fn(&mut [u8], N),
              N: From<Value> {
        let vl = frame.pop_as::<N>();
        let address = frame.pop_as::<i32>();
        let effective_address = address + offset as i32;
        Interpreter::set_memory_with(
            &mut store.mems, &state, module, 
            effective_address as usize, 
            func, vl);
    }

    /// Wraps/truncates the the Value on the stack from the given SourceN type
    /// to the DestN type, then stores it in memory.
    fn exec_store_wrap<F, SourceN, DestN>(frame: &mut StackFrame, store: &mut Store, state: &State, module: ModuleAddress, offset: u32, func: F)
        where F: Fn(&mut [u8], DestN),
              SourceN: From<Value> + Wrap<DestN> {
        let vl: DestN = frame.pop_as::<SourceN>().wrap();
        let address = frame.pop_as::<i32>();
        let effective_address = address + offset as i32;
        Interpreter::set_memory_with(
            &mut store.mems, &state, module, 
            effective_address as usize, 
            func, vl);
    }

    /// Helper function for running binary operations that pop
    /// two values from the stack and push one result
    fn exec_binop<T1, T2, Res, F>(frame: &mut StackFrame, op: F)
        where T1: From<Value>,
              T2: From<Value>,
              Res: Into<Value>,
              F: Fn(T1, T2) -> Res
     {
        let (a, b) = frame.pop2_as::<T1, T2>();
        frame.push(op(a, b).into());
    }

    /// Helper function for running binary operations that pop
    /// two values from the stack and push one result
    fn exec_uniop<T, Res, F>(frame: &mut StackFrame, op: F)
        where T: From<Value>,
              Res: Into<Value>,
              F: Fn(T) -> Res
    {
        let a = frame.pop_as::<T>();
        frame.push(op(a).into());
    }

    /// Helper function for running a function call.
    fn exec_call(frame: &mut StackFrame, store: &mut Store, state: &State, function_addr: FunctionAddress) {
        // Typecheck and get appropriate arguments off the stack to pass
        // to the called function.
        let f = &state.funcs[function_addr.0];
        let return_val = {
            assert!(f.functype.params.len() <= frame.value_stack.len());
            let params_end = frame.value_stack.len() - 1;
            let params_start = params_end - f.functype.params.len();
            let params_slice = &frame.value_stack[params_start..params_end];
            for (param, desired_type) in params_slice.iter().zip(&f.functype.params) {
                assert_eq!(param.get_type(), *desired_type);
            }
            // Some(Value::I32(3))
            Interpreter::exec(store, state, function_addr, params_slice)
        };
        
        // Great, now check that the return value matches the stated
        // return type, and push it to the values stack.
        let return_type = return_val.map(|v| v.get_type());
        assert_eq!(return_type, f.functype.return_type);
        if let Some(v) = return_val {
            frame.value_stack.push(v);
        }
    }

    /// Actually do the interpretation of the given function, assuming
    /// that a stack frame already exists for it with args and locals
    /// and such
    pub fn exec(
        store: &mut Store,
        state: &State,
        func: FunctionAddress,
        args: &[Value],
    ) -> Option<Value> {
        let func = &state.funcs[func.0];
        let frame = &mut StackFrame::from_func_instance(func, args);
        use elements::Opcode::*;
        use std::usize;
        loop {
            if frame.ip == func.body.len() {
                break;
            }
            let op = &func.body[frame.ip];

            println!("Frame: {:?}", frame);
            println!("Op: {:?}", op);
            match *op {
                Unreachable => panic!("Unreachable?"),
                Nop => (),
                Block(blocktype) => {
                    let jump_target_idx = func.jump_table
                        .binary_search_by(|jt| jt.block_start_instruction.cmp(&frame.ip))
                        .expect("Cannot find matching jump table for block statement");
                    let jump_target = &func.jump_table[jump_target_idx];
                    frame.push_label(BlockLabel(jump_target.block_end_instruction));
                }
                Loop(blocktype) => {
                    // Instruction index to jump to on branch or such.
                    let end_idx = frame.ip + 1;
                    frame.push_label(BlockLabel(end_idx));
                }
                If(blocktype) => {
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
                    // TODO: This is probably incorrect.
                    frame.pop_label(0);
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
                },
                Call(i) => {
                    let i = i as usize;
                    let function_addr = Interpreter::resolve_function(state, func.module, FuncIdx(i));
                    Interpreter::exec_call(frame, store, state, function_addr);
                }
                CallIndirect(x, _) => {
                    // Okay, x is the expected type signature of the function we
                    // are trying to call.
                    // So we pop an i32 i from the stack, use that to index into
                    // the table to get a function index, then call that
                    // function
                    let x = x as usize;
                    let func_type = Interpreter::resolve_type(state, func.module, TypeIdx(x));
                    let i = frame.pop_as::<u32>() as usize;
                    let table_addr = Interpreter::resolve_table(state, func.module);
                    let function_index = {
                        let table = &store.tables[table_addr.0];
                        table.data[i]
                    };
                    let function_addr = Interpreter::resolve_function(state, func.module, function_index);
                    // Make sure that the function we've actually retrieved has the same signature as the
                    // type we want.
                    assert_eq!(&state.funcs[function_addr.0].functype, func_type);
                    Interpreter::exec_call(frame, store, state, function_addr);
                }
                Drop => {
                    frame.pop();
                }
                Select => (),
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
                    let vl = Interpreter::get_global(&store.globals, &state, func.module, GlobalIdx(i));
                    frame.push(vl);
                }
                SetGlobal(i) => {
                    let i = i as usize;
                    let vl = frame.pop();
                    Interpreter::set_global(&mut store.globals, &state, func.module, GlobalIdx(i), vl);
                }
                I32Load(offset, _align) => {
                    Interpreter::exec_load(frame, store, state, func.module, offset, byteorder::LittleEndian::read_i32);
                },
                I64Load(offset, _align) => {
                    Interpreter::exec_load(frame, store, state, func.module, offset, byteorder::LittleEndian::read_i64);
                },
                F32Load(offset, _align) => {
                    Interpreter::exec_load(frame, store, state, func.module, offset, byteorder::LittleEndian::read_f32);
                },
                F64Load(offset, _align) => {
                    Interpreter::exec_load(frame, store, state, func.module, offset, byteorder::LittleEndian::read_f64);
                },
                I32Load8S(offset, _align) => {
                    Interpreter::exec_load_extend::<_, i8, i32>(frame, store, state, func.module, offset, |mem| mem[0] as i8);
                },
                I32Load8U(offset, _align) => {
                    Interpreter::exec_load_extend::<_, u8, i32>(frame, store, state, func.module, offset, |mem| mem[0] as u8);
                },
                I32Load16S(offset, _align) => {
                    Interpreter::exec_load_extend::<_, i16, i32>(frame, store, state, func.module, offset, byteorder::LittleEndian::read_i16);
                },
                I32Load16U(offset, _align) => {
                    Interpreter::exec_load_extend::<_, u16, i32>(frame, store, state, func.module, offset, byteorder::LittleEndian::read_u16);
                },
                I64Load8S(offset, _align) => {
                    Interpreter::exec_load_extend::<_, i8, i64>(frame, store, state, func.module, offset, |mem| mem[0] as i8);
                },
                I64Load8U(offset, _align) => {
                    Interpreter::exec_load_extend::<_, u8, i64>(frame, store, state, func.module, offset, |mem| mem[0] as u8);
                },
                I64Load16S(offset, _align) => {
                    Interpreter::exec_load_extend::<_, i16, i64>(frame, store, state, func.module, offset, byteorder::LittleEndian::read_i16);
                },
                I64Load16U(offset, _align) => {
                    Interpreter::exec_load_extend::<_, u16, i64>(frame, store, state, func.module, offset, byteorder::LittleEndian::read_u16);
                },
                I64Load32S(offset, _align) => {
                    Interpreter::exec_load_extend::<_, i32, i64>(frame, store, state, func.module, offset, byteorder::LittleEndian::read_i32);
                },
                I64Load32U(offset, _align) => {
                    Interpreter::exec_load_extend::<_, u32, i64>(frame, store, state, func.module, offset, byteorder::LittleEndian::read_u32);
                },
                I32Store(offset, _align) => {
                    Interpreter::exec_store(frame, store, state, func.module, offset, byteorder::LittleEndian::write_i32);
                },
                I64Store(offset, _align) => {
                    Interpreter::exec_store(frame, store, state, func.module, offset, byteorder::LittleEndian::write_i64);
                },
                F32Store(offset, _align) => {
                    Interpreter::exec_store(frame, store, state, func.module, offset, byteorder::LittleEndian::write_f32);
                },
                F64Store(offset, _align) => {
                    Interpreter::exec_store(frame, store, state, func.module, offset, byteorder::LittleEndian::write_f64);
                },
                I32Store8(offset, _align) => {
                    // `byteorder` doesn't have write_i8 since it's a bit redundant,
                    // so we make our own.
                    Interpreter::exec_store_wrap::<_, i32, i8>(frame, store, state, func.module, offset, |mem, x| mem[0] = x as u8);
                },
                I32Store16(offset, _align) => {
                    Interpreter::exec_store_wrap::<_, i32, i16>(frame, store, state, func.module, offset, byteorder::LittleEndian::write_i16);
                },
                I64Store8(offset, _align) => {
                    Interpreter::exec_store_wrap::<_, i64, i8>(frame, store, state, func.module, offset, |mem, x| mem[0] = x as u8);
                },
                I64Store16(offset, _align) => {
                    Interpreter::exec_store_wrap::<_, i64, i16>(frame, store, state, func.module, offset, byteorder::LittleEndian::write_i16);
                },
                I64Store32(offset, _align) => {
                    Interpreter::exec_store_wrap::<_, i64, i32>(frame, store, state, func.module, offset, byteorder::LittleEndian::write_i32);
                },
                CurrentMemory(_) => {
                    let module_addr = func.module;
                    let memory_addr = Interpreter::resolve_memory(state, module_addr);
                    let mem = &store.mems[memory_addr.0];
                    frame.push(mem.len().into())
                },
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
                I32Const(i) => {
                    Interpreter::exec_const(frame, i.into())
                }
                I64Const(l) => {
                    Interpreter::exec_const(frame, l.into())
                }
                // Why oh why are these floats represented as u32 and u64?
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
                    Interpreter::exec_const(frame, Value::from(f32::from_bits(i)));
                }
                F64Const(l) => {
                    use std::f64;
                    Interpreter::exec_const(frame, Value::from(f64::from_bits(l)));
                }
                I32Eqz => {
                    Interpreter::exec_uniop::<i32, bool, _>(frame, |x| i32::eq(&x, &0));
                }
                I32Eq => {
                    Interpreter::exec_binop(frame, |x, y| i32::eq(&x, &y));
                }
                I32Ne => {
                    Interpreter::exec_binop(frame, |x, y| i32::ne(&x, &y));
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
                    Interpreter::exec_binop(frame, |x, y| i64::eq(&x, &y));
                }
                I64Ne => {
                    Interpreter::exec_binop(frame, |x, y| i64::ne(&x, &y));
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
                    Interpreter::exec_binop(frame, |x, y| f32::eq(&x, &y));
                }
                F32Ne => {
                    Interpreter::exec_binop(frame, |x, y| f32::ne(&x, &y));
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
                    Interpreter::exec_binop(frame, |x, y| f64::eq(&x, &y));
                }
                F64Ne => {
                    Interpreter::exec_binop(frame, |x, y| f64::ne(&x, &y));
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
                },
                I64ShrU => {
                    Interpreter::exec_binop::<u64, u32, _, _>(frame, u64::wrapping_shr);
                },
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
                    use std::ops::*;
                    Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::min);
                }
                F32Max => {
                    use std::ops::*;
                    Interpreter::exec_binop::<f32, f32, _, _>(frame, f32::max);
                }
                F32Copysign => {
                    Interpreter::exec_binop::<f32, f32, _, _>(frame, copysign);
                },
                F64Abs => {
                    use std::ops::*;
                    Interpreter::exec_uniop::<f64, _, _>(frame, f64::abs);
                }
                F64Neg => {
                    use std::ops::Neg;
                    Interpreter::exec_uniop::<f64, _, _>(frame, Neg::neg);
                }
                F64Ceil => {
                    use std::ops::*;
                    Interpreter::exec_uniop::<f64, _, _>(frame, f64::ceil);
                }
                F64Floor => {
                    use std::ops::*;
                    Interpreter::exec_uniop::<f64, _, _>(frame, f64::floor);
                }
                F64Trunc => {
                    use std::ops::*;
                    Interpreter::exec_uniop::<f64, _, _>(frame, f64::trunc);
                }
                F64Nearest => {
                    // TODO: Double-check rounding behavior is correct
                    Interpreter::exec_uniop::<f64, _, _>(frame, f64::round);
                }
                F64Sqrt => {
                    use std::ops::*;
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
                    use std::ops::*;
                    Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::min);
                }
                F64Max => {
                    use std::ops::*;
                    Interpreter::exec_binop::<f64, f64, _, _>(frame, f64::max);
                }
                F64Copysign => {
                    Interpreter::exec_binop::<f64, f64, _, _>(frame, copysign);
                },
                I32WrapI64 => {
                    Interpreter::exec_uniop::<i64, i32, _>(frame, Wrap::wrap);
                },
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
                },
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
                },
                I32ReinterpretF32 => {
                    // TODO: Check that this is going the correct direction,
                    // i32 -> f32
                    Interpreter::exec_uniop(frame, f32::from_bits);
                }
                I64ReinterpretF64 => {
                    // TODO: Check that this is going the correct direction,
                    // i64 -> f64
                    Interpreter::exec_uniop(frame, f64::from_bits);
                },
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
        // Return the function's return value (if any).
        // TODO: We should check that the value matches the function's stated
        // type and arity.
        frame.value_stack.last().cloned()
    }

    /// A nice shortcut to run `exec()` with appropriate values.
    pub fn run(&mut self, func: FunctionAddress, args: &[Value]) -> Option<Value> {
        let state = &self.state;
        let store = &mut self.store;
        Interpreter::exec(store, state, func, args)
    }
}
