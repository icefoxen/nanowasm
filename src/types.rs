//! Common traits and type definitions used everywhere in nanowasm.

use std;
use std::rc::Rc;

use parity_wasm::elements;
use num::ToPrimitive;

/// This should be in std by now dammit!
pub trait TryFrom<T>
where
    Self: Sized,
{
    type Error;
    fn try_from(value: T) -> Result<Self, Self::Error>;
}

/// A type signature for a function type, intended to
/// go into the `types` section of a module.
///
/// parity-wasm has `elements::FunctionType` which is basically
/// this but with some extra serialization info we don't
/// need for execution, so we make our own.
#[derive(Debug, Clone, PartialEq)]
pub struct FuncType {
    pub params: Vec<elements::ValueType>,
    pub return_type: Option<elements::ValueType>,
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
pub struct TypeIdx(pub usize);

/// An index into a module's `function` vector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FuncIdx(pub usize);

/// An index into a module's `globals` vector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalIdx(pub usize);

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
    pub fn default_from_type(t: elements::ValueType) -> Self {
        match t {
            elements::ValueType::I32 => Value::I32(0),
            elements::ValueType::I64 => Value::I64(0),
            elements::ValueType::F32 => Value::F32(0.0),
            elements::ValueType::F64 => Value::F64(0.0),
        }
    }

    /// Get the type of the value.
    pub fn get_type(self) -> elements::ValueType {
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
/// TODO: This now exists, see https://docs.rs/num/0.1.42/num/cast/trait.AsPrimitive.html
pub trait Wrap<T> {
    fn wrap(self) -> T;
}

/// Stolen wholesale from parity-wasm
/// `src/interpreter/value.rs`
///
/// Just implements the `Wrap` trait for a given numeric type.
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

/// Convert one type to another by extending with leading zeroes
/// or one's (depending on destination type)
/// TODO: num's AsPrimitive trait also covers this case.
pub trait Extend<T> {
    /// Convert one type to another by extending with leading zeroes.
    fn extend(self) -> T;
}

/// Implements the `Extend` trait for a given numeric type.
///
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

#[derive(Clone)]
pub enum FuncBody {
    Opcodes(Vec<elements::Opcode>),
    /// A function implemented in Rust.  Currently it is very restricted;
    /// it can basically only affect the operation stack.
    /// This should be expanded but needs a bit of refactoring to do so,
    /// to expose more of the interpreter innards in a nice way (call
    /// frame, etc).
    HostFunction(Rc<Fn(&mut Vec<Value>)>),
}

use std::fmt;
impl fmt::Debug for FuncBody {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> fmt::Result {
        match *self {
            FuncBody::Opcodes(ref o) => write!(f, "Opcodes({:?})", o),
            FuncBody::HostFunction(_) => write!(f, "HostFunction(&mut Vec<Value>)"),
        }
    }
}

/// A runtime function able to be executed in the context of a module.
#[derive(Debug, Clone)]
pub struct Func {
    pub typeidx: TypeIdx,
    pub locals: Vec<elements::ValueType>,
    pub body: FuncBody,
}

/// A table.
///
/// Currently, a table is *purely* a mapping of
/// integers to anyfunc's.
///
/// Obviously mainly there for integration with Javascript,
/// though it has other uses too.
#[derive(Debug, Clone)]
pub struct Table {
    /// Actual data
    pub data: Vec<FuncIdx>,
    /// Maximum size
    pub max: Option<u32>,
}

impl Table {
    pub fn new() -> Self {
        Self {
            data: vec![],
            max: None,
        }
    }

    /// Resizes the underlying table, filling it with
    /// `FuncIdx(usize::MAX)` in the process.  Even
    /// in the (likely) case that there IS no function `usize::MAX`.
    ///
    /// BUGGO: Table values are allowed to be uninitialized, apparently.
    /// Cannot contain invalid values, apparently.
    pub fn fill(&mut self, size: u32) {
        self.data.resize(size as usize, FuncIdx(std::usize::MAX));
        self.max = Some(size);
    }

    pub fn initialize(&mut self, inits: &[(ConstExpr, Vec<FuncIdx>)]) -> Result<(), ()> {
        // TODO
        Ok(())
    }
}

/// A structure containing a memory space.
#[derive(Debug, Clone)]
pub struct Memory {
    /// Actual data
    pub data: Vec<u8>,
    /// Maximum size, in units of 65,536 byte pages (`Memory::PAGE_SIZE`).
    pub max: Option<u32>,
}

impl Memory {
    const PAGE_SIZE: usize = 65_536;

    pub fn new(size: Option<u32>) -> Self {
        let mut mem = Self {
            data: vec![],
            max: None,
        };
        if let Some(size) = size {
            let size_i = size.to_i32().expect(
                "Should never happen; 32-bit wasm should always have memory sizes << i32::MAX",
            );
            mem.resize(size_i);
        }
        mem
    }

    /// The length of the allocated storage, in pages.
    pub fn len(&self) -> u32 {
        (self.data.len() / Self::PAGE_SIZE)
            .to_u32()
            .expect("Page count of memory > u32::MAX; should never happen!")
    }

    /// Resizes the memory by the given delta, in units of `Memory::PAGE_SIZE`.
    /// That is, if delta is positive, the memory will grow, if negative it will shrink.
    /// Newly allocated memory is zero'd.
    pub fn resize(&mut self, delta: i32) {
        use std::usize;
        let delta_bytes = i32::checked_mul(Self::PAGE_SIZE as i32, delta)
            .expect("Asked for more memory than can fit in an i32?");
        // This assert should always be true if we only ever allocate mem with
        // this function, buuuuuut...
        // TODO: This check can probably be better, and see if we can get rid some of
        // the bloody `as` conversions too.
        assert!(self.data.len() < std::i32::MAX as usize);
        let new_size = self.data.len() as i32 + delta_bytes;
        self.data.resize(new_size as usize, 0);
        // BUGGO: Augh, the max size semantics here are awful, fix them.
        //self.max = Some(size);
    }

    pub fn initialize(&mut self, offset: u32, val: &[u8]) -> Result<(), ()> {
        let offset_start = offset as usize;
        let offset_end = offset_start + val.len();
        assert!(offset_end <= self.data.len());
        self.data.as_mut_slice()[offset_start..offset_end].copy_from_slice(val);
        Ok(())
    }
}

/// A restricted subset of `parity::elements::Opcode` that is
/// valid in a constant expression (ie, data initializers)
///
/// Defined at <https://webassembly.github.io/spec/core/valid/instructions.html#valid-constant>
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ConstOpcode {
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
    // BUGGO: Allowing get_global in a const expr seems VERY SUSPICIOUS,
    // since globals are supposed to be *initialized* in a const expr.
    // According to wat2wasm `get_global` in an initializer can only
    // reference globals imported by the module.
    // Which is perfectly reasonable but somewhat surprising.
    // Deal with this eventually.
    GetGlobal(u32),
}

impl TryFrom<elements::Opcode> for ConstOpcode {
    type Error = elements::Opcode;
    fn try_from(opcode: elements::Opcode) -> Result<ConstOpcode, elements::Opcode> {
        match opcode {
            elements::Opcode::I32Const(i) => Ok(ConstOpcode::I32Const(i as i32)),
            elements::Opcode::I64Const(i) => Ok(ConstOpcode::I64Const(i as i64)),
            elements::Opcode::F32Const(f) => Ok(ConstOpcode::F32Const(f32::from_bits(f))),
            elements::Opcode::F64Const(f) => Ok(ConstOpcode::F64Const(f64::from_bits(f))),
            elements::Opcode::GetGlobal(i) => Ok(ConstOpcode::GetGlobal(i)),
            op => Err(op),
        }
    }
}

/// A constant expression; a list of `ConstOpcode`'s.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstExpr(pub Vec<ConstOpcode>);

impl<'a> TryFrom<&'a [elements::Opcode]> for ConstExpr {
    type Error = elements::Opcode;
    fn try_from(opcodes: &[elements::Opcode]) -> Result<ConstExpr, elements::Opcode> {
        // Filter out `End` opcodes which are valid but uninteresting,
        // then try to turn into ConstOpcode and collect into Result<Vec<...>, ...>
        // TODO: Actually validate we end with `End` someday?
        // peekable() might make that easier.
        let ops = opcodes
            .iter()
            .filter(|op| **op != elements::Opcode::End)
            .cloned()
            .map(ConstOpcode::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ConstExpr(ops))
    }
}

/// A structure containing runtime data for a global variable.
#[derive(Debug, Clone)]
pub struct Global {
    pub mutable: bool,
    pub variable_type: elements::ValueType,
    pub value: Value,
}

impl Global {
    pub fn initialize(&mut self, init_value: Value) {
        self.value = init_value;
    }
}

/// An import declaration.  `T` is generally some sort of type
/// that says what the type and value of the import is.
#[derive(Debug, Clone)]
pub struct Import<T> {
    pub module_name: String,
    pub field_name: String,
    pub value: T,
}

/// An export declaration.
#[derive(Debug, Clone)]
pub struct Export<T> {
    pub name: String,
    pub value: T,
}

/// Error type using the `failure` crate.
#[derive(Debug, Fail)]
pub enum Error {
    /// Tried to import a module that does not exist.
    #[fail(display = "Module {} does not exist, desired by module {}!", module, dependent_module)]
    ModuleNotFound {
        module: String,
        dependent_module: String,
    },
    /// Tried to import a value but the module does not export it.
    #[fail(display = "Module {} does not export value {} of expected type {} imported by module {}", module, name, typ, dependent_module)]
    NotExported {
        module: String,
        name: String,
        typ: String,
        dependent_module: String,
    },
    /// Validation failed, for whatever reason.
    #[fail(display = "Module {} is invalid: {}", module, reason)]
    Invalid {
        module: String,
        reason: String,
    },
    /// Version mismatch in a module.
    #[fail(display = "Module {} has the wrong version; expected {}, got {}", module, expected, got)]
    VersionMismatch {
        module: String,
        expected: u32,
        got: u32,
    },
    /// Module cannot be loaded because it is malformed: multiple tables,
    /// indexes to non-existent elements, etc.
    /// Sort of a catch-all for all sorts of edge-cases that a properly-assembled module
    /// should not have.
    #[fail(display = "Module {} is malformed: {}", module, message)]
    Malformed {
        module: String,
        message: String
    },
}
