use parity_wasm;
use parity_wasm::builder;
use parity_wasm::elements::{self, Module, Opcode};
use types::*;
use loader::*;
use interpreter::*;

extern crate wabt;

macro_rules! load_module {
    ($name: expr) => ({
        let input_wat = include_str!($name);
        let module_bytes = wabt::wat2wasm(input_wat).unwrap();
        let module = parity_wasm::deserialize_buffer(&module_bytes).unwrap();
        module
    })
}

lazy_static! {
    static ref INC: Module = load_module!("../test_programs/inc.wat");
    static ref FIB: Module = load_module!("../test_programs/fib.wat");
}

#[test]
fn test_create() {
    let loaded_module = LoadedModule::new("inc", INC.clone()).unwrap();
    let validated_module = loaded_module.validate();
    let mut interp = Interpreter::new().with_module(validated_module);
    println!("{:#?}", interp);
    // assert!(false);
}

#[test]
fn test_create_fib() {
    let mut loaded_module = LoadedModule::new("fib", FIB.clone()).unwrap();
    let validated_module = loaded_module.validate();
    let interp = Interpreter::new().with_module(validated_module);
    println!("{:#?}", interp);
    // assert!(false);
}

#[test]
fn test_run_fib() {
    let mut loaded_module = LoadedModule::new("fib", FIB.clone()).unwrap();
    let validated_module = loaded_module.validate();
    let mut interp = Interpreter::new().with_module(validated_module).unwrap();

    interp.run(FunctionAddress(1), &vec![Value::I32(30)]);
    assert!(false);
}

/// Helper function to run a small program with some
/// inputs and compare the results to the given desired outputs.
/// Constraints: Makes a single function named "testfunc" with a single output,
/// no locals or memories or such...
fn test_stack_program(program: &[Opcode], args: &[Value], desired_output: Option<Value>) {
    let input_types = args.iter().map(|v| v.get_type()).collect::<Vec<_>>();
    println!("Input types: {:?}", input_types);
    println!("Args: {:?}", args);
    let output_type = desired_output.map(|v| v.get_type());
    // Args get put into local variables, not the stack, so for convenience
    // we load them onto the stack for you.
    let mut init_ops: Vec<Opcode> = args.iter()
        .enumerate()
        .map(|(i, _arg)| Opcode::GetLocal(i as u32))
        .collect();
    init_ops.extend(program.iter().cloned());
    let module = builder::ModuleBuilder::new()
        .function()
        .signature()
        .with_params(input_types)
        .with_return_type(output_type)
        .build()
        .body()
        .with_opcodes(elements::Opcodes::new(init_ops))
        .build()
        .build()
        .build();
    let mut loaded = LoadedModule::new("test", module).unwrap();
    let validated_module = loaded.validate();
    let mut interp = Interpreter::new().with_module(validated_module).unwrap();
    let run_result = interp.run(FunctionAddress(0), args);
    assert_eq!(run_result, desired_output)
}

#[test]
fn test_i32_add() {
    test_stack_program(
        &vec![Opcode::I32Add],
        &vec![Value::I32(1), Value::I32(2)],
        Some(Value::I32(3)),
    );
}

#[test]
fn test_i64_add() {
    test_stack_program(
        &vec![Opcode::I64Add],
        &vec![Value::I64(-1), Value::I64(99)],
        Some(Value::I64(98)),
    );
}

#[test]
fn test_i32_sub() {
    test_stack_program(
        &vec![Opcode::I32Sub],
        &vec![Value::I32(1), Value::I32(2)],
        Some(Value::I32(-1)),
    );
}

#[test]
fn test_i64_sub() {
    test_stack_program(
        &vec![Opcode::I64Sub],
        &vec![Value::I64(-1), Value::I64(99)],
        Some(Value::I64(-100)),
    );
}
