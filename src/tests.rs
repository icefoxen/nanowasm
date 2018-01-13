use parity_wasm;
use parity_wasm::builder;
use parity_wasm::elements::{self, Opcode};
use super::*;

/// Make sure that adding a non-validated module to a program fails.
#[test]
#[should_panic]
fn test_validate_failure() {
    let input_file = "test_programs/inc.wasm";
    let module = parity_wasm::deserialize_file(input_file).unwrap();
    let mut mod_instance = LoadedModule::new("inc", module);
    let interp = Interpreter::new().with_module(mod_instance);
}

#[test]
fn test_create() {
    let input_file = "test_programs/inc.wasm";
    let module = parity_wasm::deserialize_file(input_file).unwrap();
    let mut mod_instance = LoadedModule::new("inc", module);
    mod_instance.validate();
    let interp = Interpreter::new().with_module(mod_instance);
    println!("{:#?}", interp);
    // assert!(false);
}

#[test]
fn test_create_fib() {
    let input_file = "test_programs/fib.wasm";
    let module = parity_wasm::deserialize_file(input_file).unwrap();
    let mut mod_instance = LoadedModule::new("fib", module);
    mod_instance.validate();
    let interp = Interpreter::new().with_module(mod_instance);
    println!("{:#?}", interp);
    // assert!(false);
}

#[test]
fn test_run_fib() {
    let input_file = "test_programs/fib.wasm";
    let module = parity_wasm::deserialize_file(input_file).unwrap();
    let mut mod_instance = LoadedModule::new("fib", module);
    mod_instance.validate();
    let mut interp = Interpreter::new().with_module(mod_instance);

    interp.run(FunctionAddress(1), &vec![Value::I32(30)]);
    assert!(false);
}

/// Helper function to run a small program with some
/// inputs and compare the results to the given desired outputs.
/// Constraints: Makes a single function with a single output,
/// no locals or memories or such...
fn test_stack_program(program: &[Opcode], args: &[Value], desired_output: Option<Value>) {
    let input_types = args.iter().map(|v| v.get_type()).collect::<Vec<_>>();
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
        .main()
            .signature()
                .with_params(input_types)
                .with_return_type(output_type)
                .build()
            .body()
                .with_opcodes(elements::Opcodes::new(init_ops))
                .build()
            .build()
        .build()
        ;
    let mut loaded = LoadedModule::new("test", module);
    loaded.validate();
    let mut interp = Interpreter::new().with_module(loaded);
    let run_result = interp.run(FunctionAddress(0), args);
    assert_eq!(run_result, desired_output)
}

#[test]
fn test_i32_add() {
    test_stack_program(
        &vec![Opcode::I32Add], 
        &vec![Value::I32(1), Value::I32(2)], 
        Some(Value::I32(3)));
}

#[test]
fn test_i64_add() {
    test_stack_program(
        &vec![Opcode::I64Add], 
        &vec![Value::I64(-1), Value::I64(99)], 
        Some(Value::I64(98)));
}

#[test]
fn test_i32_sub() {
    test_stack_program(
        &vec![Opcode::I32Sub], 
        &vec![Value::I32(1), Value::I32(2)], 
        Some(Value::I32(1)));
}

#[test]
fn test_i64_sub() {
    test_stack_program(
        &vec![Opcode::I64Sub], 
        &vec![Value::I64(-1), Value::I64(99)], 
        Some(Value::I64(100)));
}
