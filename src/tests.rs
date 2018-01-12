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
            
        // interp.run_module_function("fib", FuncIdx(1), &vec![Value::I32(30)]);
        interp.run_function(FunctionAddress(1), &vec![Value::I32(30)]);
        assert!(false);
    }

