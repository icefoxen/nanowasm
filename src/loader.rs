use parity_wasm::elements;

use types::*;

/// Takes a slice of `Local`'s (local variable *specifications*),
/// and creates a vec of their types.
///
/// Slightly trickier than just a map+collect.
fn types_from_locals(locals: &[elements::Local]) -> Vec<elements::ValueType> {
    // This looks like it should just be a map and collect but actually isn't,
    // 'cause we need to iterate the inner loop.  We could make it a map but it's
    // trickier and not worth the bother.
    let num_local_slots = locals.iter().map(|x| x.count() as usize).sum();
    let mut v = Vec::with_capacity(num_local_slots);
    for local in locals {
        for _i in 0..local.count() {
            let t = local.value_type();
            v.push(t);
        }
    }
    v
}

/// A loaded wasm module
#[derive(Debug, Clone)]
pub struct LoadedModule {
    /// Module name.  Not technically necessary, but handy.
    pub name: String,
    /// Function type vector
    pub types: Vec<FuncType>,
    /// Function value vector
    pub funcs: Vec<Func>,
    /// Index of start function, if any.
    pub start: Option<usize>,
    /// wasm 1.0 defines only a single table,
    /// but we can import multiple of them?
    pub tables: Table,
    /// wasm 1.0 defines only a single memory.
    pub mem: Memory,
    pub globals: Vec<Global>,

    pub validated: bool,
}

impl LoadedModule {
    /// Instantiates and initializes a new module from the `parity_wasm` module type.
    /// This basically goes from a representation very close to the raw webassembly
    /// binary format to a representation more convenient to be loaded.
    ///
    /// Does NOT validate the module or run the start function though!
    pub fn new(name: &str, module: elements::Module) -> Self {
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

            if let Some(_elements) = module.elements_section() {
                // TODO
                println!("Elements");
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

            if let Some(_data) = module.data_section() {
                // TODO
                println!("data");
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
    ///
    /// Currently unimplemented.
    pub fn validate(&mut self) {
        // TODO: Do this!  Unless we can use the parity_wasm validator...
        self.validated = true;
    }
}