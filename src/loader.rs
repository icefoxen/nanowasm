/// Methods for loading a wasm binary module into a more convenient form.

use std::rc::Rc;
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
    pub start: Option<FuncIdx>,
    /// wasm 1.0 defines only a single table.
    /// Even if we import some other table we can only
    /// do it if there's not an existing one, I think.
    pub tables: Option<Table>,
    /// Initializer code for table
    /// `(offset, values)`
    pub table_initializers: Vec<(ConstExpr, Vec<FuncIdx>)>,
    /// wasm 1.0 defines only a single memory.
    pub mem: Option<Memory>,
    /// Initializer code for data segments.
    pub mem_initializers: Vec<(ConstExpr, Vec<u8>)>,
    /// Global values.
    pub globals: Vec<(Global, ConstExpr)>,
    /// Exported values
    pub exported_functions: Vec<Export<FuncIdx>>,
    pub exported_tables: Option<Export<()>>,
    pub exported_memories: Option<Export<()>>,
    pub exported_globals: Vec<Export<GlobalIdx>>,
    /// Imported values
    pub imported_functions: Vec<Import<TypeIdx>>,
    pub imported_tables: Option<Import<elements::TableType>>,
    pub imported_memories: Option<Import<elements::MemoryType>>,
    pub imported_globals: Vec<Import<elements::GlobalType>>,
}

/// A wrapper type that assures at compile-time that
/// a module has been validated.
#[derive(Debug, Clone)]
pub struct ValidatedModule(LoadedModule);

impl ValidatedModule {
    /// Extracts the actual `LoadedModule` from the `ValidatedModule`
    pub(crate) fn into_inner(self) -> LoadedModule {
        self.0
    }

    /// Borrows the actual `LoadedModule` from the `ValidatedModule`
    pub fn borrow_inner(&self) -> &LoadedModule {
        &self.0
    }
}

fn malformed<T>(name: &str, msg: &str) -> Result<T, Error> {
    let e = Error::Malformed {
        module: name.to_owned(),
        message: msg.to_owned(),
    };
    Err(e)
}

impl LoadedModule {
    /// Instantiates and initializes a new module from the `parity_wasm` module type.
    /// This basically goes from a representation very close to the raw webassembly
    /// binary format to a representation more convenient to be loaded into the interpreter's
    /// runtime data.
    ///
    /// Does NOT validate the module or run the start function though!
    pub fn new(name: &str, module: elements::Module) -> Result<Self, Error> {
        if module.version() != 1 {
            return Err(Error::VersionMismatch {
                module: name.to_owned(),
                expected: 1,
                got: module.version(),
            });
        }

        let mut m = Self {
            name: name.to_owned(),
            types: vec![],
            funcs: vec![],
            start: None,

            tables: None,
            table_initializers: vec![],
            mem: None,
            mem_initializers: vec![],
            globals: vec![],

            imported_functions: vec![],
            imported_tables: None,
            imported_memories: None,
            imported_globals: vec![],
            exported_functions: vec![],
            exported_tables: None,
            exported_memories: None,
            exported_globals: vec![],
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
        match (module.code_section(), module.function_section()) {
            (Some(code), Some(functions)) => {
                if code.bodies().len() != functions.entries().len() {
                    return malformed(
                        name,
                        "Number of function bodies != number of function entries",
                    );
                }
                // Evade double-borrow of m here.
                let types = &m.types;
                let converted_funcs = code.bodies()
                    .iter()
                    .zip(functions.entries())
                    .map(|(c, f)| {
                        // Make sure the function signature is a valid type.
                        let type_idx = f.type_ref() as usize;
                        if type_idx >= types.len() {
                            return malformed(
                                name,
                                "Function refers to a type signature that does not exist!",
                            );
                        }
                        Ok(Func {
                            typeidx: TypeIdx(type_idx),
                            locals: types_from_locals(c.locals()),
                            body: FuncBody::Opcodes(c.code().elements().to_owned()),
                        })
                    })
                    .collect::<Result<Vec<Func>, Error>>()?;
                m.funcs.extend(converted_funcs);
            }
            (None, None) => (),
            _ => {
                return malformed(
                    name,
                    "Code section exists but type section does not, or vice versa!",
                );
            }
        }

        // Allocate tables
        if let Some(table) = module.table_section() {
            // currently we can only have one table section with
            // 0 or 1 elements in it, so.
            if table.entries().len() > 1 {
                return malformed(name, "More than one table entry, should never happen!");
            }
            if let Some(table) = table.entries().iter().next() {
                // TODO: As far as I can tell, the memory's minimum size is never used?
                let _min = table.limits().initial();
                let max = table.limits().maximum();

                // TODO: It's apparently valid for a memory to have no max size?
                let mut t = Table::new();
                if let Some(max) = max {
                    t.fill(max);
                }
                m.tables = Some(t);

                if let Some(elements) = module.elements_section() {
                    for segment in elements.entries() {
                        let table_idx = segment.index();
                        if table_idx != 0 {
                            return malformed(
                                name,
                                "Had an Elements segment that referred to table != 0!",
                            );
                        }
                        let offset_code = ConstExpr::try_from(segment.offset().code()).or_else(
                            |_| malformed(name, "Elements section contained invalid const values"),
                        )?;

                        let members: Vec<FuncIdx> = segment
                            .members()
                            .iter()
                            .map(|x| FuncIdx(*x as usize))
                            .collect();
                        m.table_initializers.push((offset_code, members));
                    }
                }
            }
        }

        // Allocate memories
        if let Some(memory) = module.memory_section() {
            // currently we can only have one memory section with
            // 0 or 1 elements in it, so.
            if memory.entries().len() > 1 {
                return malformed(name, "More than one memory entry, should never happen!");
            }
            if let Some(memory) = memory.entries().iter().next() {
                // TODO: As far as I can tell, the memory's minimum size is never used?
                let _min = memory.limits().initial();
                let max = memory.limits().maximum();

                // TODO: It's apparently valid for a memory to have no max size?
                let mut mem = Memory::new(max);

                if let Some(data) = module.data_section() {
                    for segment in data.entries() {
                        let mem_idx = segment.index();
                        if mem_idx != 0 {
                            return malformed(
                                name,
                                "Had a Data segment that referred to memory != 0!",
                            );
                        }
                        let offset_code = ConstExpr::try_from(segment.offset().code()).or_else(
                            |_| malformed(name, "Data section had invalid constant values!"),
                        )?;
                        let members = segment.value().to_owned();
                        m.mem_initializers.push((offset_code, members));
                    }
                }
                m.mem = Some(mem);
            }
        }

        // Load globals
        if let Some(globals) = module.global_section() {
            let global_iter = globals
                .entries()
                .iter()
                .map(|global| {
                    let global_type = global.global_type().content_type();
                    let mutability = global.global_type().is_mutable();
                    let init_code = ConstExpr::try_from(global.init_expr().code()).or_else(|_| {
                        malformed(name, "Globals section had invalid constant values!")
                    })?;
                    let global = Global {
                        variable_type: global_type,
                        mutable: mutability,
                        value: Value::default_from_type(global_type),
                    };
                    Ok((global, init_code))
                })
                .collect::<Result<Vec<(Global, ConstExpr)>, Error>>()?;
            m.globals.extend(global_iter);
        }

        // Load imports
        if let Some(imports) = module.import_section() {
            for entry in imports.entries() {
                let module_name = entry.module().to_owned();
                let field_name = entry.field().to_owned();
                match *entry.external() {
                    elements::External::Function(i) => {
                        m.imported_functions.push(Import {
                            module_name,
                            field_name,
                            value: TypeIdx(i as usize),
                        });
                    }
                    elements::External::Table(i) => {
                        m.imported_tables = Some(Import {
                            module_name,
                            field_name,
                            value: i,
                        });
                    }
                    elements::External::Memory(i) => {
                        m.imported_memories = Some(Import {
                            module_name,
                            field_name,
                            value: i,
                        });
                    }
                    elements::External::Global(i) => {
                        m.imported_globals.push(Import {
                            module_name,
                            field_name,
                            value: i,
                        });
                    }
                }
            }
        }

        // Load exports
        if let Some(exports) = module.export_section() {
            for entry in exports.entries() {
                let name = entry.field().to_owned();
                match *entry.internal() {
                    elements::Internal::Function(i) => m.exported_functions.push(Export {
                        name,
                        value: FuncIdx(i as usize),
                    }),
                    elements::Internal::Table(i) => {
                        m.exported_tables = Some(Export { name, value: () })
                    }
                    elements::Internal::Memory(i) => {
                        m.exported_memories = Some(Export { name, value: () })
                    }
                    elements::Internal::Global(i) => m.exported_globals.push(Export {
                        name,
                        value: GlobalIdx(i as usize),
                    }),
                }
            }
        }

        // Check for start section
        //println!("Module start section: {:?}", module.start_section());
        if let Some(start) = module.start_section() {
            let start = start as usize;
            // Ensure start function is in bounds.
            let max_start_idx = m.funcs.len() + m.imported_functions.len();
            if start >= max_start_idx {
                let message = format!(
                    "unknown function for start: {}, max start index: {}",
                    start, max_start_idx
                );
                let e = Error::Invalid {
                    module: name.to_owned(),
                    reason: message,
                };
                return Err(e);
            }

            // Ensure start function signature is correct.
            // Remember, the indices of imported functions go before
            // the indices of locally defined functions, so we have
            // to figure out whether we're indexing into the functions vec
            // or the imported functions vec.
            // Might want to combine those?
            let startfunc_typeidx = if start < m.imported_functions.len() {
                m.imported_functions[start].value
            } else {
                m.funcs[start - m.imported_functions.len()].typeidx
            };
            let startfunc_type = &m.types[startfunc_typeidx.0];
            let valid_startfunc_type = &FuncType {
                params: vec![],
                return_type: None,
            };
            if startfunc_type != valid_startfunc_type {
                let message = format!(
                    "Invalid start function type: {:?}, should take nothing and return nothing",
                    startfunc_type
                );
                let e = Error::Invalid {
                    module: name.to_owned(),
                    reason: message,
                };
                return Err(e);
            }

            m.start = Some(FuncIdx(start));
        }

        Ok(m)
    }

    /// Adds a host function, plus an export for it.  Not really ideal, but what can one do
    /// when parity-wasm doesn't handle them either?  Hmmm.
    pub fn add_host_func<T>(&mut self, export_name: &str, func: T, params: &FuncType)
    where
        T: Fn(&mut Vec<Value>) + 'static,
    {
        // Add parameters to the type list if necessary
        let type_idx = if let Some(i) = self.types.iter().position(|t| t == params) {
            i
        } else {
            self.types.push(params.clone());
            self.types.len() - 1
        };
        // Create and add function
        let f = Func {
            typeidx: TypeIdx(type_idx),
            locals: vec![],
            body: FuncBody::HostFunction(Rc::new(func)),
        };
        self.funcs.push(f);
        let func_idx = self.funcs.len() - 1;
        // Add export for function.
        self.exported_functions.push(Export {
            name: export_name.to_owned(),
            value: FuncIdx(func_idx),
        })
    }

    /// Validates the module: makes sure types are correct,
    /// all the indices into various parts of the module are valid, etc.
    ///
    /// TODO: Currently does nothing
    pub fn validate(self) -> ValidatedModule {
        ValidatedModule(self)
    }
}
