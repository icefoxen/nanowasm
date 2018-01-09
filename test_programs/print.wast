(module
  (type (;0;) (func (param) (result)))
  (import "foo_module" "foo" (func (;1;) (type 0)))
  (start 0)
  (func (;0;) (type 0) (param) (result)
     call 1
     i32.const 1
     i32.const 2
     i32.add
     drop
  )
)


