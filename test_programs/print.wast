(module
  (type (;0;) (func (param) (result)))
  (type (;1;) (func (param i32) (result)))
  (import "foo_module" "foo" (func (;1;) (type 0)))
  (import "foo_module" "printi" (func (;2;) (type 1)))
  (start 0)
  (func (;0;) (type 0) (param) (result)
     i32.const 5
     i32.const 5
     i32.const 5
     call 2
     drop
     drop
     drop
  )
)


