;; A small fibonacci program for a benchmark.
(module
  (type $in0out0 (func (param) (result)))
  (type $in1out1 (func (param i32) (result i32)))
  (start $start)
  (func $start (type $in0out0)
     i32.const 30
     call $fib
     drop
  )

  ;; A fibonacci function that operates on s32's
  (func $fib (type $in1out1)
     (local $tmp i32)
     get_local 0
     ;; Save our argument into $tmp
     tee_local $tmp

     ;; If our argument < 2...
     i32.const 2
     i32.lt_s
     if $ifcase (result i32)
        ;; return 1
        i32.const 1
     else
        ;; Otherwise, call fib($tmp - 1) * fib($tmp - 2)
        get_local $tmp
        i32.const 1
        i32.sub
        call $fib 

        get_local $tmp
        i32.const 2
        i32.sub
        call $fib 

        i32.mul
     end
  )
)


