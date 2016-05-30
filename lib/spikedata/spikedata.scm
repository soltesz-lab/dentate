
(module spikedata

        (read-spike-times)

        (import scheme chicken)

(require-extension matchable typeclass rb-tree)
(require-library srfi-1 srfi-13 irregex data-structures files posix extras ploticus)
(import
 (only srfi-1 filter list-tabulate fold)
 (only srfi-13 string-trim-both string-null?)
 (only files make-pathname)
 (only posix glob)
 (only data-structures ->string alist-ref compose string-split)
 (only extras fprintf random read-lines)
 (only mathh cosh tanh log10)
 (only irregex irregex-match string->irregex)
 )

(define comment-pat (string->irregex "^#.*"))

(define (sample n v)
  (let ((ub (vector-length v)))
    (let ((idxs (list-tabulate n (lambda (i) (random ub)))))
      (map (lambda (i) (vector-ref v i)) idxs)
      ))
    )


(define list-fold fold)

(define (read-spike-times data)
  (let ((m (rb-tree-map -)))
    (with-instance ((<PersistentMap> m))
                   (list-fold
                    (lambda (t.n msp) 
                      (update msp (car t.n) (cdr t.n) append))
                    (empty) data))))

)
