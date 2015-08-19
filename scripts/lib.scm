
(require-extension matchable typeclass rb-tree)
(require-library srfi-1 irregex data-structures files posix extras ploticus)
(import
 (only srfi-1 filter list-tabulate)
 (only files make-pathname)
 (only posix glob)
 (only data-structures ->string alist-ref compose)
 (only extras fprintf random)
 (only mathh cosh tanh log10)
 (prefix ploticus plot:)
 )

(define comment-pat (string->irregex "^#.*"))

(define (sample n v)
  (let ((ub (vector-length v)))
    (let ((idxs (list-tabulate n (lambda (i) (random ub)))))
      (map (lambda (i) (vector-ref v i)) idxs)
      ))
    )


(define (read-cell-types datadir)
  (define (string->prototype s)
    (cond ((string=? s "forest:") 'forest)
          ((string=? s "single:") 'single)
          (else (error 'read-cell-types "unknown prototype string" s))))
  (let* (
         (celltypes-path (make-pathname datadir "celltypes.dat"))
         (lines (filter 
                 (lambda (line) (not (or (irregex-match comment-pat line) (string-null? line))))
                 (read-lines celltypes-path)))
         (n     (string->number (car lines)))
         (rest  (map (lambda (line) (string-split  line " ")) (cdr lines)))
         )
    (if (not (= n (length rest)))
        (error 'read-cell-types "number of entries does not match first line in file" celltypes-path))
    (map (lambda (celltype) 
           (cond ((string=? (car celltype) "cardinality:")
                  (match-let (((_ cell-number type-name prototype template) line))
                             `(,(string->symbol type-name)
                               (cardinality . ,(string->number cell-number))
                               (,(string->prototype prototype) . ,template))
                             ))
                 ((string=? (car celltype) "indexfile:")
                  (match-let (((_ index-file type-name prototype template) celltype))
                             `(,(string->symbol type-name) 
                               (indexfile . ,index-file)
                               (,(string->prototype prototype) . ,template))))
                 (else
                  (error *read-cell-types "unknown index type" line))
                 ))

         rest
         ))
  )
                  

(define (read-cell-ranges datadir celltypes)
  (car
   (fold (lambda (celltype ax)
           (let ((type-name (car celltype)))
             (match-let (((lst offset) ax))
                        (cond ((alist-ref 'cardinality (cdr celltype)) =>
                               (lambda (cell-number)
                                 (list (cons (list type-name offset (+ offset cell-number)) lst)
                                       (+ offset cell-number))))
                              ((alist-ref 'indexfile (cdr celltype)) =>
                               (lambda (index-file)
                                 (match-let (((min-index max-index)
                                              (fold (lambda (line ax)
                                                      (let ((x (string-trim-both line)))
                                                        (match-let (((min-index max-index) ax))
                                                                   (let ((n (string->number x)))
                                                                     (list (min n min-index) 
                                                                           (max n max-index))
                                                                     ))
                                                        ))
                                                    (list +inf.0 -inf.0)
                                                    (cdr (read-lines (make-pathname datadir index-file))))))
                                            (list (cons (list type-name min-index max-index) lst)
                                                  max-index))))
                              (else (error 'read-cell-ranges "unknown cell type" celltype)))
                        ))
           )
         (list '() 0) celltypes))
  )

(define list-fold fold)

(define (read-spike-times data)
  (let ((m (rb-tree-map -)))
    (with-instance ((<PersistentMap> m))
                   (list-fold
                    (lambda (t.n msp) 
                      (update msp (car t.n) (cdr t.n) append))
                    (empty) data))))

