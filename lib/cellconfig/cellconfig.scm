
(module cellconfig

        (read-cell-types read-cell-ranges)

        (import scheme chicken)

(require-extension matchable)
(require-extension srfi-1 srfi-13 irregex data-structures files posix extras)
#;(import
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
                  (match-let (((_ cell-number type-name prototype template) celltype))
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
                  (error 'read-cell-types "unknown index type" celltype))
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
                                                      (let ((x (string-split (string-trim-both line) " ")))
                                                        (match-let (((min-index max-index) ax))
                                                                   (let ((n (string->number (car x))))
                                                                     (list (min n min-index) 
                                                                           (max n max-index))
                                                                     ))
                                                        ))
                                                    (list +inf.0 -inf.0)
                                                    (cdr (read-lines (make-pathname datadir index-file))))))
                                            (list (cons (list type-name 
                                                              (inexact->exact min-index)
                                                              (inexact->exact max-index)) lst)
                                                  max-index))))
                              (else (error 'read-cell-ranges "unknown cell type" celltype)))
                        ))
           )
         (list '() 0) celltypes))
  )

)
