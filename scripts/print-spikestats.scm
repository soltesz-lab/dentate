(require-extension typeclass matchable rb-tree srfi-1 spikedata cellconfig)

(define comment-pat (string->irregex "^#.*"))


;(plot:procdebug #t)

(define (print-spike-stats-range celltype min max spike-times)

  (let* ((m (rb-tree-map -))
         (range-data
          (with-instance ((<PersistentMap> m))
                         ((foldi spike-times)
                          (match-lambda* 
                           ((t ns (nspikes lst))
                            (let ((ns1 (filter (lambda (n) (and (<= min n) (<= n max))) ns)))
                              (if (null? ns1) 
                                  (list nspikes lst)
                                  (list (+ nspikes (length ns1))
                                        (cons `(,t . ,ns1) lst))))))
                          `(0 ())
                          )))
         )

    (printf "Cell type ~A:~%" celltype)
    (printf "~A total spikes~%" (car range-data))
    (pp (spike-stats (cadr range-data) nmax tmax))

))


(define (print-spike-stats datadir spike-file)
  (let* (
         (celltypes (read-cell-types datadir))
         (cellranges (read-cell-ranges datadir celltypes)) 
         )

    (match-let (((spike-times nmax tmax) (read-spike-times spike-file)))

        (for-each
         (match-lambda ((celltype min max)
                        (print-spike-stats-range celltype min max spike-times)))

         cellranges)
        
        ))
)




(apply print-spike-stats (command-line-arguments))
