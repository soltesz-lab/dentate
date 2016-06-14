(require-extension typeclass matchable rb-tree srfi-1 spikedata cellconfig)

(define comment-pat (string->irregex "^#.*"))

(define list-fold fold)

;(plot:procdebug #t)

(define (merge-spikes x lst) (merge x lst <))

(define (print-spike-stats-range celltype min max tmax spike-times)

  (let* ((m (rb-tree-map -))
         (range-data
          (with-instance ((<PersistentMap> m))
                         ((foldi spike-times)
                          (match-lambda* 
                           ((t ns (nspikes msp))
                            (let ((ns1 (filter (lambda (n) (and (<= min n) (<= n max))) ns)))
                              (if (null? ns1) 
                                  (list nspikes msp)
                                  (list (+ nspikes (length ns1))
					(list-fold (lambda (n msp) 
						     (update msp n (list t) merge-spikes)) msp ns1))
			      ))
			   ))
                          `(0 ,(empty))
                          )))
         )

    (printf "Cell type ~A:~%" celltype)
    (printf "~A total spikes~%" (car range-data))
    (pp (spike-stats (cadr range-data) (car range-data) tmax))

))


(define (print-spike-stats datadir spike-file)
  (let* (
         (celltypes (read-cell-types datadir))
         (cellranges (read-cell-ranges datadir celltypes)) 
         )

    (match-let (((spike-times nmax tmax) (read-spike-times spike-file)))

        (for-each
         (match-lambda ((celltype min max)
                        (print-spike-stats-range celltype min max tmax spike-times)))

         cellranges)
        
        ))
)




(apply print-spike-stats (command-line-arguments))
