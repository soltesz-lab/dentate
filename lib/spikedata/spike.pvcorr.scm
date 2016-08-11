(require-extension typeclass matchable rb-tree srfi-1 fmt spikedata cellconfig getopt-long statistics)

(define comment-pat (string->irregex "^#.*"))

(define list-fold fold)
(define list-map map)

(define (merge-real xs lst) (merge xs lst <))

(define (alist-update k v merge alst)
  (let recur ((alst alst) (hd '()))
    (if (null? alst)
        (cons (cons k v) hd)
        (if (= k (caar alst))
            (append (cons (cons (caar alst) (merge v (cdar alst))) hd) (cdr alst))
            (recur (cdr alst) (cons (car alst) hd))
            ))
    ))

(define (find-range ranges)
  (lambda (x) 
    (let recur ((ranges ranges) (i 0))
      (if (null? ranges)
          (list i x)
          (if (< x (car ranges))
              (list i x)
              (recur (cdr ranges) (+ 1 i))
              ))
      ))
  )

(define (merge-ranges lst1 lst2) 
  (fold (lambda (r lst) 
          (let ((i (car r)) (xs (cdr r)))
            (alist-update i xs merge-real lst)))
        lst2 lst1))


(define (pvcorr-spike-trains-range celltype min max tmax spike-times x-ranges rank write-pvs-loc)

  (let* ((m (rb-tree-map -))
         (n-range (length x-ranges))
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
						     (update msp n
                                                             (list ((find-range x-ranges) t))
                                                             merge-ranges)) msp ns1))
			      ))
			   ))
                          `(0 ,(empty))
                          )))
         )

    (printf "Cell type ~A (gid ~A to ~A):~%" celltype min max)
    (printf "~A total spikes~%" (car range-data))

    (let* ((n-counts
            (with-instance ((<PersistentMap> m)) 
                           ((foldi-right (cadr range-data))
                            (match-lambda* 
                             ((n ts lst) 
                              (cons (cons n (sort (list-map (lambda (t) (cons (car t) (length (cdr t)))) ts)
                                                  (lambda (x y) (< (car x) (car y))))) 
                                    lst)))
                            '()
                            )
                           ))

           (n-pvpairs 
            (let recur ((i 0) (ax '()))
              (if (< i (- n-range 1))
                  (recur (+ i 1)
                         (cons (map
                                (lambda (en)
                                  (list
                                   (if (assoc i (cdr en))
                                       (alist-ref i (cdr en)) 0)
                                   (if (assoc (+ i 1) (cdr en))
                                       (alist-ref (+ i 1) (cdr en)) 0)))
                                n-counts) ax))
                  (reverse ax))))

           )


      (if write-pvs-loc
          (call-with-output-file 
              (make-pathname (or (and (string? write-pvs-loc) write-pvs-loc) ".")
                             (sprintf "~A_spikevectors.dat" celltype))
            (lambda (out)
              (for-each (lambda (en)
                          (fprintf out " ~A" (car en))
                          (let recur ((i 0))
                            (if (< i n-range)
                                (begin
                                  (if (assoc i (cdr en))
                                      (fprintf out " ~A" (alist-ref i (cdr en)))
                                      (fprintf out " 0"))
                                  (recur (+ i 1)))))
                          (fprintf out "~%"))
                        n-counts)
              ))
          )

   (if rank
       (let ((n-corrs
              (map (lambda (pvp) 
                     (let-values (((rs p) (spearman-rank-correlation pvp)))
                       (list rs p)))
                   n-pvpairs)))
         (let ((out (current-output-port)))
           (fmt out (fmt-join (lambda (ts) (fmt-join (lambda (t) (num t 10 4)) ts " ")) n-corrs "\n"))
           (fprintf out "~%")
           ))
       (let ((n-corrs
              (map correlation-coefficient
                   n-pvpairs)))
         (let ((out (current-output-port)))
           (fmt out (fmt-join (lambda (t) (num t 10 4)) n-corrs " "))
           (fprintf out "~%")
           ))
       ))
   ))
  

(define (pvcorr-spike-trains datadir spike-file x-ranges rank write-pvs-loc)
  (let* (
         (celltypes (read-cell-types datadir))
         (cellranges (read-cell-ranges datadir celltypes)) 
         )

    (match-let (((spike-times nmax tmax) (read-spike-times spike-file)))

        (for-each
         (match-lambda ((celltype min max)
                        (pvcorr-spike-trains-range celltype min max tmax spike-times x-ranges rank write-pvs-loc)))

         cellranges)
        
        ))
)



(define opt-grammar
  `(
    (x-ranges
     "X ranges to process"
     (value       
      (required "X1:X2:...:XN")
      (transformer ,(lambda (x) 
                      (let ((kv (string-split x ":")))
                        (map string->number kv))))
      ))

    (spike-file
     "path to spike file"
     (single-char #\s)
     (value (required DATA-FILE)))

    (data-dir
     "model dataset directory"
     (single-char #\d)
     (value (required DIR)))

    (rank
     "compute rank correlation and its significance")

    (write-vectors
     "write population spike vectors to files in given DIR"
     (single-char #\w)
     (value (optional DIR)))

    (help  "Print help"
	    (single-char #\h))
  
  ))

;; Use args:usage to generate a formatted list of options (from OPTS),
;; suitable for embedding into help text.
(define (pvcorr-spiketrains:usage)
  (print "Usage: " (car (argv)) " [options...] operands ")
  (newline)
  (print "Where operands are spike raster files")
  (newline)
  (print "The following options are recognized: ")
  (newline)
  (width 35)
  (print (parameterize ((indent 5)) (usage opt-grammar)))
  (exit 1))


;; Process arguments and collate options and arguments into OPTIONS
;; alist, and operands (filenames) into OPERANDS.  You can handle
;; options as they are processed, or afterwards.

(define opts    (getopt-long (command-line-arguments) opt-grammar))
(define opt     (make-option-dispatch opts opt-grammar))

(if (opt 'help) 
    (pvcorr-spiketrains:usage)
    (pvcorr-spike-trains (opt 'data-dir) (opt 'spike-file) (opt 'x-ranges) (opt 'rank) (opt 'write-vectors))
    )


