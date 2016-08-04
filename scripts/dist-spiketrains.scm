(require-extension typeclass matchable rb-tree srfi-1 spikedata cellconfig getopt-long)

(define comment-pat (string->irregex "^#.*"))

(define list-fold fold)

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


(define (dist-spike-trains-range celltype min max tmax spike-times x-ranges)

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

    (let* ((n-ranges
            (let recur ((i 0) (ranges x-ranges) (ids '()))
              (if (null? ranges) 
                  (reverse ids)
                  (let ((range-ids
                         (list->u32vector
                          (with-instance ((<PersistentMap> m)) 
                                         ((foldi-right (cadr range-data))
                                          (match-lambda* 
                                           ((n ts lst) (if (assoc i ts) (cons n lst) lst)))
                                          '()
                                          )
                                         ))
                         ))
                    (recur (+ i 1) (cdr ranges) (cons range-ids ids))))
              ))
           (n-dists 
            (map (lambda (x y) (levenshtein-distance (u32vector-length x) x (u32vector-length y) y))
                 n-ranges (cdr n-ranges)))
           )

   (call-with-output-file (sprintf "~A_spikevectors.dat" celltype)
     (lambda (out)
       (for-each (lambda (range)
                   (for-each (lambda (t) (fprintf out " ~A" t)) (u32vector->list range))
                   (fprintf out "~%"))
                 n-ranges)
       ))

   (call-with-output-file (sprintf "~A_spikedist.dat" celltype)
     (lambda (out)
       (for-each (lambda (t) (fprintf out " ~A" t)) n-dists)
       (fprintf out "~%")
       ))
   ))
  )

(define (dist-spike-trains datadir spike-file x-ranges)
  (let* (
         (celltypes (read-cell-types datadir))
         (cellranges (read-cell-ranges datadir celltypes)) 
         )

    (match-let (((spike-times nmax tmax) (read-spike-times spike-file)))

        (for-each
         (match-lambda ((celltype min max)
                        (dist-spike-trains-range celltype min max tmax spike-times x-ranges)))

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

    (help  "Print help"
	    (single-char #\h))
  
  ))

;; Use args:usage to generate a formatted list of options (from OPTS),
;; suitable for embedding into help text.
(define (dist-spiketrains:usage)
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
    (dist-spiketrains:usage)
    (dist-spike-trains (opt 'data-dir) (opt 'spike-file) (opt 'x-ranges))
    )


