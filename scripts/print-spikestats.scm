(require-extension typeclass matchable rb-tree srfi-1 spikedata cellconfig getopt-long)

(define comment-pat (string->irregex "^#.*"))

(define list-fold fold)


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



(define opt-grammar
  `(
    (x-range
     "X range to process"
     (value       
      (required "X-MIN:X-MAX")
      (transformer ,(lambda (x) 
                      (let ((kv (string-split x ":")))
                        (cons (string->number (car kv))
                              (string->number (cadr kv))))))
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
(define (print-spikestats:usage)
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
    (print-spikestats:usage)
    (print-spike-stats (opt 'data-dir) (opt 'spike-file))
    )


