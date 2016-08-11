(require-extension typeclass matchable rb-tree srfi-1 spikedata cellconfig getopt-long)

(require-library ploticus)
(import
 (prefix ploticus plot:)
 )

(define comment-pat (string->irregex "^#.*"))


;(plot:procdebug #t)

(define (plot-range xrange xinc temp-path celltype min max spike-times y i)
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
    (printf "   ~A total spikes~%" (car range-data))

    (if (null? (cadr range-data)) #f

        (begin
          

          (let ((dataport (open-output-file temp-path)))
            (for-each (match-lambda ((t . ns) 
                                     (for-each (lambda (n) (fprintf dataport "~A,~A~%" t n)) ns)))
                      (cadr range-data))
            (close-output-port dataport))
          
          (plot:proc "getdata"
                     `(
                       ;("showdata"   . "yes")
                       ("delim"      . "comma")
                       ("fieldnames" . "t n")
                       ("pathname"   . ,temp-path)
                       ))
          
          (plot:proc "areadef"
                     `(
                       ("rectangle" . ,(sprintf "2 ~A 30 ~A" y (+ y 3)))
                       ("areacolor" . "white")
                       
                       ("xrange"          . ,(sprintf "~A ~A" (car xrange) (cdr xrange)))
                       ("xaxis.stubs"     . ,(sprintf "inc ~A" xinc))
                       ("xaxis.stubdetails" . "adjust=0.1,0.2")
                       ("xaxis.label"     . ,(if (= i 0) "Time [ms]" ""))
                       
                       ("yautorange"     , "datafield=n")
                       ("yaxis.label"     . ,(->string celltype))
                       ("yaxis.axisline"  . "no")
                       ("yaxis.tics"      . "no")
                       ;("yaxis.stubs"     . "inc 100000")
                       )
                     )

          (plot:proc "legendentry"
                     `(("sampletype" .  "color")
                       ("details"    .  "black") 
                       ("tag"        .  "0")
                       ))
          
          (plot:proc "scatterplot"
                     `(("xfield" .  "t")
                       ("yfield" .  "n")
                       ("symbol" . "style=fill shape=circle fillcolor=blue radius=0.05")
                       ("cluster"   . "no")
                       ))

          #t
          ))
    )
)


(define (raster-plot datadir spike-file plot-label xrange #!key (x-inc 10))
  (let* (
         (celltypes (read-cell-types datadir))
         (cellranges (read-cell-ranges datadir celltypes)) 
         )

    (match-let (((spike-times nmax tmax) (read-spike-times spike-file)))

     (let-values (
                  ((fd1 temp-path1) (file-mkstemp "/tmp/raster-plot.s1.XXXXXX"))
                  )
       (file-close fd1)
       
       
       (plot:init 'eps (make-pathname
                        "." 
                        (sprintf "~A_raster.eps" 
                                 (pathname-strip-directory
                                  (pathname-strip-extension spike-file )))))
       (plot:arg "-cm" )
       (plot:arg "-textsize"   "12")
       (plot:arg "-pagesize"   "12,23")
       (plot:arg "-cpulimit"   "1200")
       (plot:arg "-maxrows"    "3000000")
       (plot:arg "-maxfields"  "5000000")
       (plot:arg "-maxvector"  "7000000")
       
       (plot:proc "page"
                  `(
                    ("title" . ,plot-label)
                    ))
       
       (fold
        (match-lambda* (((celltype min max) (i y)) 
                        (if (plot-range xrange x-inc temp-path1 
                                        celltype min max spike-times y i)
                            (list (+ i 1) (+ y 4))
                            (list (+ i 1) y))))
        (list 0 1) (reverse (cons (list 'stim (+ nmax 1) +inf.0) cellranges)))
       
       
       (plot:end)
       
       ))
    )
)


(define opt-grammar
  `(
    (x-range
     "X range to plot"
     (value       
      (required "X-MIN:X-MAX")
      (transformer ,(lambda (x) 
                      (let ((kv (string-split x ":")))
                        (cons (string->number (car kv))
                              (string->number (cadr kv))))))
      ))

    (x-inc
     "X tic increment"
     (value       
      (required "NUMBER")
      (transformer ,string->number)))

    (spike-file
     "path to spike file"
     (single-char #\s)
     (value (required DATA-FILE)))

    (data-dir
     "model dataset directory"
     (single-char #\d)
     (value (required DIR)))

    (plot-label
     "plot label"
     (single-char #\l)
     (value (required LABEL)))

    (help  "Print help"
	    (single-char #\h))
  
  ))

;; Use args:usage to generate a formatted list of options (from OPTS),
;; suitable for embedding into help text.
(define (plot-raster:usage)
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
    (plot-raster:usage)
    (raster-plot (opt 'data-dir) (opt 'spike-file)
                 (opt 'plot-label) (opt 'x-range)
                 x-inc: (or (opt' x-inc) 10))
    )

