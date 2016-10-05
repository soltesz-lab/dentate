(require-extension typeclass matchable rb-tree srfi-1 spikedata cellconfig getopt-long)

(require-library ploticus)
(import
 (prefix ploticus plot:)
 )

(define comment-pat (string->irregex "^#.*"))

(define (choose lst) (list-ref lst (random (- (length lst) 1))))

(define colors '("x000000" "x00FF00" "x0000FF" "xFF0000" "x01FFFE" "xFFA6FE"
                 "xFFDB66" "x006401" "x010067" "x95003A" "x007DB5" "xFF00F6" "xFFEEE8" "x774D00"
                 "x90FB92" "x0076FF" "xD5FF00" "xFF937E" "x6A826C" "xFF029D" "xFE8900" "x7A4782"
                 "x7E2DD2" "x85A900" "xFF0056" "xA42400" "x00AE7E" "x683D3B" "xBDC6FF" "x263400"
                 "xBDD393" "x00B917" "x9E008E" "x001544" "xC28C9F" "xFF74A3" "x01D0FF" "x004754"
                 "xE56FFE" "x788231" "x0E4CA1" "x91D0CB" "xBE9970" "x968AE8" "xBB8800" "x43002C"
                 "xDEFF74" "x00FFC6" "xFFE502" "x620E00" "x008F9C" "x98FF52" "x7544B1" "xB500FF"
                 "x00FF78" "xFF6E41" "x005F39" "x6B6882" "x5FAD4E" "xA75740" "xA5FFD2" "xFFB167" 
                 "x009BFF" "xE85EBE"))


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
                       ("symbol" . ,(sprintf "style=fill shape=circle fillcolor=~A radius=0.05"
                                             (choose colors)))
                       ("cluster"   . "no")
                       ))

          #t
          ))
    )
)


(define (raster-plot datadir spike-file plot-label xrange
                     #!key  (x-inc 10) (cpu-limit 1200)
                     (max-rows 5000000) (max-fields 5000000)
                     (max-vector 5000000))
  (let* (
         (celltypes (and datadir (read-cell-types datadir)))
         (cellranges (or (and datadir (read-cell-ranges datadir celltypes))
                         (let ((lines (read-lines "celltypes.dat")))
                           (fold (lambda (line lst)
                                   (let ((x (string-split (string-trim-both line) " ")))
                                     (let ((type-name (car x))
                                           (min-index (string->number (cadr x)))
                                           (max-index (string->number (caddr x))))
                                       (cons (list type-name 
                                                   (inexact->exact min-index)
                                                   (inexact->exact max-index)) lst)
                                       )))
                                 '() lines))))
         )

    (match-let (((spike-times nmax tmax) (read-spike-times xrange spike-file)))

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
       (plot:arg "-cpulimit"   (number->string cpu-limit))
       (plot:arg "-maxrows"    (number->string max-rows))
       (plot:arg "-maxfields"  (number->string max-fields))
       (plot:arg "-maxvector"  (number->string max-vector))
       
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

    (cpu-limit
     "time limit [s]"
     (single-char #\t)
     (value (required TIME)
            (transformer ,string->number)))

    (max-rows
     "maximum rows"
     (value (required NUMBER)
            (transformer ,string->number)))
    (max-fields
     "maximum fields"
     (value (required NUMBER)
            (transformer ,string->number)))
    (max-vector
     "maximum vector size"
     (value (required NUMBER)
            (transformer ,string->number)))
    

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
                 x-inc: (or (opt' x-inc) 10)
                 cpu-limit: (or (opt 'cpu-limit) 1200)
                 max-rows: (or (opt 'max-rows)  5000000)
                 max-fields: (or (opt 'max-fields) 5000000)
                 max-vector: (or (opt 'max-vector) 5000000)
                 )
    )

