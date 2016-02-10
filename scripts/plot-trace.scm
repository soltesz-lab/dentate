(require-extension typeclass simdata)
(include "lib.scm")

;(plot:procdebug #t)

(define (plot-trace xrange temp-path celltype min max trace-values y i)

  (let* ((m (rb-tree-map -))
         (range-data
          (with-instance ((<PersistentMap> m))
                         ((foldi spike-times)
                          (match-lambda* 
                           ((t ns lst)
                            (let ((ns1 (filter (lambda (n) (and (<= min n) (<= n max))) ns)))
                              (if (null? ns1) lst
                                  (cons `(,t . ,ns1) lst)))))
                          '()
                          )))
         )

    (if (null? range-data) #f

        (begin
          
          (let ((dataport (open-output-file temp-path)))
            (for-each (match-lambda ((t . ns) 
                                     (for-each (lambda (n) (fprintf dataport "~A,~A~%" t n)) ns)))
                      range-data)
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
                       
                       ("xrange"          . ,xrange)
                       ("xaxis.stubs"     . "inc 100")
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
    ))


(define (raster-plot datadir spike-file plot-label xrange)
  (match-let 

   (
    (celltypes (read-cell-types datadir))

    ((data tmax nmax)
     (fold
      (lambda (spike-file ax)
        (match-let (((data tmax nmax)  ax))
          (let ((data1 (map (lambda (line) (map string->number (string-split  line " ")))
                          (filter (lambda (line) (not (irregex-match comment-pat line)))
                                  (read-lines spike-file)))))
            (let ((t1 (fold (lambda (row ax) (max (car row) ax)) tmax data1))
                  (nmax1 (fold (lambda (row ax) (fold max ax (cdr row))) nmax data1)))
              (list (append data1 data) (max tmax t1) nmax1)
              ))
          ))
        '(() 0.0 0)
        (list spike-file)))
    )

   (print "tmax = " tmax)
   (print "nmax = " nmax)

   (let* (
          
          (cellranges (read-cell-ranges datadir celltypes)) 

          (spike-times (read-spike-times data))

          )

  (let-values (
               ((fd1 temp-path1) (file-mkstemp "/tmp/raster-plot.s1.XXXXXX"))
	       )
	 (file-close fd1)
	 

	 (plot:init 'png (make-pathname
                          "." 
                          (sprintf "~A_raster.png" 
                                   (pathname-strip-directory
                                    (pathname-strip-extension spike-file )))))
	 (plot:arg "-cm" )
	 (plot:arg "-textsize"   "12")
	 (plot:arg "-pagesize"   "35,20");;PAPER
	 (plot:arg "-cpulimit"   "60")
	 (plot:arg "-maxrows"    "700000")
	 (plot:arg "-maxfields"  "1400000")
	 (plot:arg "-maxvector"  "700000")

         (plot:proc "page"
                    `(
                      ("title" . ,plot-label)
                      ))

         (fold
          (match-lambda* (((celltype min max) (i y)) 
                          (if (plot-range xrange temp-path1 
                                          celltype min max spike-times y i)
                              (list (+ i 1) (+ y 4))
                              (list (+ i 1) y))))
          (list 0 1) (reverse (cons (list 'stim nmax +inf.0) cellranges)))


         (plot:end)

       ))
))

(apply trace-plot (command-line-arguments))
