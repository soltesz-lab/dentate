(require-extension typeclass)
(include "lib.scm")



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
               ((fd2 temp-path2) (file-mkstemp "/tmp/raster-plot.s2.XXXXXX"))
	       )
	 (file-close fd1)
	 (file-close fd2)

         (let ((m (rb-tree-map -))
               (dataport (open-output-file temp-path1)))
           (with-instance ((<PersistentMap> m))
                          ((for-each-ascending spike-times)
                           (match-lambda 
                            ((t . ns)
                             (for-each (lambda (n) (fprintf dataport "~A,~A~%" t n)) ns))
                            )))
           (close-output-port dataport))
	 
	 (plot:init 'png (make-pathname
                          "." 
                          (sprintf "~A_raster.png" 
                                   (pathname-strip-directory
                                    (pathname-strip-extension spike-file )))))
	 
	 (plot:arg "-cm" )
	 (plot:arg "-pagesize"   "35,20");;PAPER
	 (plot:arg "-textsize"   "12")
	 (plot:arg "-cpulimit"   "60")
	 (plot:arg "-maxrows"    "700000")
	 (plot:arg "-maxfields"  "1400000")
	 (plot:arg "-maxvector"  "700000")
	 
	 (plot:proc "getdata"
		  `(
;		    ("showdata"   . "yes")
		    ("delim"      . "comma")
		    ("fieldnames" . "xcoord ycoord")
		    ("pathname"   . ,temp-path1)
		    ))
       
	 (plot:proc "areadef"
		  `(("title"     . ,(sprintf "~A" plot-label))
                    ("titledetails" . "adjust=0,0.2")
		    ("rectangle" . "1 1 30 18")
		    ("areacolor" . "white")

		    ("xrange"          . ,xrange)
		    ("xaxis.axisline"  . "no")
		    ("xaxis.tics"      . "no")
;;		    ("xaxis.stubs"     . "inc 50")
;;		    ("xaxis.stubrange" . "0")
;;		    ("xaxis.stubdetails" . "adjust=0,1")

		    ("yautorange"      . "datafield=2")
;;		    ("yaxis.label"     . "Cell #")
		    ("yaxis.axisline"  . "no")
		    ("yaxis.tics"      . "no")
;;		    ("yaxis.stubs"     . "inc 10")
;;		    ("yaxis.stubrange" . "0")
		    )
		  )

       (plot:proc "legendentry"
		  `(("sampletype" .  "color")
		    ("details"    .  "black") 
		    ("tag"        .  "0")
		    ))

       (plot:proc "scatterplot"
		  `(("xfield"    .  "xcoord")
		    ("yfield"    .  "ycoord")
                    ;("rectangle" . "5 5")
                    ("linelen"   . "0.01")
		    ("linedetails"   . "width=1.0")
		    ("linedir"   . "v")
		    ))
		    
       ;(plot:proc "bars"
	;	  `(("locfield"    .  "t")
	;	    ("lenfield"    .  "count")
	;	    ("thinbarline"    .  "color=gray(0.5)")
         ;           ))
       
       (plot:end)

       ))
))

(apply raster-plot (command-line-arguments))
