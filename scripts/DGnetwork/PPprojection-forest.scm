(use srfi-1 srfi-13 mathh matchable kd-tree mpi getopt-long fmt npcloud)


(define local-config (make-parameter '()))
(define trees-directory (make-parameter '()))


(MPI:init)

(define opt-grammar
  `(
    
    (pp-cell-prefix "Specify the prefix for PP cell file names"
		    (value (required PREFIX)))

    (pp-cells "Specify number of PP cell modules and PP cells per module, separated by colon"
                (value (required "N-MOD:N-PP-CELL")
                       (transformer ,(lambda (x) (map string->number (string-split x ":"))))))
    
    (output-dir "Write results in the given directory"
               (single-char #\o)
               (value (required PATH)))

    (trees-dir "Load post-synaptic trees from the given directory"
               (single-char #\t)
               (value (required PATH)))

    (forest "Load the given forest of post-synaptic cells"
	     (single-char #\f)
	     (value (required INDEX)
		    (transformer ,string->number)))
    
    (presyn-dir "Load pre-synaptic location coordinates from the given directory"
                   (single-char #\p)
                   (value (required PATH)))
    
    (radius "Connection radius"
            (single-char #\r)
            (value (required RADIUS)
                   (transformer ,(lambda (x) (string->number x))))
            )

    (layers "Comma-separated list of connection layers of postsynaptic cells"
	     (single-char #\l)
	     (value (required LAYERS)
		    (transformer ,(lambda (x) (map string->number (string-split x ","))))))
    
    (label "Use the given label for outpput projection files"
	   (value (required LABEL)))

    (weights "Load weights from specified file"
               (single-char #\w)
               (value (required PATH)))

    (verbose "print additional debugging information" 
             (single-char #\v))
    
    (help         (single-char #\h))
    ))

;; Process arguments and collate options and arguments into OPTIONS
;; alist, and operands (filenames) into OPERANDS.  You can handle
;; options as they are processed, or afterwards.

(define opts    (getopt-long (command-line-arguments) opt-grammar))
(define opt     (make-option-dispatch opts opt-grammar))

;; Use usage to generate a formatted list of options (from OPTS),
;; suitable for embedding into help text.
(define (my-usage)
  (print "Usage: " (car (argv)) " [options...] ")
  (newline)
  (print "The following options are recognized: ")
  (newline)
  (print (parameterize ((indent 5)) (usage opt-grammar)))
  (exit 1))

(if (opt 'help)
    (my-usage))

(if (opt 'verbose)
    (npcloud-verbose 1))

(if (npcloud-verbose)
    (pp (local-config) (current-error-port)))

(define my-comm (MPI:get-comm-world))
(define myrank  (MPI:comm-rank my-comm))
(define mysize  (MPI:comm-size my-comm))


(define-syntax
  SetExpr
  (syntax-rules
      (population section genpoint-section union)
    ((SetExpr (population p))
     (lambda (repr) 
       (case repr 
             ((list) (map (lambda (cell)
                            (list (cell-index cell)
                                  (cell-origin cell)))
                          p))
             ((tree) (let ((pts (map (lambda (cell)
                                       (list (cell-index cell)
                                             (cell-origin cell)))
                                     p)))
                       (list->kd-tree pts
                                      make-point: (lambda (v) (second v))
                                      make-value: (lambda (i v) (list (first v) 0.0)))

                       ))
             )))
    ((SetExpr (genpoint-section p t))
     (lambda (repr)
       (case repr
         ((list)
          (map (lambda (cell) 
                 (list (cell-index cell) 
                       (list (cell-section-ref (quote t) cell))))
               p))
         ((tree)
          (genpoint-cells-sections->kd-tree p (quote t)))
         )))
    ((SetExpr (section p t))
     (lambda (repr)
       (case repr
         ((list)
          (map (lambda (cell) 
                 (list (cell-index cell) 
                       (list (cell-section-ref (quote t) cell))))
               p))
         ((tree)
          (cells-sections->kd-tree p (quote t)))
         )))
    ((SetExpr (union x y))
     (lambda (repr) (append ((SetExpr x) repr) ((SetExpr y) repr))))
    ))

(define neg -)


(define (PointsFromFileWhdr x) (load-points-from-file x #t))
(define (PointsFromFileWhdr* x) (load-points-from-file* x #t))
(define PointsFromFile* load-points-from-file*)
(define PointsFromFile load-points-from-file)

(define (LoadTree topology-filename points-filename label)
  (load-layer-tree 4 topology-filename points-filename label))

(define (LayerProjection label r source target target-layers weights output-dir) 
  (layer-tree-projection label
                         (source 'tree) (target 'list) target-layers weights
                         r my-comm myrank mysize output-dir))
(define (SegmentProjection label r source target) 
  (segment-projection label
                      (source 'tree) (target 'list) 
                      r my-comm myrank mysize))
(define (Projection label r source target) 
  (projection label
              (source 'tree) (target 'list) 
              r my-comm myrank mysize))

(define forest (opt 'forest))


;; Post-synaptic cells
(define PostSyns
  (let* (
	 (forest-pts (car (PointsFromFileWhdr* (make-pathname (opt 'trees-dir) (make-pathname (number->string forest) "GCcoordinates.dat")))))
	 
	 (forest-size (kd-tree-size forest-pts))
	 
	 (forest-layout
	  (kd-tree-fold-right*
	   (lambda (i p ax) 
	     (cons (list (inexact->exact i) p) ax))
	   '() forest-pts))
	 
	 (forest-dendrites
	  (fold-right
	   (match-lambda* 
	    (((i p) lst)
	     (let ((li (- i (* (- forest 1) 1000))))
	       (cons
		(LoadTree (sprintf "~A/~A/DGC_dendrite_topology_~A.dat" 
				   (opt 'trees-dir) forest
				   (fmt #f (pad-char #\0 (pad/left 6 (num (- li 1))))))
			  (sprintf "~A/~A/DGC_dendrite_points_~A.dat" 
				   (opt 'trees-dir) forest
				   (fmt #f (pad-char #\0 (pad/left 6 (num (- li 1))))))
			  'Dendrites)
		lst))))
	   '() forest-layout))
	 )
    
    (fold-right
     (match-lambda*
      (((gid p) dendrite-tree lst)
					;(print "gid = " gid)
					;(print "dendrite-tree = ") (pp ((dendrite-tree 'nodes)))
       (cons (make-cell 'DGC gid p (list (cons 'Dendrites dendrite-tree))) lst)))
     '()
     forest-layout
     forest-dendrites
     ))
  )


;; Connection points for perforant path synapses
(define PPCells
  (let* (
         (pp-cell-params (or (opt 'pp-cells) (list 1 1000)))
         (pp-cell-prefix (or (opt 'pp-cell-prefix) "PPCell"))

         (n-modules (car pp-cell-params))
         (n-pp-cells-per-module (cadr pp-cell-params))


         (pp-contacts
	  (let recur ((gid 0) (modindex 1) (lst '()))
            (if (<= modindex n-modules)
                (let inner ((gid gid) (cellindex 1) (lst lst))
                  (if (<= cellindex n-pp-cells-per-module)
		      (let ((root (modulo gid mysize)))
			(if (= myrank root)
			    (inner (+ gid 1)
				   (+ cellindex 1)
				   (cons
				    (list (+ 1 gid)
					  (kd-tree->list*
					   (car
					    (PointsFromFileWhdr
					     (make-pathname (opt 'presyn-dir) 
							    (make-pathname (fmt #f (pad-char #\0 (pad/left 2 (num modindex))))
									   (sprintf "~A_~A.dat" pp-cell-prefix
										    (fmt #f (pad-char #\0 (pad/left 4 (num cellindex)))))
									   )))
					    )))
				    lst))
			    (inner (+ gid 1) (+ cellindex 1) lst)))
                      (recur gid (+ 1 modindex) lst)))
                lst)
            ))
	 )

    (fold-right
      (match-lambda*
       (((gid pp-contacts) lst)
        (if (> (length pp-contacts) 0)
	    (cons (make-cell 'PPCell gid (car pp-contacts) (list (cons 'PPsynapses pp-contacts))) lst)
	    lst)))
      `()
      pp-contacts
      )
    ))



(define PPprojection-forest

    (let* (
	   (target (SetExpr (section Postsyns Dendrites)))
	   (source (SetExpr (section PPCells PPsynapses)))
	   (output-dir (make-pathname (opt 'output-dir) (number->string forest)))
           (weights (let* (
                           (in (open-input-file (opt 'weights)))
                           (data
                            (let ((lines (read-lines in)))
                              (close-input-port in)
                              (map (compose string->number string-trim-both) lines)))
                           )
                      (list->f64vector data)))
	   )
      
      (print "weights size = " (f64vector-length weights))
      
      (if (= myrank 0)
	  (create-directory output-dir))

      (let ((PPtoForest (LayerProjection (opt 'label) (opt 'radius) source target (opt 'layers) weights output-dir)))
	PPtoForest))
    )
  

(MPI:finalize)
