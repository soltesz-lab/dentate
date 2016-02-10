(use srfi-1 mathh matchable kd-tree mpi getopt-long fmt npcloud)
(include "mathh-constants")

(define (choose lst n) (list-ref lst (random n)))

(define local-config (make-parameter '()))
(define trees-directory (make-parameter '()))


(MPI:init)

(define opt-grammar
  `(
    (random-seeds "Use the given seeds for random number generation"
                  (value (required SEED-LIST)
                         (transformer ,(lambda (x) (map string->number
                                                        (string-split x ","))))))
    
    (grid-cells "Specify number of grid cell modules and grid cells per module, separated by colon"
                (value (required "N-MOD:N-GRID-CELL")
                       (transformer ,(lambda (x) (map string->number (string-split x ":"))))))
    
    (output-dir "Write results in the given directory"
               (single-char #\o)
               (value (required PATH)))

    (trees-dir "Load post-synaptic trees from the given directory"
               (single-char #\t)
               (value (required PATH)))
    
    (presyn-dir "Load pre-synaptic location coordinates from the given directory"
                   (single-char #\p)
                   (value (required PATH)))
    
    (radius "Connection radius"
            (single-char #\r)
            (value (required RADIUS)
                   (transformer ,(lambda (x) (string->number x))))
            )
    
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

(define random-seeds (make-parameter (apply circular-list (or (opt 'random-seeds) (list 13 17 19 23 29 37)))))

(define (randomSeed)
  (let ((v (car (random-seeds))))
     (random-seeds (cdr (random-seeds)))
     v))
(define randomInit random-init)

(define randomNormal random-normal)
(define randomUniform random-uniform)

(define (PointsFromFileWhdr x) (load-points-from-file x #t))
(define (PointsFromFileWhdr* x) (load-points-from-file* x #t))
(define PointsFromFile* load-points-from-file*)
(define PointsFromFile load-points-from-file)

(define (LoadTree topology-filename points-filename label)
  (load-layer-tree 4 topology-filename points-filename label))

(define (LayerProjection label r source target target-layers output-dir) 
  (layer-tree-projection label
                         (source 'tree) (target 'list) target-layers
                         r my-comm myrank mysize output-dir))
(define (SegmentProjection label r source target) 
  (segment-projection label
                      (source 'tree) (target 'list) 
                      r my-comm myrank mysize))
(define (Projection label r source target) 
  (projection label
              (source 'tree) (target 'list) 
              r my-comm myrank mysize))

;; Dentate granule cells
(define DGCs
  (let* (
         (DGCpts (car (PointsFromFileWhdr* (make-pathname (opt 'trees-dir)  "GCcoordinates.dat"))))

         (DGCsize (kd-tree-size DGCpts))

         (DGClayout
          (kd-tree-fold-right*
           (lambda (i p ax) (cons (list i p) ax))
           '() DGCpts))

         (DGCdendrites
	  (let recur ((myindex (- DGCsize 1))
		      (lst '()))
	    (if (>= myindex 0)
		(recur (- myindex 1)
		       (cons
			(LoadTree (sprintf "~A/DGC_dendrite_topology_~A.dat" 
					   (opt 'trees-dir) 
					   (fmt #f (pad-char #\0 (pad/left 6 (num myindex)))))
				  (sprintf "~A/DGC_dendrite_points_~A.dat" 
					   (opt 'trees-dir) 
					   (fmt #f (pad-char #\0 (pad/left 6 (num myindex)))))
				  'Dendrites)
			lst))
		lst)
		))
	 )

    (fold-right
      (match-lambda*
       (((gid p) dendrite-tree lst)
        (cons (make-cell 'DGC gid p (list (cons 'Dendrites dendrite-tree))) lst)))
      '()
      DGClayout
      DGCdendrites
      )
    ))

(print "loaded DGCs")

;; Connection points for grid cell perforant path synapses
(define GridCells
  (let* (
         (grid-cell-params (or (opt 'grid-cells) (list 1 1000)))

         (n-modules (car grid-cell-params))
         (n-grid-cells-per-module (cadr grid-cell-params))

         (pp-contacts
	  (let recur ((gid (+ 1 myrank)) (modindex 1) (lst '()))
            (if (<= modindex n-modules)
                (let inner ((gid gid) (cellindex myrank) (lst lst))
                  (print "gid = " gid " cellindex = " cellindex)
                  (if (< cellindex n-grid-cells-per-module)
                      (inner (+ gid mysize)

                             (+ cellindex mysize)
                             (cons
                              (list gid 
                                     (kd-tree->list*
                                      (car
                                       (PointsFromFileWhdr
                                        (make-pathname (opt 'presyn-dir) 
                                                       (make-pathname (fmt #f (pad-char #\0 (pad/left 2 (num modindex))))
                                                                      (sprintf "GridCell_~A.dat" 
                                                                               (fmt #f (pad-char #\0 (pad/left 4 (num (+ 1 cellindex))))))
                                                                      )))
                                       )))
                              lst))
                      (recur gid (+ 1 modindex) lst)))
                lst)
            )) 
	 )

    (fold-right
      (match-lambda*
       (((gid pp-contacts) lst)
        (print "grid cell gid = " gid)
        (if (> (length pp-contacts) 0)
	    (cons (make-cell 'GridCell gid (car pp-contacts) (list (cons 'PPsynapses pp-contacts))) lst)
	    lst)))
      `()
      pp-contacts
      )
    ))

(print "loaded Grid cells")



(define PPtoDGC_projection
  (let* ((target (SetExpr (section DGCs Dendrites)))
         (source (SetExpr (section GridCells PPsynapses)))
        )
    (let ((PPtoDGC (LayerProjection 'PPtoDGC (opt 'radius) source target '(2 3) (opt 'output-dir))))
      PPtoDGC)))

(MPI:finalize)
