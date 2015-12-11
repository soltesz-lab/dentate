(use srfi-1 mathh matchable kd-tree mpi getopt-long picnic-utils fmt)
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
    
    (trees-directory "Load trees from the given directory"
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
    (picnic-verbose 1))

(if (picnic-verbose)
    (pp (local-config) (current-error-port)))

(define my-comm (MPI:get-comm-world))
(define myrank  (MPI:comm-rank my-comm))
(define mysize  (MPI:comm-size my-comm))

(define-syntax
  SetExpr
  (syntax-rules
      (population section union)
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
    ((SetExpr (section p t))
     (lambda (repr)
       (case repr
         ((list)
          (map (lambda (cell) 
                 (list (cell-index cell) 
                       (cell-section-ref (quote t) cell)))
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

(define PointsFromFile load-points-from-file)

(define (LoadTree topology-filename points-filename label)
  (load-layer-tree 4 topology-filename points-filename label 'GC))

(define (SegmentProjection label r source target) 
  (segment-projection label
                      (source 'tree) (target 'list) 
                      r my-comm myrank mysize))
(define (Projection label r source target) 
  (projection label
              (source 'tree) (target 'list) 
              r my-comm myrank mysize))


(define GCs
  (let* (
         (GCpts (car (PointsFromFile (make-pathname (opt 'trees-directory)  "GCcoordinates.dat"))))

         (GCsize (kd-tree-size GCpts))

         (GClayout
          (kd-tree-fold-right*
           (lambda (i p ax) (if (= (modulo i mysize) myrank) (cons p ax) ax))
           '() GCpts))

         (GCdendrites
          (let recur ((myindex GCsize) (ax '()))
            (let ((ax1 (if (< myindex 1) ax
                           (if (= (modulo myindex mysize) myrank)
                               (cons (LoadTree (sprintf "~A/DGC_dendrite_topology_~A.dat" 
                                                        (opt 'trees-directory) 
                                                        (fmt #f (pad-char #\0 (pad/left 6 (num myindex)))))
                                               (sprintf "~A/DGC_dendrite_points_~A.dat" 
                                                        (opt 'trees-directory) 
                                                        (fmt #f (pad-char #\0 (pad/left 6 (num myindex)))))
                                               'GC
                                               )
                                     ax)
                               ax))))
              (recur (- myindex 1) ax1))))
         )

    (fold-right
      (match-lambda*
        (((gid p) dendrite-tree)
         (cons (make-cell 'GC gid p (list dendrite-tree)) lst)))
      '()
      GClayout
      GCdendrites
      )
    ))



#|
(define GridCells
  (let* ((PPlayouts
           (let* ((pts (kd-tree->list*
                         (car (let ((comp76.comp75.s
                                      (PointsFromFile "GoCcoordinates.dat")))
                                comp76.comp75.s))))
                  (layout pts))
             (if (picnic-write-pointsets) (write-pointset 'GoC pts))
             (if (picnic-write-layouts) (write-layout 'GoC layout))
             layout))
         (GoCBasolateralDendrites139
           (let ((v143 (randomInit 13.0)))
             (let ((result
                     (fold-right
                       (match-lambda*
                         (((gid p142) lst)
                          (match-let
                            (((i pts)
                              (fold (match-lambda*
                                      (((f n) (i lst))
                                       (list (+ i n)
                                             (append
                                               (list-tabulate
                                                 n
                                                 (lambda (j) (list (+ i j 1) (f))))
                                               lst))))
                                    (list (inexact->exact 0) '())
                                    (list (list (lambda ()
                                                  (make-segmented-process
                                                    (comp77.comp75.f gid p142 v143)
                                                    (sample-uniform)
                                                    (inexact->exact 5.0)
                                                    (inexact->exact 5.0)))
                                                (inexact->exact comp77.comp75.n))))))
                            (cons (make-segmented-section
                                    gid
                                    p142
                                    'BasolateralDendrites
                                    pts)
                                  lst))))
                       '()
                       GoC_layout138)))
               (if (picnic-write-sections)
                 (write-sections
                   'GoC
                   'BasolateralDendrites
                   GoC_layout138
                   result))
               result)))
         (GoCApicalDendrites140
           (let ((v145 (randomInit 17.0)))
             (let ((result
                     (fold-right
                       (match-lambda*
                         (((gid p144) lst)
                          (match-let
                            (((i pts)
                              (fold (match-lambda*
                                      (((f n) (i lst))
                                       (list (+ i n)
                                             (append
                                               (list-tabulate
                                                 n
                                                 (lambda (j) (list (+ i j 1) (f))))
                                               lst))))
                                    (list (inexact->exact 2.0) '())
                                    (list (list (lambda ()
                                                  (make-segmented-process
                                                    (comp96.comp75.f gid p144 v145)
                                                    (sample-uniform)
                                                    (inexact->exact 3.0)
                                                    (inexact->exact 5.0)))
                                                (inexact->exact comp96.comp75.n))))))
                            (cons (make-segmented-section
                                    gid
                                    p144
                                    'ApicalDendrites
                                    pts)
                                  lst))))
                       '()
                       GoC_layout138)))
               (if (picnic-write-sections)
                 (write-sections 'GoC 'ApicalDendrites GoC_layout138 result))
               result)))
         (GoCAxons141
           (let ((v147 (randomInit 23.0)))
             (let ((result
                     (fold-right
                       (match-lambda*
                         (((gid p146) lst)
                          (match-let
                            (((i pts)
                              (fold (match-lambda*
                                      (((f n) (i lst))
                                       (list (+ i n)
                                             (append
                                               (list-tabulate
                                                 n
                                                 (lambda (j) (list (+ i j 1) (f))))
                                               lst))))
                                    (list (inexact->exact 4.0) '())
                                    (list (list (lambda ()
                                                  (make-process
                                                    (comp115.comp75.f gid p146 v147)
                                                    (sample-uniform)
                                                    (inexact->exact 2.0)))
                                                (inexact->exact comp115.comp75.n))))))
                            (cons (make-section gid p146 'Axons pts) lst))))
                       '()
                       GoC_layout138)))
               (if (picnic-write-sections)
                 (write-sections 'GoC 'Axons GoC_layout138 result))
               result))))
    (fold-right
      (match-lambda*
        (((gid p)
          GoCBasolateralDendrites139
          GoCApicalDendrites140
          GoCAxons141
          lst)
         (cons (make-cell
                 'GoC
                 gid
                 p
                 (list GoCBasolateralDendrites139
                       GoCApicalDendrites140
                       GoCAxons141))
               lst)))
      '()
      GoC_layout138
      GoCBasolateralDendrites139
      GoCApicalDendrites140
      GoCAxons141)))


(define PPtoGC_projection
  (let ((target (SetExpr (section comp131.GoC ApicalDendrites))))
    (let ((source (SetExpr (section comp131.GC ParallelFibers))))
      (let ((PPtoGC (SegmentProjection 'PPtoGC r source target)))
        PPtoGC))))
|#

(MPI:finalize)
