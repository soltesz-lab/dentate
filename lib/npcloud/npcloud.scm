;;
;; Spatial and geometric connectivity utility procedures.
;;
;; Copyright 2016 Ivan Raikov.
;;
;; This program is free software: you can redistribute it and/or
;; modify it under the terms of the GNU General Public License as
;; published by the Free Software Foundation, either version 3 of the
;; License, or (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful, but
;; WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;; General Public License for more details.
;;
;; A full copy of the GPL license can be found at
;; <http://www.gnu.org/licenses/>.
;;

(module npcloud

        *


        (import scheme chicken)

        
        (require-extension datatype matchable regex
                           mpi mathh typeclass kd-tree 
                           digraph graph-dfs)


        (require-library srfi-1 srfi-4 srfi-13 irregex files posix data-structures)


        (import 
                (only srfi-1 
                      fold fold-right filter-map filter every zip list-tabulate delete-duplicates partition 
                      first second third take drop concatenate)
                (only srfi-4 
                      s32vector s32vector-length s32vector-ref s32vector-set! make-s32vector
                      f64vector f64vector? f64vector-ref f64vector-set! f64vector-length f64vector->list list->f64vector make-f64vector
                      f64vector->blob u32vector->blob list->u32vector )
                (only srfi-13 string= string< string-null? string-prefix? string-trim-both)
                (only irregex string->irregex irregex-match)
                (only files make-pathname)
                (only posix glob find-files)
                (only extras read-lines pp fprintf )
                (only ports with-output-to-port )
                (only data-structures ->string alist-ref compose identity string-split merge sort atom?)
                (only lolevel extend-procedure procedure-data extended-procedure?)
                )

        (define npcloud-verbose (make-parameter 0))


        (define (d fstr . args)
          (let ([port (current-error-port)])
            (if (positive? (npcloud-verbose)) 
                (begin (apply fprintf port fstr args)
                       (flush-output port) ) )))


        (include "mathh-constants")

        (define (find-index pred lst)
          (let recur ((lst lst) (i 0))
            (cond ((null? lst) #f)
                  ((pred (car lst)) i)
                  (else (recur (cdr lst) (+ i 1)))
                  )))

        (define (sign x) (if (negative? x) -1.0 1.0))

        (define (f64vector-empty? x) (zero? (f64vector-length x)))

        (import-instance (<KdTree> KdTree3d)
                         (<Point> Point3d))
        
        ;; convenience procedure to access to results of kd-tree-nearest-neighbor queries
        (define (kdnn-point x) (cadr x))
        (define (kdnn-distance x) (caddr x))
        (define (kdnn-index x) (caar x))
        (define (kdnn-parent-index x) (car (cadar x)))
        (define (kdnn-parent-distance x) (cadr (cadar x)))


        (define point->list f64vector->list)
        
        
        (define-record-type pointset (make-pointset prefix id points boundary)
          pointset? 
          (prefix pointset-prefix)
          (id pointset-id)
          (points pointset-points)
          (boundary pointset-boundary)
          )

        (define-record-type swcpoint (make-swcpoint id type coords radius pre)
          swcpoint? 
          (id swcpoint-id)
          (type swcpoint-type)
          (coords swcpoint-coords)
          (radius swcpoint-radius)
          (pre swcpoint-pre)
          )

        (define-record-type layer-point (make-layer-point id coords radius seci layer)
          layer-point? 
          (id layer-point-id)
          (coords layer-point-coords)
          (radius layer-point-radius)
          (seci layer-point-section-index)
          (layer layer-point-layer)
          )
	
	(define-record-printer (layer-point x out)
	  (fprintf out "#(layer-point id=~A coord=~A radius=~A seci=~A layer=~A)"
		   (layer-point-id x)
		   (layer-point-coords x)
		   (layer-point-radius x)
		   (layer-point-section-index x)
		   (layer-point-layer x)
		   ))
        
        (define-record-type cell (make-cell ty index origin sections)
          cell? 
          (ty cell-type)
          (index cell-index)
          (origin cell-origin)
          (sections cell-sections)
          )
        
	
        
        (define (cell-section-ref name cell)
          (let ((v (alist-ref name (cell-sections cell))))
            (if (not v) (error 'cell-section-ref "unable to find section" name cell)
                v)))

        
        (define (write-pointset name pts)
            (call-with-output-file (sprintf "~A.pointset.dat" name)
              (lambda (out)
                (for-each (match-lambda
                           ((gid p)
                            (fprintf out "~A ~A ~A~%" 
                                     (coord 0 p)
                                     (coord 1 p)
                                     (coord 2 p))))
                          pts))
              ))

        
        (define (write-layout name pts #!optional rank)
            (call-with-output-file (if rank 
                                       (sprintf "~A.~A.layout.dat" name rank)
                                       (sprintf "~A.layout.dat" name))
              (lambda (out)
                (for-each (match-lambda
                           ((gid p)
                            (fprintf out "~A ~A ~A ~A~%" 
                                     gid
                                     (coord 0 p)
                                     (coord 1 p)
                                     (coord 2 p))))
                          pts))
              ))




        (define (cells-sections->kd-tree cells section-name 
                                         #!key 
                                         (make-value (lambda (i v) (list (car v) 0.0)))
                                         (make-point (lambda (v) (cadr v))))
          (let ((t 
                 (let recur ((cells cells) (points '()))
                   (if (null? cells)
                       (list->kd-tree
                        points
                        make-value: make-value
                        make-point: make-point)
                       (let* ((cell (car cells))
			      (cell-i (cell-index cell)))
                         (recur (cdr cells) 
                                (let inner ((sections (append (cell-section-ref section-name cell)))
                                            (points points))
                                  (if (null? sections) points
                                      (inner
                                       (cdr sections)
                                       (append (map (lambda (p) (list cell-i p)) (cdr (car sections))) points))
                                      ))
                                ))
                       ))
                 ))
            t))



        (define (pointCoord axis p)
          (coord (inexact->exact axis) p))



        (define (layer-point-projection prefix my-comm myrank size cells layers fibers
					zone cell-start fiber-start)

          (d "rank ~A: prefix = ~A zone = ~A layers = ~A length cells = ~A~%" 
             myrank prefix zone layers (length cells))

          (fold (lambda (cell ax)

                  (d "rank ~A: cell gid = ~A~%" myrank (car cell))

                  (let* ((gid (+ cell-start (car cell)))
                         (root (modulo gid size))
                         (sections (cadr cell)))
                    
                    (fold 
                     
                     (lambda (secg ax)
                       (let ((query-data
                              ((secg 'fold-nodes)
                               (lambda (i lp ax)
                                 (d "rank ~A: querying point ~A (coords ~A) (layer ~A) (section ~A)~%" 
                                    myrank i (layer-point-coords lp) 
                                    (layer-point-layer lp)
				    (layer-point-section-index lp))
                                 (fold
                                  (lambda (x ax) 
                                    (let (
                                          (source (car x))
                                          (target gid)
                                          (distance (cadr x))
                                          (layer (layer-point-layer lp))
                                          (section (layer-point-section-index lp))
                                          )
                                      (if (member layer layers)
                                          (append (list source target distance layer section i) ax)
                                          ax)
                                      ))
                                  ax
				  
                                  (delete-duplicates
                                   (map (lambda (x) 
                                          (d "rank ~A: query result = ~A (~A) (~A) ~%" 
                                             myrank (kdnn-point x) (kdnn-distance x) (kdnn-parent-index x))
                                          (list (+ fiber-start (kdnn-parent-index x))
                                                (+ (kdnn-distance x) (kdnn-parent-distance x))
                                                ))
                                        (kd-tree-near-neighbors* fibers zone (layer-point-coords lp)))
                                   (lambda (u v) (= (car u) (car v)))
                                   )
                                  ))
                               '()))
                             )
			 (MPI:barrier my-comm)
			 (d "rank ~A: cell = ~A root = ~A: before gatherv~%" myrank cell root)

                         (let* ((res0 (MPI:gatherv-f64vector (list->f64vector query-data) root my-comm))
                                
                                (res1 (or (and (= myrank root) (filter (lambda (x) (not (f64vector-empty? x))) res0)) '())))
			   (d "rank ~A: cell = ~A: after gatherv~%" myrank cell)
                           (append res1 ax))
                         
                         ))
                     ax sections)
                    ))
                '() cells)
          )



        

        (define (point-projection prefix my-comm myrank size pts fibers zone point-start nn-filter)
          (let ((tbl (make-hash-table = number-hash)))
            (for-each 
             (lambda (px)
               
               (printf "~A: rank ~A: px = ~A zone=~A ~%"  prefix myrank px zone)
               
                  (let* ((i (+ point-start (car px)))
                         (root (modulo i size))
                         (dd (d "~A: rank ~A: querying point ~A (root ~A)~%" prefix myrank px root))
                         (query-data
                          (let ((pd (cadr px))) 
                            (fold
                             (lambda (x ax) 
                               (let ((source (car x))
                                     (target i)
                                     (distance (cadr x)))
                                 (if (and (> distance  0.) (not (= source target)))
                                     (append (list source target distance) ax)
                                     ax)
                                 ))
                             '()
                             (delete-duplicates
                              (map (lambda (x) 
                                     (let ((res (list (car (cadar x))  
                                                      (+ (caddr x) (cadr (cadar x))))))
                                       (d "~A: x = ~A res = ~A~%" prefix x res)
                                       res))
                                   (nn-filter pd (kd-tree-near-neighbors* fibers zone pd))
                                   )
                              (lambda (u v) (= (car u) (car v)))
                              ))
                            ))
                         )

                    
                    (let* ((res0 (MPI:gatherv-f64vector (list->f64vector query-data) root my-comm))
                           (res1 (or (and (= myrank root) (filter (lambda (x) (not (f64vector-empty? x))) res0)) '())))

		      (if (= myrank root)
			  (for-each 
			   (lambda (vect) 
			     (let* ((entry-len 3)
				    (data-len (/ (f64vector-length vect) entry-len)))
			       
			       (printf "~A: rank ~A: px = ~A data-len=~A ~%"  prefix myrank px data-len)
			       
			       (let recur ((k 0))
				 (if (< k data-len)
				     (let ((source (inexact->exact (f64vector-ref vect (* k entry-len))))
					   (target (inexact->exact (f64vector-ref vect (+ 1 (* k entry-len)))))
					   (distance (f64vector-ref vect (+ 2 (* k entry-len)))))
				       (let ((val (list source distance)))
					 (hash-table-update!/default
					  tbl target (lambda (lst) (merge (list val) lst (lambda (x y) (< (cadr x) (cadr y)))))
					  (list val)))
				       (recur (+ 1 k))
				       )))
			       ))
			   res1))
                      ))
                  )
             pts)
            tbl
            ))

        
        (define comment-pat (string->irregex "^#.*"))


        (define (load-points-from-file filename . header)

          (let ((in (open-input-file filename)))
            
            (if (not in) (error 'load-points-from-file "file not found" filename))

            (let* ((lines
		    (let ((lines (read-lines in)))
		      (close-input-port in)
		      (filter (lambda (line) (not (irregex-match comment-pat line)))
			      lines)))

                   (lines1 (if header (cdr lines) lines))

                   (point-data
                    (filter-map
                     (lambda (line) 
                       (let ((lst (map string->number (string-split line " \t"))))
                         (and (not (null? lst)) (apply make-point lst))))
                     lines1))

                   (point-tree (list->kd-tree point-data))
                   )
              
              (list point-tree)
              
              ))
          )


        (define (load-points-from-file* filename . header)

          (let ((in (open-input-file filename)))
            
            (if (not in) (error 'load-points-from-file "file not found" filename))

            (let* ((lines
		    (let ((lines (read-lines in)))
		      (close-input-port in)
		      (filter (lambda (line) (not (irregex-match comment-pat line)))
			      lines)))

                   (lines1 (if header (cdr lines) lines))

                   (point-data
                    (filter-map
                     (lambda (line) 
                       (let* ((line-data (map string->number (string-split line " \t"))))
			 (if (null? line-data) #f
			     (let*((id (car line-data))
				   (lst (cdr line-data)))
			       (and (not (null? lst)) (list id (apply make-point lst) #f))))))
                     lines1))

                   (point-tree (list->kd-tree* point-data))
                   )
              
              (list point-tree)
              
              ))
          )


        (define (make-swc-tree-graph lst label)
          
          (let* (
                 (g              (make-digraph label #f))
                 (node-info      (g 'node-info))
                 (node-info-set! (g 'node-info-set!))
                 (add-node!      (g 'add-node!))
                 (add-edge!      (g 'add-edge!))
                 )

            ;; insert nodes
            (let recur ((lst lst))

              (if (not (null? lst))

                  (let ((point (car lst)))

                    (let ((node-id (swcpoint-id point)))

                      (add-node! node-id point)

                      (recur (cdr lst))))))

            ;; insert edges
            (let recur ((lst lst))
              
              (if (not (null? lst))
                  
                  (let ((point (car lst)))
                    

                    (let ((node-id (swcpoint-id point))
                          (pre-id  (swcpoint-pre point)))

                      (if (> pre-id 0)

                          (let* ((pre-point   (node-info pre-id))
                                 (pre-coords  (and pre-point (swcpoint-coords pre-point)))
                                 (node-coords (swcpoint-coords point))
                                 (distance    (sqrt (dist2 node-coords pre-coords))))
                            
                            (add-edge! (list pre-id node-id distance))))
                        
                      (recur (cdr lst))
                      ))
                  ))
            g 
            ))


        (define (swc-tree-graph->section-points cell-index cell-origin type g gdistv gsegv)
          
          (let* ((node-info (g 'node-info))
                 (succ      (g 'succ))
                 (offset    (let ((cell-loc (point->list cell-origin))
                                  (root-loc (point->list (swcpoint-coords (node-info 1)))))
                              (map - cell-loc root-loc))))

            (d "swc-tree-graph->section-points: offset = ~A~%" offset)


            (let recur ((n 1) (lst '()))

              (let* (
                     (point (node-info n))
                     (point-type (swcpoint-type point))
                     (point-pre (swcpoint-pre point))
                     (proceed? (or (= point-type type)
                                   (case (swcpoint-type point)
                                     ((0 1 5 6) #t)
                                     (else #f))))
                     )

                (d "swc-tree-graph->section-points: n = ~A point-type = ~A proceed? = ~A~%" 
                   n point-type proceed?)

                (d "swc-tree-graph->section-points: succ n = ~A~%" (succ n))
                  
                (if proceed?

                    (let (
                          (point1 (list
                                   (s32vector-ref gsegv n)
                                   (apply make-point (map + offset (point->list (swcpoint-coords point))))))
                          )

                      (fold (lambda (x ax) (recur x ax))
                            (cons point1 lst)
                            (succ n)))

                    lst)

                ))
            ))


        (define (make-layer-tree-graph topology-sections topology-layers topology points label)
          
          (let* (
                 (g              (make-digraph label topology-layers))
                 (node-info      (g 'node-info))
                 (node-info-set! (g 'node-info-set!))
                 (add-node!      (g 'add-node!))
                 (add-edge!      (g 'add-edge!))
                 )

            ;; insert nodes
            (let recur ((lst points))

              (if (not (null? lst))

                  (let ((point (car lst)))

                    (let ((node-id (layer-point-id point)))

                      (add-node! node-id point)

                      (recur (cdr lst))))))

            ;; insert edges from dendritic topology
            (let recur ((lst (car topology)))

              (if (not (null? lst))
                  
                  (match-let (((dest-id src-id) (car lst)))
              
                             (let* ((dest-point   (node-info dest-id))
                                    (dest-coords  (layer-point-coords dest-point))
                                    (node-point   (node-info src-id))
                                    (node-coords  (layer-point-coords node-point))
                                    (distance     (sqrt (dist2 dest-coords node-coords))))
                               (add-edge! (list src-id dest-id distance))))
                  
                  (recur (cdr lst))
                  ))

            g 
            ))


        (define (tree-graph-distances+segments g nseg)


          (define n        ((g 'capacity)))
          (define distv    (make-f64vector (+ 1 n) -1.0))
          (define rdistv   (make-f64vector (+ 1 n) -1.0))
          (define segv     (make-s32vector (+ 1 n) -1))


          ;; determine distances from root
          (define (traverse-dist es)
            (if (null? es) distv
                (match-let (((i j dist) (car es)))
                  (if (>= (f64vector-ref distv j) 0.0)
                      (traverse-dist (cdr es))
                      (let ((idist (f64vector-ref distv i)))
                        (f64vector-set! distv j (+ idist dist))
                        (let ((distv1 (traverse-dist ((g 'out-edges) j))))
                          (traverse-dist es)))
                      ))
                ))
          
	 
          ;; determine distances from end (reverse distance)
          (define (traverse-rdist es)
            (if (null? es) rdistv
                (match-let (((i j dist) (car es)))
                  (if (>= (f64vector-ref rdistv i) 0.0)
                      (traverse-rdist (cdr es))
                      (let ((jdist (f64vector-ref distv j)))
                        (f64vector-set! rdistv i (+ jdist dist))
                        (let ((rdistv1 (traverse-rdist ((g 'in-edges) i))))
                          (traverse-rdist es)))
                      ))
                ))


          (define (compute-segv distv rdistv)
            (let recur ((n n))
              (if (>= n 1)
                  (let* ((dist  (f64vector-ref distv n))
                         (rdist (f64vector-ref rdistv n))
                         (len   (and (positive? dist) (positive? rdist) (+ dist rdist)))
                         (delta (and len (round (/ len nseg))))
                         (seg   (and delta (round (/ dist delta)))))
                    (if seg (s32vector-set! segv n (exact->inexact seg)))
                    (recur (- n 1))
                    ))
              ))
          
          (let ((in-edges (g 'in-edges)) 
                (out-edges (g 'out-edges)) 
                (terminals ((g 'terminals)))
                (roots ((g 'roots))))
            (for-each (lambda (x) (f64vector-set! distv x 0.0)) roots)
            (for-each (lambda (x) (s32vector-set! segv x 0)) roots)
            (for-each (lambda (x) (f64vector-set! rdistv x 0.0)) terminals)
            (traverse-dist (concatenate (map (lambda (x) (out-edges x)) roots)))
            (traverse-rdist (concatenate (map (lambda (x) (in-edges x)) terminals)))
            (compute-segv distv rdistv)
            (list distv segv)
          ))


  
        (define (load-swc filename label type nseg)
          
          (let ((in (open-input-file filename)))
            
            (if (not in) (error 'load-swc "file not found" filename))
            
            (let* (
                   (lines
                    (let ((lines (read-lines in)))
                      (close-input-port in)
                      (filter (lambda (line) (not (irregex-match comment-pat line))) 
                              lines)))

                   (swc-data
                    (filter-map
                     (lambda (line) 
                       (let ((lst (map string->number (string-split line " \t"))))
                         (and (not (null? lst)) 
                              (match-let (((id my-type x y z radius pre) lst))
                                         (make-swcpoint id my-type (make-point x y z)
                                                        radius pre)))
                         ))
                     lines))

                   (swc-graph (make-swc-tree-graph swc-data label))

                   (dist+segs  (tree-graph-distances+segments swc-graph nseg))

                   )

              (cons type (cons swc-graph dist+segs)))
          ))



        (define (load-swcdir path label type nseg)
          
          (let ((pat ".*.swc"))

            (let ((flst (find-files path
                                    test: (regexp pat)
                                    action: cons
                                    seed: (list) 
                                    limit: 0)))

              (d "load-swcdir: flst = ~A~%" flst)

              (map (lambda (fn) (load-swc fn label type nseg)) (sort flst string<?))
              ))
          )


        (define (load-matrix-from-lines lines)
          (let ((dimensions-line (car lines)))
            (match-let (((m n) (map string->number (string-split dimensions-line " \t"))))
                       (let ((matrix-lines (take (cdr lines) m))
                             (rest-lines (drop (cdr lines) m)))
                         `(,m ,n
                              ,(map (lambda (line) (map string->number (string-split line " \t"))) matrix-lines)
                              ,rest-lines)))
            ))

          
        (define (load-layer-tree nlayers topology-filename points-filename label)
	  (let* (
		 (topology-lines
		  (let* ((topology-in (open-input-file topology-filename))
			 (lines (if (not topology-in)
				    (error 'load-layer-tree "topology file not found" topology-filename)    
				    (read-lines topology-in))))
		    (close-input-port topology-in)
		    (filter (lambda (line) (not (irregex-match comment-pat line))) lines)))
		 
		 (points-lines
		  (let* ((points-in (open-input-file points-filename))
			 (lines (if (not points-in) 
				    (error 'load-layer-tree "points file not found" points-filename)
				    (read-lines points-in))))
		    (close-input-port points-in)
		    (filter (lambda (line) (not (irregex-match comment-pat line))) lines)))
		 
		 (topology-layers
		  (cadr
		   (let ((layer-lines (take topology-lines nlayers)))
		     (fold-right (match-lambda* 
				  ((line (i lst))
				   (match-let (((nsecs . sec-ids) (map string->number (string-split line " \t"))))
					      (if (= (length sec-ids) nsecs)
						  (list (+ 1 i) (cons sec-ids lst))
						  (error 'load-layer-tree "number of sections mismatch in layer description" nsecs sec-ids))
					      )))
				 '(0 ()) layer-lines))))
		 
		 (topology-sections
		  (let ((rest-lines (drop topology-lines nlayers)))
		    (let ((points-sections-line (car rest-lines)))
		      (match-let (((nsections . point-numbers) (map string->number (string-split points-sections-line " \t"))))
				 (if (not (= nsections (length point-numbers)))
				     (error 'load-layer-tree "number of sections mismatch in section description"))
				 point-numbers))
		    ))
		 
		 (topology-data
		  (let ((rest-lines (drop topology-lines (+ 1 nlayers))))
		    (match-let (((mdend ndend tdend rest-lines) (load-matrix-from-lines rest-lines)))
			       (if (not (= ndend 2)) 
				   (error 'load-layer-tree "invalid dendrite topology dimensions" mdend ndend))
			       (match-let (((msoma nsoma tsoma rest-lines) (load-matrix-from-lines rest-lines)))
					  (if (not (= nsoma 2)) 
					      (error 'load-layer-tree "invalid soma topology dimensions" msoma nsoma))
					  (list tdend tsoma)
					  ))
		    ))
		 
		 (points-lines 
		  (let ((points-header (map string->number (string-split (car points-lines) " \t"))))
		    (if (not (= (car points-header) (length (cdr points-lines))))
			(error 'load-layer-tree "number of points mismatch in points matrix" points-header))
		    (cdr points-lines)))
		 
		 (points-data
		  (let recur ((id 0) (pts '()) (seci 0) (secs topology-sections) (lines points-lines))
		    (if (null? secs) pts
			(let* (
			       (npts1 (car secs))
			       (id.pts1 (fold 
					 (match-lambda* 
					  ((line (id . lst))
					   (let ((pt (map string->number (string-split line " \t"))))
					     (match-let (((x y z radius) pt))
							(let ((layer (find-index (lambda (layer) (member seci layer)) topology-layers)))
							  (cons (+ 1 id)
								(cons (make-layer-point id (make-point x y z) radius seci layer) lst))
							  ))
					     )))
					 (cons id pts)
					 (take lines npts1)))
			       )
			  (recur (car id.pts1) (cdr id.pts1) (+ 1 seci) (cdr secs) (drop lines npts1))
			  ))
		    ))
		 
		 (tree-graph (make-layer-tree-graph topology-sections topology-layers topology-data points-data label))
		 )
	    
	    tree-graph
	    
	    ))



        (define (layer-tree-projection label source-tree target-sections target-layers zone my-comm myrank size output-dir)

          (MPI:barrier my-comm)
	  
          (let ((my-results
                 (layer-point-projection label my-comm myrank size target-sections target-layers source-tree zone 0 0)))

            (MPI:barrier my-comm)

            (call-with-output-file (make-pathname output-dir (sprintf "~A.~A.dat"  label (if (> size 1) myrank "")))
              (lambda (out)
		(for-each 
		 (lambda (my-data)
		   (let* ((my-entry-len 6)
			  (my-data-len (/ (f64vector-length my-data) my-entry-len)))
		     (d "rank ~A: length my-data = ~A~%" myrank my-data-len)
		     (let recur ((m 0))
		       (if (< m my-data-len)
			   (let* (
				  (my-entry-offset (* m my-entry-len))
				  (source   (inexact->exact (f64vector-ref my-data my-entry-offset)))
				  (target   (inexact->exact (f64vector-ref my-data (+ 1 my-entry-offset))))
				  (distance (f64vector-ref my-data (+ 2 my-entry-offset)))
				  (layer    (inexact->exact (f64vector-ref my-data (+ 3 my-entry-offset))))
				  (section  (inexact->exact (f64vector-ref my-data (+ 4 my-entry-offset))))
				  (node     (inexact->exact (f64vector-ref my-data (+ 5 my-entry-offset))))
				  )
			     (fprintf out "~A ~A ~A ~A ~A ~A~%" source target distance layer section node)
			     (recur (+ 1 m)))))
		     ))
		 my-results)))
            ))


        (define (projection label source-tree target zone maxn my-comm myrank size output-dir) 

          (MPI:barrier my-comm)
	  
          (let ((my-results (point-projection label my-comm myrank size
					      target source-tree zone 0
					      (lambda (x nn) nn))))

            (MPI:barrier my-comm)
            
            (call-with-output-file (make-pathname output-dir (sprintf "~A.~A.dat"  label (if (> size 1) myrank "")))
              (lambda (out)
                (hash-table-for-each 
                 my-results
                 (lambda (target lst)
                   (let ((lst-len (length lst)))
                     (printf "~A: rank ~A: target ~A length lst = ~A~%" label myrank target lst-len)
                     (let ((lst1 (if (and (> maxn 0) (> lst-len maxn))
                                     (take lst maxn) lst)))
                       (for-each
                        (lambda (x) 
                          (let ((source (list-ref x 0))
                                (distance (list-ref x 1)))
                            (fprintf out "~A ~A ~A~%" source target distance))
                          )
                        lst1
                        ))
                     ))
                 ))
              ))
          )
        
)
