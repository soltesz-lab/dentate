(use srfi-1 mathh matchable kd-tree fmt npcloud)
(include "mathh-constants")

(define (choose lst n) (list-ref lst (random n)))

(define trees-dir "/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/GC/1")
(define myindex 78)


(define (LoadTree topology-filename points-filename label)
  (load-layer-tree 4 topology-filename points-filename label))

(define DGCdendrites
  (LoadTree (sprintf "~A/DGC_dendrite_topology_~A.dat" 
		     trees-dir
		     (fmt #f (pad-char #\0 (pad/left 6 (num myindex)))))
	    (sprintf "~A/DGC_dendrite_points_~A.dat" 
		     trees-dir
		     (fmt #f (pad-char #\0 (pad/left 6 (num myindex)))))
	    'Dendrites)
  )

(pp ((DGCdendrites 'nodes)))

