(use fmt npcloud lmdb files posix srfi-4)


(define (main args)

  (print args)

  (let ((fname (car args))
        (offset (string->number (cadr args)))
        (ids (map string->number (cddr args))))
    
    (let ((dbs (trees-opendb/lmdb fname)))

      (for-each
       (lambda (id)
         (let (
               (topology-filename (sprintf "DGC_dendrite_topology_~A.dat" 
                                           (fmt #f (pad-char #\0 (pad/left 6 (num id))))))
               (points-filename (sprintf "DGC_dendrite_points_~A.dat" 
                                         (fmt #f (pad-char #\0 (pad/left 6 (num id))))))
               (spines-filename (sprintf "DGC_spine_density_~A.dat" 
                                         (fmt #f (pad-char #\0 (pad/left 6 (num id))))))
               )
         (let ((tree-data (cons id (load-layer-tree-data 4 topology-filename points-filename spines-filename))))
           (layer-tree-write/lmdb dbs tree-data))))
       ids)

      (trees-closedb/lmdb dbs))

    ))
    

(main (command-line-arguments))
