
(module spikedata

        (read-spike-times
         spike-stats)

        (import scheme chicken)

(require-extension matchable typeclass rb-tree)
(require-library srfi-1 srfi-13 irregex data-structures files posix extras ploticus)
(import
 (only srfi-1 filter filter-map list-tabulate fold)
 (only srfi-13 string-trim-both string-null?)
 (only files make-pathname)
 (only posix glob)
 (only data-structures ->string alist-ref compose string-split)
 (only extras fprintf random read-lines)
 (only mathh cosh tanh log10)
 (only irregex irregex-match string->irregex)
 )

(define comment-pat (string->irregex "^#.*"))

(define (sample n v)
  (let ((ub (vector-length v)))
    (let ((idxs (list-tabulate n (lambda (i) (random ub)))))
      (map (lambda (i) (vector-ref v i)) idxs)
      ))
    )


(define list-fold fold)

(define (read-spike-times spike-file . rest)

  (match-let

   (((data tmax nmax)
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
      (cons spike-file rest)
      )))

  (let ((m (rb-tree-map -)))
    (list
     (with-instance ((<PersistentMap> m))
                    (list-fold
                     (lambda (t.n msp) 
                       (update msp (car t.n) (cdr t.n) append))
                     (empty) data))
     nmax tmax))
  ))


(define (diffs xs)
  (if (null? xs) '()
      (reverse
       (cadr
        (fold (match-lambda* ((x (prev lst)) (list x (cons (- x prev) lst))))
              (list (car xs) '()) (cdr xs))
        ))
      ))

(define (sum xs) (fold + 0.0 xs))

(define (mean xs)
  (if (null? xs) 0.0
      (/ (sum xs) (length xs))))

(define (square x) (* x x))

(define (variance xs)
  (if (< (length xs) 2)
      (error "variance: sequence must contain at least two elements")
      (let ((mean1 (mean xs)))
        (/ (sum (map (lambda (x) (square (- mean1 x))) xs))
           (- (length xs) 1)))))


(define (spike-stats data nmax tmax)
  (let* (
         ;; event times per node
         (event-times
          (let ((v (make-vector nmax '())))
            (for-each (match-lambda
                       ((t . ns)
                        (for-each (lambda (n) (vector-set! v (- n 1) (cons t (vector-ref v (- n 1)))))
                                  ns)))
                      (reverse data))
            v))
         
         (event-intervals (map diffs (filter pair? (vector->list event-times))))
         (mean-event-intervals (map mean event-intervals))
         (mean-event-interval (mean mean-event-intervals))
         (stdev-event-interval (if (null? mean-event-intervals) 0.0 
                                   (sqrt (variance mean-event-intervals))))
         (cv-event-interval (if (zero? mean-event-interval) 0.0
                                (/ stdev-event-interval mean-event-interval)))
         
         (nevents (filter-map (lambda (x) (and (not (null? x)) (length (cdr x)))) (vector->list event-times)))
         (mean-rates (map (lambda (x) (* 1000 (/ x tmax))) nevents))
         (mean-event-frequency (round (mean mean-rates)))
         
         )

     `(
       (nmax . ,nmax)
       (tmax . ,tmax)
       (mean-nevents         . ,(mean nevents))
       (mean-event-frequency . ,mean-event-frequency)
       (mean-event-interval  . ,mean-event-interval)
       (stdev-event-interval . ,stdev-event-interval)
       (cv-event-interval    . ,cv-event-interval)
       )
    
     )
  )


)
