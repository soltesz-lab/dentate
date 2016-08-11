
(module spikedata

        (read-spike-times
         spike-stats
         levenshtein-distance)

        (import scheme chicken foreign)

(require-extension matchable typeclass rb-tree)
(require-library srfi-1 srfi-4 srfi-13 irregex data-structures files posix extras)
(import
 (only srfi-1 filter filter-map list-tabulate fold)
 (only srfi-4 u32vector)
 (only srfi-13 string-trim-both string-null?)
 (only files make-pathname)
 (only posix glob)
 (only data-structures ->string alist-ref compose string-split merge)
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
        (let ((data0 (read-lines spike-file)))
          (fold (lambda (line ax)
                  (match-let (((data tmax nmax)  ax))
                             (let ((row (map string->number (string-split  line " "))))
                               (let ((data1 (cons row data))
                                     (tmax1 (max (car row) tmax))
                                     (nmax1 (fold max nmax (cdr row))))
                                 (list data1 tmax1 nmax1)))))
                ax data0)))
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
  (cond ((null? xs) '())
	(else
	 (reverse
	  (cadr
	   (fold (match-lambda* ((x (prev lst)) (list x (cons (- x prev) lst))))
		 (list (car xs) '()) (cdr xs))
	   ))
	 ))
  )

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


(define (spike-stats event-times nevents tmax)
  (let* (
         ;; event times per node
         (event-times-lst (map cdr 
				(let ((m (rb-tree-map -)))
				  (with-instance ((<PersistentMap> m)) 
						 (list-values event-times))))
			  )
         (event-intervals (filter pair? (map diffs event-times-lst)))
         (mean-event-intervals (map mean event-intervals))
         (mean-event-interval (mean mean-event-intervals))
         (stdev-event-interval (if (or (null? mean-event-intervals)
                                       (null? (cdr mean-event-intervals)))  0.0 
                                   (sqrt (variance mean-event-intervals))))
         (cv-event-interval (if (zero? mean-event-interval) 0.0
                                (/ stdev-event-interval mean-event-interval)))

         (nevents-lst (filter-map (lambda (x) (and (not (null? x)) (length x))) event-times-lst))
         (mean-rates (map (lambda (x) (* 1000 (/ x tmax))) nevents-lst))
         (mean-event-frequency (round (mean mean-rates)))
         
         )

     `(
       (tmax . ,tmax)
       (ncells               . ,(length event-times-lst))
       (mean-nevents         . ,(mean nevents-lst))
       (mean-event-frequency . ,mean-event-frequency)
       (mean-event-interval  . ,mean-event-interval)
       (stdev-event-interval . ,stdev-event-interval)
       (cv-event-interval    . ,cv-event-interval)
       )
    
     )
  )

(foreign-declare 
#<<EOF
#include <assert.h>
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

EOF
)



;; Computes the Levenshtein edit distance between two integer vectors.
(define levenshtein-distance
    (foreign-lambda* unsigned-int ((unsigned-int m) (u32vector s1) (unsigned-int n) (u32vector s2))
#<<EOF
    // Algorithm from 
    // https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C
    unsigned int result, s1len, s2len, x, y, lastdiag, olddiag;
    s1len = m;
    s2len = n;
    unsigned int column[s1len+1];
    for (y = 1; y <= s1len; y++)
        column[y] = y;
    for (x = 1; x <= s2len; x++) {
        column[0] = x;
        for (y = 1, lastdiag = x-1; y <= s1len; y++) {
            olddiag = column[y];
            column[y] = MIN3(column[y] + 1, column[y-1] + 1, lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    result = (column[s1len]);

    C_return(result);
EOF
))



)
