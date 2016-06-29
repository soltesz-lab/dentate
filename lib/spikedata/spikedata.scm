
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


static void init_distance_matrix(int **dm, int m, int n)
{
   unsigned int i, j;

   for (j = 0; j < n; j++)
   {
      dm[0][j] = j;
   }   
   for (i = 1; i < m; i++)
   {
      dm[i][0] = i;
      for (j = 1; j < n; j++)
      {
          dm[i][j] = 0;
      }   
   }   
}


// minimum of three numbers
static int minimum (int x, int y, int z)
{
    int u, res;

    u   = (x<y)?x:y;
    res = (u<z)?u:z;

    return res;
}

//  compare (x, y) returns 0 if x is equal to y, a negative integer if
//  x is less than y, and a positive integer if x is greater than y.
static int compare (int x, int y)
{ 
    int res;

    if (x == y)
    {
      res = 0;
    } else
    {
      res = (x<y)?-1:1;    
    }

    return res;
}


EOF
)



;; Computes the Levenshtein edit distance between two integer vectors.
(define levenshtein-distance
    (foreign-lambda* unsigned-int ((unsigned-int m) (u32vector xv) (unsigned-int n) (u32vector yv))
#<<EOF
     unsigned int result, i, j; int cost, x, y, z, **mat, *buf, *s, *t; 

     if (m == 0) 
     {
        result = n;
     } else if (n == 0) 
     {
        result = m;
     } else 
     {
        assert ((buf = malloc (sizeof(int)*(m+1)*(n+1))) != NULL);
        assert ((mat = malloc (sizeof(int*)*(m+1))) != NULL);
        for (i = 0; i <= m; i++)
        {
            mat[i] = buf+(i*n);
        }
        init_distance_matrix(mat,m+1,n+1);

        for (i = 1; i <= m; i++)
        {
           s = mat[i];
           t = mat[i-1];
           for (j = 1; j <= n; j++)
           {
              cost = abs (compare (xv[i-1], yv[j-1]));
              x = 1 + t[j];
              y = 1 + s[j-1];
              z = cost + t[j-1];
              s[j] = minimum(x,y,z);
           }
        }
        result = mat[m][n];
        free(mat);
        free(buf);
     }

     C_return(result);
EOF
))



)
