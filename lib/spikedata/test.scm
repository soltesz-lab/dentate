
(use srfi-4 spikedata)

(define x (list->u32vector (map char->integer (string->list "Saturday"))))
(define y (list->u32vector (map char->integer (string->list "Sunday"))))

(print (levenshtein-distance (u32vector-length x) x (u32vector-length y) y))

(define x (u32vector 1 2 3 4))
(define y (u32vector 4 3 2 4 1))

(print (levenshtein-distance (u32vector-length x) x (u32vector-length y) y))

