
(require-extension lalr-driver)

;; parser

(include "calc.yy.scm")

;;;
;;;;   The lexer
;;;

(define (port-line port) 
  (let-values (((line _) (port-position port)))
    line))
  
(define (port-column port)
  (let-values (((_ column) (port-position port)))
    column))

(define (make-lexer errorp in)
  (lambda ()
    (letrec ((skip-spaces
              (lambda ()
                (let loop ((c (peek-char in)))
                  (if (and (not (eof-object? c))
                           (or (char=? c #\space) (char=? c #\tab)))
                      (begin
                        (read-char in)
                        (loop (peek-char in)))))))
             (skip-line
              (lambda ()
                (let loop ((c (peek-char in)))
                  (if (and (not (eof-object? c)) (not (char=? c #\newline)) (not (char=? c #\return)))
                      (begin
                        (read-char in)
                        (loop (peek-char in)))
                      ))
                ))
             (read-number
              (lambda (l)
                (let ((c (peek-char in)))
                  (if (or (char-numeric? c) (char=? #\. c) (char=? #\- c) (char=? #\e c))
                      (read-number (cons (read-char in) l))
                      (string->number (apply string (reverse l))) ))
                ))
             (read-id
              (lambda (l)
                (let ((c (peek-char in)))
                  (if (or (char-alphabetic? c) (char=? #\_ c))
                      (read-id (cons (read-char in) l))
                      (string->symbol (apply string (reverse l))) ))
                ))
             (read-string
              (lambda (l)
	       (let ([c (peek-char in)])
		 (cond [(eq? 'eof c)   (errorp "unexpected end of string constant")]
		       [(char=? c #\\) (let ((n (read-char in)))
					 (read-string (cons n l)))]
		       [(char=? c #\") (begin (read-char in) (apply string (reverse l))) ]
		       [else (read-string (cons (read-char in) l))] ))
               ))
             )

      ;; -- skip spaces
      (skip-spaces)
      ;; -- read the next token
      (let loop ()
        (let* ((location (make-source-location "*stdin*" (port-line in) (port-column in) -1 -1))
               (c (read-char in)))
          (cond ((eof-object? c)      '*eoi*)
                ((char=? c #\newline) (make-lexical-token 'NEWLINE location #f))
                ((char=? c #\+)       (make-lexical-token '+       location #f))
                ((char=? c #\-)       (make-lexical-token '-       location #f))
                ((char=? c #\*)       (make-lexical-token '*       location #f))
                ((char=? c #\/)       (let ((n (peek-char in)))
                                        (if (char=? n #\/)
                                            (begin (skip-line) (loop))
                                            (make-lexical-token '/ location #f))))
                ((char=? c #\=)       (make-lexical-token '=       location #f))
                ((char=? c #\,)       (make-lexical-token 'COMMA   location #f))
                ((char=? c #\()       (make-lexical-token 'LPAREN  location #f))
                ((char=? c #\))       (make-lexical-token 'RPAREN  location #f))
                ((char=? c #\")       (make-lexical-token 'STRING  location (read-string (list c))))
                ((char-numeric? c)    (make-lexical-token 'NUM     location (read-number (list c))))
                ((char-alphabetic? c) (make-lexical-token 'ID      location (read-id (list c))))
                (else
                 (errorp "PARSE ERROR : illegal character: " c)
                 (skip-spaces)
                 (loop))))))))



;;;
;;;;   Environment management
;;;


(define *env* (make-parameter (list (cons '$$ 0))))


(define (init-bindings)
  (*env* (list (cons '$$ 0)))
  (add-binding 'PI 3.14159265358979)
  (add-binding 'int round)
  (add-binding 'cos cos)
  (add-binding 'sin sin)
  (add-binding 'tan tan)
  (add-binding 'expt expt)
  (add-binding 'sqrt sqrt)
  (add-binding 'loadPoints load-points-from-file)
  )


(define (add-binding var val)
  (*env* (cons (cons var val) (*env*)))
  val)


(define (get-binding var)
  (let ((p (assq var (*env*))))
    (if p
        (cdr p)
        0)))


(define (invoke-func proc-name args)
  (let ((proc (get-binding proc-name)))
    (if (procedure? proc)
        (apply proc args)
        (begin
          (display "ERROR: invalid procedure:")
          (display proc-name)
          (newline)
          0))))


;; (init-bindings)

(define (errorp message . args)
  (display message)
  (if (and (pair? args) 
           (lexical-token? (car args)))
      (let ((token (car args)))
        (display (or (lexical-token-value token)
                     (lexical-token-category token)))
        (let ((source (lexical-token-source token)))
          (if (source-location? source)
              (let ((line (source-location-line source))   
                    (column (source-location-column source)))
                (if (and (number? line) (number? column))
                    (begin
                      (display " (at line ")
                      (display line)
                      (display ", column ")
                      (display (+ 1 column))
                      (display ")")))))))
      (for-each display args))
  (newline))

(define (calc-lexer in) (make-lexer errorp in))

(define (calc-eval lexer) (calc-parser lexer errorp))

(define (calc-eval-string s) 
  (calc-parser (calc-lexer (open-input-string (string-append s "\n"))) errorp))

