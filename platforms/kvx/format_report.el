;;; format_report.el ---


;;; Commentary:


;;; Code:

;; Const
(setq num-regexp "[0-9.]+")
(setq name-regexp "[A-Za-z0-9_.]+")
(setq name-with-point-regexp "[A-Za-z0-9_.]+")
(setq sep "[ \t]+")

(setq test-name-regexp (concat "RUN.*" "\\(" "OCL" name-with-point-regexp "/" num-regexp "\\)") )
(setq kernel-name-regexp (concat "clEnqueueNDRangeKernel('" "\\("  name-regexp "\\)" "'" ))
(setq kernel-timing-regexp (concat "TIMING |" sep ">>>" sep "\\(" num-regexp "\\)" sep "\\(" ".s" "\\)" ))


(defun my-format-perf-report(&optional file-to-parse)
  (interactive)
  (goto-char 1)
  (if (called-interactively-p 'any)
      (setq output (get-buffer-create "*perf-report*"))
    (setq output nil)
    )
  (when file-to-parse
    (find-file file-to-parse)
    (princ file-to-parse)
    )
  (while
      (search-forward-regexp (concat kernel-name-regexp  "\\|" test-name-regexp ) nil t)
    (goto-char (match-beginning 0))
    (let ((data (match-data)))
      (cond
       ((looking-at test-name-regexp)
        ;; parse test properties
        (setq test-name (match-string 1))
        (search-forward-regexp "GetParam() = (\\([a-zA-Z0-9. ,]+\\)")
        (setq test-param (match-string 1))
        (search-forward-regexp "mean=\\([0-9.]+\\)")
        (setq test-mean (match-string 1))
        (princ (format "\n%s\t(%s)\t%s\tms" test-name test-param test-mean) output)
        )
       ((looking-at kernel-name-regexp)
        ;; parse kernel properties
        (setq kernel-name (match-string 1))
        (search-forward-regexp kernel-timing-regexp nil t)
        (setq kernel-timing (match-string 1))
        (setq kernel-timing-unity (match-string 2))
        (replace-match "xxx" )
        (princ (format "\t%s:\t%s\t%s" kernel-name kernel-timing kernel-timing-unity) output)
        )
       )
      (set-match-data data)
      (goto-char (match-end 0))
      )
    )
  (switch-to-buffer output)
  )


(provide 'format_report)

;;; format_report.el ends here
