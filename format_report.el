;;; format_report.el ---


;;; Commentary:


;;; Code:

;; Const
(setq num-regexp "[0-9]+")
(setq name-regexp "[A-Za-z0-9_.]+")
(setq name-with-point-regexp "[A-Za-z0-9_.]+")

(setq test-name-regexp (concat "RUN.*" "\\(" "OCL" name-with-point-regexp "/" num-regexp "\\)") )
(setq kernel-name-regexp (concat "clEnqueueNDRangeKernel('" "\\("  name-regexp "\\)" "'" ))


(defun my-format-perf-report()
  (interactive)
  (goto-char 1)
  (setq output (get-buffer-create "*perf-report*"))
  (while
      (search-forward-regexp (concat kernel-name-regexp  "\\|" test-name-regexp ) nil t)
    (goto-char (match-beginning 0))

    (let ((data (match-data)))
      (cond
       ((looking-at test-name-regexp)
        (setq test-name (match-string 1))
        (search-forward-regexp "mean=\\([0-9.]+\\)")
        (setq test-mean (match-string 1))
        (princ (format "\n%s: %s\n" test-name test-mean) output)
        )
       ((looking-at kernel-name-regexp)
        (setq kernel-name (match-string 1))
        (princ (format " %s " kernel-name) output)
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
