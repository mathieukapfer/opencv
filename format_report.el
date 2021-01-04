;;; format_report.el ---


;;; Commentary:
;;

;;; Code:

(defun my-format-perf-report()
  (interactive)
  (goto-char 1)
  (setq output (get-buffer-create "*perf-report*"))
  (while (search-forward-regexp "RUN.*\\(OCL[A-Za-z0-9_.]*/[0-9]+\\)" nil t)
    (setq test-name (match-string 1))
    (search-forward-regexp "mean=\\([0-9.]+\\)")
    (setq test-mean (match-string 1))
    (princ (format "%s: %s\n" test-name test-mean) output)
    )
  (switch-to-buffer output)
  )


(provide 'format_report)

;;; format_report.el ends here
