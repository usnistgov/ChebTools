
import subprocess, sys
subprocess.check_call('pandoc --filter pandoc-citeproc --bibliography paper.bib  paper.md -o paper.pdf',
                      shell=True,stdout=sys.stdout,stderr=sys.stderr)