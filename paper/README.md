# Paper

This directory contains the paper-style write-up for MisinfoEQA.

- `misinfoeqa_paper.md`: GitHub-readable version.
- `misinfoeqa_paper.tex`: LaTeX version.
- `references.bib`: BibTeX references.

To compile the LaTeX version:

```bat
pdflatex misinfoeqa_paper.tex
bibtex misinfoeqa_paper
pdflatex misinfoeqa_paper.tex
pdflatex misinfoeqa_paper.tex
```

The paper is written as a workshop or portfolio artifact. It makes a dataset-QA
claim, not a state-of-the-art misinformation detection claim.

