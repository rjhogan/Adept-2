dist_pkgdata_DATA = adept_documentation.pdf adept_reference.pdf
EXTRA_DIST = adept_documentation.tex adept_reference.pdf

documentation: adept_documentation adept_reference

adept_documentation:
	pdflatex adept_documentation.tex
	pdflatex adept_documentation.tex
	pdflatex adept_documentation.tex

adept_reference:
	pdflatex adept_reference.tex

.PHONY: documentation adept_documentation adept_reference
