# If you have pdflatex installed, type "make" to create the
# documentation, "make clean" to delete it

documentation: adept_documentation.pdf adept_reference.pdf

adept_documentation.pdf: adept_documentation.tex
	pdflatex adept_documentation.tex
	pdflatex adept_documentation.tex
	pdflatex adept_documentation.tex

adept_reference.pdf: adept_reference.tex
	pdflatex adept_reference.tex

clean:
	rm adept_documentation.pdf adept_reference.pdf

.PHONY: documentation clean
