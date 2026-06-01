# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help Makefile apidoc

# Catch-all target: route all unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

apidoc:
	@$(SPHINXBUILD) -b html -q "$(SOURCEDIR)" "$(BUILDDIR)" >/dev/null 2>&1 || true
	@sphinx-apidoc -o "$(SOURCEDIR)/api_reference/_generated" -t "$(SOURCEDIR)/_templates/apidoc" ../kimodo ../kimodo/**/tests* ../kimodo/**/test* -f
