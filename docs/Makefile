
StrfVersion = 0.16.0
StrfRev = v$(StrfVersion)
ifdef STRF_LOCAL_DOCS
	StrfRootLink = link:../..
else
	StrfRootLink = https://github.com/robhz786/strf/blob/$(StrfRev)
endif


all: \
    out/syntax.svg \
    out/install.html \
    out/tutorial.html \
    out/quick_reference.html \
    out/locale_hpp.html \
    out/howto_add_destination.html \
    out/howto_add_printable_types.html \
    out/howto_override_printable_types.html \
    out/cuda.html \
    out/versus_fmtlib.html \
    out/why_not_std_locale.html \
    out/strf_hpp.html \
    out/destination_hpp.html \
    out/to_cfile_hpp.html \
    out/to_streambuf_hpp.html \
    out/to_string_hpp.html \
    out/iterator_hpp.html

clean:
	rm -rf out/

out/ :
	mkdir -p out

out/asciidoctor-default.css : asciidoctor-default.css out/
	cp -f $< $@

out/my-style.css : my-style.css out/asciidoctor-default.css
	cp -f $< $@

out/syntax.svg : syntax.svg out/my-style.css
	cp -f $< $@

out/install.html : install.adoc out/my-style.css
	asciidoctor -v --attribute=strf-version=$(StrfVersion) --attribute=strf-src-root=$(StrfRootLink) $< -o $@

out/tutorial.html : tutorial.adoc out/my-style.css
	asciidoctor -v $< -o $@

out/quick_reference.html : quick_reference.adoc ref_format_functions.adoc out/my-style.css
	asciidoctor -v --attribute=strf-src-root=$(StrfRootLink) $< -o $@

out/locale_hpp.html : locale_hpp.adoc out/my-style.css
	asciidoctor -v $< -o $@

out/howto_add_destination.html : howto_add_destination.adoc out/my-style.css
	asciidoctor -v --attribute=strf-version=$(StrfVersion) --attribute=strf-src-root=$(StrfRootLink) $< -o $@

out/howto_add_printable_types.html : howto_add_printable_types.adoc out/my-style.css
	asciidoctor -v --attribute=strf-version=$(StrfVersion) --attribute=strf-src-root=$(StrfRootLink) $< -o $@

out/howto_override_printable_types.html : howto_override_printable_types.adoc out/my-style.css
	asciidoctor -v --attribute=strf-version=$(StrfVersion) --attribute=strf-src-root=$(StrfRootLink) $< -o $@

out/cuda.html : cuda.adoc out/my-style.css
	asciidoctor -v --attribute=strf-version=$(StrfVersion) --attribute=strf-src-root=$(StrfRootLink) $< -o $@

out/versus_fmtlib.html : versus_fmtlib.adoc out/my-style.css
	asciidoctor -v --attribute=strf-version=$(StrfVersion) --attribute=strf-src-root=$(StrfRootLink) $< -o $@

out/why_not_std_locale.html : why_not_std_locale.adoc out/my-style.css
	asciidoctor -v -a stylesheet=my-style.css $< -o $@

out/strf_hpp.html : strf_hpp.adoc \
                    ref_facets_pack.adoc \
                    ref_printing_syntax.adoc \
                    ref_format_functions.adoc \
                    ref_printable_types.adoc \
                    ref_printable_types_requirements.adoc \
                    ref_charset.adoc \
                    ref_width.adoc \
                    ref_numpunct.adoc \
                    ref_misc.adoc \
                    out/my-style.css /
	asciidoctor -v -a stylesheet=my-style.css $< -o $@

out/destination_hpp.html : destination_hpp.adoc out/my-style.css
	asciidoctor -v -a stylesheet=my-style.css $< -o  $@

out/to_cfile_hpp.html : to_cfile_hpp.adoc out/my-style.css
	asciidoctor -v -a stylesheet=my-style.css $< -o  $@

out/to_streambuf_hpp.html : to_streambuf_hpp.adoc out/my-style.css
	asciidoctor -v -a stylesheet=my-style.css $< -o  $@

out/to_string_hpp.html : to_string_hpp.adoc out/my-style.css
	asciidoctor -v -a stylesheet=my-style.css $< -o  $@

out/iterator_hpp.html : iterator_hpp.adoc out/my-style.css
	asciidoctor -v -a stylesheet=my-style.css $< -o  $@

