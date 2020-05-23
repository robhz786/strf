mkdir -p out

asciidoctor -v quick_reference.adoc -o  out/quick_reference.html

asciidoctor -v strf_hpp.adoc -o - \
    | sed 's/20em/34em/g' \
    | sed 's/td.hdlist1{/td.hdlist1{min-width:9em;/g' \
    > out/strf_hpp.html

asciidoctor -v outbuf_hpp.adoc -o - \
    | sed 's/20em/25em/g' \
    | sed 's/td.hdlist1{/td.hdlist1{min-width:9em;/g' \
    > out/outbuf_hpp.html

asciidoctor -v to_cfile_hpp.adoc -o - \
    | sed 's/20em/25em/g' \
    | sed 's/td.hdlist1{/td.hdlist1{min-width:9em;/g' \
    > out/to_cfile_hpp.html

asciidoctor -v to_streambuf_hpp.adoc -o - \
    | sed 's/20em/25em/g' \
    | sed 's/td.hdlist1{/td.hdlist1{min-width:9em;/g' \
    > out/to_streambuf_hpp.html

asciidoctor -v to_string_hpp.adoc -o - \
    | sed 's/20em/25em/g' \
    | sed 's/td.hdlist1{/td.hdlist1{min-width:9em;/g' \
    > out/to_string_hpp.html

cp syntax.svg out/syntax.svg
