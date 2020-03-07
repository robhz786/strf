asciidoctor -v quick_reference.adoc -o  quick_reference.html
asciidoctor -v reference.adoc -o - \
    | sed 's/20em/29em/g' \
    | sed 's/td.hdlist1{/td.hdlist1{min-width:9em;/g' \
    > reference.html
