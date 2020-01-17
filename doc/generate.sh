asciidoctor quick_reference.adoc -o  quick_reference.html
asciidoctor reference.adoc -o - | sed 's/20em/28em/g' > reference.html
