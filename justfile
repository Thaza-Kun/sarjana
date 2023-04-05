set shell := ["powershell.exe", "-c"]
# set dotenv-load

clean-ref file='./thesis/references.bib':
    bibtex-tidy --curly --numeric --tab --align=13 --sort=key --duplicates=key --no-escape --sort-fields --strip-comments --no-remove-dupe-fields {{file}}

present folder root='presentations':
    cd {{root}} | quarto render ./{{folder}}/index.qmd