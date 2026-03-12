# LaTeX Template

Simple LaTeX template for academic reports and assignments.

## Structure

```
LatexTemplate
│
├── Build/                # Output files generated during compilation (PDF, aux, log, etc.)
│
├── Inputs/
│   ├── Figures/          # Figures used in the document
│   ├── Pictures/         # Images or illustrations
│   └── Tables/           # Tables that can be included with \input
│
├── Sections/             # Document sections
│   └── Titlepage.tex     # Title page of the document
│
├── Main.tex              # Main entry point of the document
├── Preamble.tex          # Packages and global formatting settings
├── References.bib        # Bibliography database
└── .gitignore            # Prevents temporary LaTeX files from being tracked by Git
```

## Compile

Compile the document from the root folder:

```
latexmk -pdf Main.tex
```

or

```
pdflatex Main.tex
```

## .gitignore

The `.gitignore` file excludes temporary LaTeX compilation files such as `.aux`, `.log`, `.toc`, and `.synctex.gz`.
This keeps the repository clean by only tracking source files like `.tex`, `.bib`, and images.
