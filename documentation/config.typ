// config.typ
#let apply-styling(doc) = {
  set text(16pt, font: "New Computer Modern")
  set par(justify: true, leading: 0.58em)
  set page(header: [
    #set text(size: 12pt)
    _Valdemar Støvring Storgaard_
    #h(1fr)
    Erhvervsakademi København
  ], height: auto)
  
  set page(fill: rgb("#1c1c1c"))
  set text(fill: rgb("#e0e0e0"))
  
  show link: it => underline(text(it, blue))
  
 show raw.where(block: false): it => box(
  fill: rgb("#2d2d2d"),
  inset: (x: 2pt, y: 6pt),
  radius: 3pt,
  baseline: 0.5em,
  text(fill: rgb("#539804"), it)
)

  show raw.where(lang: "python"): it => block(
    fill: rgb("#282c34"),
    inset: 8pt,
    radius: 4pt,
    width: 100%,
    text(size: 11pt, it)
  )
  
  doc
}

#let infobox(content) = block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  text(fill: rgb("#1c1c1c"), content)
)

