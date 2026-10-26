# Attribution and license

QuantileMix implements quantile g-computation described by
Keil et al. (2020), DOI: https://doi.org/10.1289/EHP5838.
Alexander Keil's R qgcomp 2.19.6 is the numerical reference:
https://github.com/alexpkeil1/qgcomp . Its license is GPL (>= 2).
The Python implementation is distributed under GPL-2.0-or-later, with
reference conventions and adaptations credited in the source.

This is a separate, unofficial Python implementation. The original authors
have not reviewed or endorsed it. DeepMix is prior partial Python coverage;
its code was not copied into this implementation.

Only the documented subsets are supported; the current methods, checks and
limits are listed in `docs/r-api-parity.md`.

The reference manual in `docs/manual` adapts applicable text and reference-entry
structure from Alexander Keil's GPL-licensed R qgcomp 2.19.6 documentation.
Original documentation credit is preserved. Python signatures, containers,
restrictions and examples were adapted or added on October 23, 2026. The manual
layout and wording were revised October 25, 2026. The upstream reference and
source identities are recorded in `docs/reference.md`. The editable manual is
`docs/manual/reference.md`; the PDF is built from that Markdown. The manual is distributed
under GPL-2.0-or-later. The full license terms and warranty disclaimer are in
`LICENSE`. R-only functions are listed as coverage targets,
not Python APIs.

The paper `docs/site/paper/quantilemix-paper.pdf` is authored by Nilson
Chapagain and distributed under Creative Commons Attribution 4.0
(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ .
The software and adapted reference manual retain the licenses stated above.
