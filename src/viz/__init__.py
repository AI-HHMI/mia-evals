"""Diagnostic figures: what a prediction looks like beside the truth it is scored against.

Separate from the metrics because these answer a different question. A metric compresses a whole
volume to a number that ranks; these show why the number came out the way it did. PQ 0.0031 and
voi_merge 6.74 cannot distinguish "merged everything into one object" from "shattered everything
into dust", and those need opposite fixes -- one picture separates them immediately.

Both modules postprocess through the same registry the scorer uses, so what is drawn is what was
scored rather than a second implementation that could quietly disagree with it.
"""
