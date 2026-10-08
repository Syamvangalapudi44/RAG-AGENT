| Chunking | Pipeline | Accuracy | P@3 | Chunks | Mean size (chars) | Abstention rate |
|---|---|---|---|---|---|---|
| fixed | baseline | 1/8 = 12% | 0.29 | 16 | 169 | 0% |
| fixed | resolved | 5/8 = 62% | 0.33 | 16 | 169 | 12% |
| structure | baseline | 3/8 = 38% | 0.43 | 22 | 135 | 0% |
| structure | resolved | 7/8 = 88% | 0.38 | 22 | 135 | 12% |

### Per-query outcome (✓ correct / error type)

| Q | fixed/baseline | fixed/resolved | structure/baseline | structure/resolved |
|---|---|---|---|---|
| 1 | conflict | ✓ | conflict | ✓ |
| 2 | fragment | fragment | ✓ | ✓ |
| 3 | negation | negation | negation | negation |
| 4 | fragment | fragment | ✓ | ✓ |
| 5 | conflict | ✓ | conflict | ✓ |
| 6 | conflict | ✓ | conflict | ✓ |
| 7 | ✓ | ✓ | ✓ | ✓ |
| 8 | fabricated | ✓ | fabricated | ✓ |
