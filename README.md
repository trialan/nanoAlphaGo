# nanoAlphaGo
Minimalist implementation of AlphaGo Zero. Inspired by nanoGPT.

This is WIP.

## Go Vocabulary
1. <u> Liberty </u> In a given position, a liberty of a stone is an empty instersection adjacent to that stone, or adjacent to a stone connected to that stone.

## Lessons learned along the way
1. GPT-4 does make subtle mistakes, so what I did is that whilst GPT-4 wrote
   much of the code, I carefully wrote the test suite for good coverage. This
uncovered a couple mistakes.
