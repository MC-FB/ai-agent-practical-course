# Benchmark Failure Analysis

- DAG run: `389eda0b-15e5-40d4-875b-a117b529fcc7` / `mistral-30-cosine-loop-v2-20260616`
- Direct run: `e5d3817a-991f-42b7-9161-cbc599ecd596` / `mistral-30-cosine-loop-v2-20260616`
- Compared rows: `30`
- DAG cosine: `0.9201`
- Direct cosine: `0.9222`
- DAG wins / direct wins / ties: `4` / `1` / `25`

## Error Buckets

- `both_systems_wrong`: 4
- `formatting`: 1
- `other`: 24
- `wrong_factual`: 1

## Worst DAG Deltas

| ID | Category | Delta | Gold | DAG | Direct | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 5ab7c6995542993667794005 | `wrong_factual` | -0.814 | Bothtec | The Radio | Bothtec | Before composing music for the game which contains Chocobos and Moogles, what cover band was he involved in? |
| 5a859a0c5542992a431d1b69 | `other` | 0.000 | no | no | no | Do both Icehouse pieces and El Grande come from the same game? |
| 5a83a2a15542993344746063 | `other` | 0.000 | June 1925 | 1925 | 1925 | In regards to the high school that forced Manchester High School to change its name in 1922, when was its first graduating class? |
| 5ae00de055429906c02daab5 | `other` | 0.000 | Philip K. Dick | Philip K. Dick | Philip K. Dick | Morgan Paull played Dave Holden in a 1982 film loosely adapted from a novel by what author? |
| 5ae534bb5542990ba0bbb21d | `other` | 0.000 | Liberty | Liberty | Liberty | In between Liberty and MIT Technology Review which has a circulation of over 200,000? |
| 5add1aa65542990d50227de1 | `other` | 0.000 | Tom Shadyac | Tom Shadyac | Tom Shadyac | Who was the director of the 2007 American fantasy comedy film in which the actor best known for playing Michael Scott on the the American version of "The Office" starred? |
| 5a87954f5542996e4f308856 | `other` | 0.000 | Bactris | Bactris | Bactris | Which genus has more species, Bactris and Epigaea? |
| 5a8d9e0f554299441c6ba015 | `other` | 0.000 | Vaisakhi List | Vaisakhi List | Vaisakhi List | The actor that played Gutthi on Comedy nights with Kapil Show also starred in what 2016 Punjabi film directed by Smeep Kang? |
| 5ab7c3ed5542991d322237b2 | `other` | 0.000 | yes | yes | yes | Are Mike Bryan and Ray Ruffels both tennis players? |
| 5ab26a4b55429970612095fc | `formatting` | 0.000 | Hohenstaufen | House of Hohenstaufen | House of Hohenstaufen | Tile Kolup pretended to be a member of what House? |
