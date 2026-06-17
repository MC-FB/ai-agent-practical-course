# Benchmark Failure Analysis

- DAG run: `70692dae-4eeb-4dea-a7e0-0b0f18778b55` / `mistral-30-evidence-faithfulness-v3-20260617`
- Direct run: `608fe126-4592-4935-a131-624c670def26` / `mistral-30-evidence-faithfulness-v3-20260617`
- Compared rows: `30`
- DAG cosine: `0.9183`
- Direct cosine: `0.9337`
- DAG wins / direct wins / ties: `3` / `1` / `26`

## Error Buckets

- `both_systems_wrong`: 3
- `other`: 26
- `wrong_factual`: 1

## Worst DAG Deltas

| ID | Category | Delta | Gold | DAG | Direct | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 5ae08dad55429945ae9593b9 | `wrong_factual` | -2.453 | "PM Magazine" |  | PM Magazine | Willie Geist frequently serves as fill-in anchor on "Today" for a tv journalist that was the host of what show from 1980-86? |
| 5a8a367a55429930ff3c0d03 | `other` | 0.000 | Atlanta, Georgia, United States | Atlanta, Georgia, United States | Atlanta, Georgia, United States | Where is the company that came out with VisionPLUS headquartered? |
| 5ae0945d55429906c02daadd | `other` | 0.000 | Lucy Maud Montgomery | Lucy Maud Montgomery | Lucy Maud Montgomery | Wich children's novelist whow was first published in 1939 gained an internaional following writting about an orphaned girl named Anne Shirley? |
| 5a8dd5f955429917b4a5bcc7 | `both_systems_wrong` | 0.000 | paracyclist | paracycling | paracycling | Other than racing, what sport does the 1998 champion of the Toyota GRand Prix practice? |
| 5ac24cea55429951e9e6853a | `other` | 0.000 | Kate Millett | Kate Millett | Kate Millett | Which Oxford University graduate did Midge Mackenzie interview in "Women Talking"? |
| 5a84cb255542991dd0999dfa | `other` | 0.000 | Robin Gibb | Robin Gibb | Robin Gibb | Which singer of the Bee Gees sung a middle part of a song in which the majority was sung by Barry Gibb? |
| 5ae80919554299540e5a56f6 | `other` | 0.000 | The Changing Scottish Landscape | The Changing Scottish Landscape | The Changing Scottish Landscape | Isabella Kelly was born at a ruined castle characterized as one of the most isolated fortifications in Britain by who? |
| 5ab2d9f1554299545a2cfac9 | `other` | 0.000 | Franz Ferdinand | Franz Ferdinand | Franz Ferdinand | What band in the "Art Wave" scene is based in Glasgow? |
| 5adf38ae5542993a75d26433 | `other` | 0.000 | film director | film director | film director | Both Alexander Hall and Pierre Morel are involved in which profession? |
| 5addbf645542997dc7907014 | `other` | 0.000 | Vernon Kay | Vernon Kay | Vernon Kay | Who hosted the ITV show Celebrities Under Pressure and All Star Family Fortunes? |
