# Benchmark Failure Analysis

- DAG run: `b499eb36-51cd-45ae-be4f-227dcd3fa147` / `mistral-30-evidence-faithfulness-v2-20260617`
- Direct run: `79495ae9-e03d-473b-8580-b4c45c0e76a0` / `mistral-30-evidence-faithfulness-v2-20260617`
- Compared rows: `30`
- DAG cosine: `0.8627`
- Direct cosine: `0.9337`
- DAG wins / direct wins / ties: `1` / `5` / `24`

## Error Buckets

- `both_systems_wrong`: 3
- `other`: 24
- `wrong_factual`: 3

## Worst DAG Deltas

| ID | Category | Delta | Gold | DAG | Direct | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 5ab8736455429916710eb058 | `wrong_factual` | -0.860 | yes | True | yes | Are Yut and Tsuro both board games? |
| 5ac4e13f554299076e296e2d | `wrong_factual` | -0.640 | Theodor W. Adorno | Frank Wedekind | Theodor W. Adorno | which German philosopher wrote "The opera "Lulu" |
| 5a855ca15542992a431d1b12 | `wrong_factual` | -0.473 | Liu Ye, Yu Shaoqun and Leon Lai | Yu Shaoqun | Liu Ye, Yu Shaoqun, Leon Lai | The Chinese actress also known as Crystal Liu stars in Night Peacock with which three other actresses? |
| 5ae0945d55429906c02daadd | `other` | -0.391 | Lucy Maud Montgomery | L. M. Montgomery | Lucy Maud Montgomery | Wich children's novelist whow was first published in 1939 gained an internaional following writting about an orphaned girl named Anne Shirley? |
| 5a8e1027554299653c1aa15f | `other` | -0.038 | 2009 Big 12 Conference | year: 2009, conference: Big 12 Conference | 2009, Big 12 Conference | Which year and which conference was the 14th season for this conference as part of the NCAA Division that the Colorado Buffaloes played in with a record of 2-6 in conference play? |
| 5a8a367a55429930ff3c0d03 | `other` | 0.000 | Atlanta, Georgia, United States | Atlanta, Georgia, United States | Atlanta, Georgia, United States | Where is the company that came out with VisionPLUS headquartered? |
| 5a8dd5f955429917b4a5bcc7 | `both_systems_wrong` | 0.000 | paracyclist | paracycling | paracycling | Other than racing, what sport does the 1998 champion of the Toyota GRand Prix practice? |
| 5ac24cea55429951e9e6853a | `other` | 0.000 | Kate Millett | Kate Millett | Kate Millett | Which Oxford University graduate did Midge Mackenzie interview in "Women Talking"? |
| 5a84cb255542991dd0999dfa | `other` | 0.000 | Robin Gibb | Robin Gibb | Robin Gibb | Which singer of the Bee Gees sung a middle part of a song in which the majority was sung by Barry Gibb? |
| 5ae80919554299540e5a56f6 | `other` | 0.000 | The Changing Scottish Landscape | The Changing Scottish Landscape | The Changing Scottish Landscape | Isabella Kelly was born at a ruined castle characterized as one of the most isolated fortifications in Britain by who? |
