# Benchmark Failure Analysis

- DAG run: `dad5682b-a890-4ac9-befc-5dbe07ade7f7` / `mistral-30-evidence-faithfulness-20260617`
- Direct run: `1a96a2ed-6989-487c-a4ee-dd49eec14d7a` / `mistral-30-evidence-faithfulness-20260617`
- Compared rows: `30`
- DAG cosine: `0.8547`
- Direct cosine: `0.9259`
- DAG wins / direct wins / ties: `2` / `6` / `22`

## Error Buckets

- `both_systems_wrong`: 3
- `formatting`: 1
- `other`: 23
- `wrong_factual`: 3

## Worst DAG Deltas

| ID | Category | Delta | Gold | DAG | Direct | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 5ae0945d55429906c02daadd | `wrong_factual` | -0.915 | Lucy Maud Montgomery | no | Lucy Maud Montgomery | Wich children's novelist whow was first published in 1939 gained an internaional following writting about an orphaned girl named Anne Shirley? |
| 5ac4e13f554299076e296e2d | `wrong_factual` | -0.618 | Theodor W. Adorno | No German philosopher wrote the opera *Lulu*; it was composed by Alban Berg. | Theodor W. Adorno | which German philosopher wrote "The opera "Lulu" |
| 5a855ca15542992a431d1b12 | `wrong_factual` | -0.473 | Liu Ye, Yu Shaoqun and Leon Lai | Yu Shaoqun | Liu Ye, Yu Shaoqun, Leon Lai | The Chinese actress also known as Crystal Liu stars in Night Peacock with which three other actresses? |
| 5aba6d4b5542994dbf019906 | `both_systems_wrong` | -0.340 | English-language | Italian | English | Giuseppe Tornatore, an italian film director and screenwriter, wrote and directed his film "The Best Offer" in what language? |
| 5ab8919655429934fafe6e13 | `formatting` | -0.038 | IRA | Provisional Irish Republican Army (IRA) South Armagh Brigade | Provisional Irish Republican Army | Who used a Barrack buster to shoot down a British Army Lynx helicopter |
| 5a8e1027554299653c1aa15f | `other` | -0.038 | 2009 Big 12 Conference | year: 2009, conference: Big 12 Conference | 2009, Big 12 Conference | Which year and which conference was the 14th season for this conference as part of the NCAA Division that the Colorado Buffaloes played in with a record of 2-6 in conference play? |
| 5a8a367a55429930ff3c0d03 | `other` | 0.000 | Atlanta, Georgia, United States | Atlanta, Georgia, United States | Atlanta, Georgia, United States | Where is the company that came out with VisionPLUS headquartered? |
| 5a8dd5f955429917b4a5bcc7 | `both_systems_wrong` | 0.000 | paracyclist | paracycling | paracycling | Other than racing, what sport does the 1998 champion of the Toyota GRand Prix practice? |
| 5ac24cea55429951e9e6853a | `other` | 0.000 | Kate Millett | Kate Millett | Kate Millett | Which Oxford University graduate did Midge Mackenzie interview in "Women Talking"? |
| 5a84cb255542991dd0999dfa | `other` | 0.000 | Robin Gibb | Robin Gibb | Robin Gibb | Which singer of the Bee Gees sung a middle part of a song in which the majority was sung by Barry Gibb? |
