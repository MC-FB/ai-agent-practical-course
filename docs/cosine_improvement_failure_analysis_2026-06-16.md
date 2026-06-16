# Benchmark Failure Analysis

- DAG run: `88752550-3dec-4238-ade3-7bebe9233d7d` / `mistral-30-paired-research-20260615`
- Direct run: `06d9b4a9-eeb4-4775-b336-150ce2598fa4` / `mistral-30-paired-research-20260615`
- Compared rows: `30`
- DAG cosine: `0.7392`
- Direct cosine: `0.8082`
- DAG wins / direct wins / ties: `5` / `8` / `17`

## Error Buckets

- `bad_or_misaligned_gold`: 1
- `both_systems_wrong`: 5
- `formatting`: 5
- `other`: 16
- `pipeline_decomposition`: 1
- `wrong_factual`: 2

## Worst DAG Deltas

| ID | Category | Delta | Gold | DAG | Direct | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 5abc1ca15542993a06baf88d | `pipeline_decomposition` | -0.809 | Vladimir Menshov | none | Vladimir Menshov | Which acter/director is known for depicting the Russian working class life and directed a film that won an Academy Award in 1981? |
| 5ae4581e55429970de88d92f | `wrong_factual` | -0.700 | Yanzhou | Jining | Yanzhou District | Which city is under Jining, Kaiyuan, Liaoning or Yanzhou District? |
| 5a8fdb955542990a98493572 | `wrong_factual` | -0.695 | WikiLeaks | TechCorp | WikiLeaks | One of the directors of Mediastan is well-known for founding what organisation in 2006? |
| 5ab31adc554299233954ff0a | `formatting` | -0.599 | 2.3 million pesos | 2300000 | 2.3 million pesos | How much Argentina currency was involved in Boudougate? |
| 5ab48a7c5542991751b4d7ab | `other` | -0.207 | Aircraft | airplane | Aircraft | What is this machine that is able to fly by gaining support from the air whose maneuvering is referenced to a standard rate turn, also known as a rate one turn? |
| 5ae616ce5542992663a4f259 | `formatting` | -0.203 | Charles and Thomas Guard | ['Charles Guard', 'Thomas Guard'] | Charles and Thomas Guard | What are the names of the co-directors of the psychological horror film starring Emily Browning and Elizabeth Banks? |
| 5ab946d7554299743d22eaaf | `both_systems_wrong` | -0.189 | the fourth-ranking Republican leader in the House | Representative | U.S. Representative | What is the rank of the incumbent that RJ Harris challenged in 2010? |
| 5ac43c7b5542997ea680ca34 | `formatting` | -0.183 | Tamworth | Danglemah is between Tamworth and Walcha. | Tamworth and Walcha | Danglemah, New South Wales is between which town and the town on the south-eastern edge of the Northern Tablelands, New South Wales Australia? |
| 5ab930c75542996be202043a | `other` | 0.000 | Arrested Development | Arrested Development | Arrested Development | What other series is the actress who plays Malory Archer well known for? |
| 5ae3fd2c5542995dadf2428f | `both_systems_wrong` | 0.000 | no | yes | Yes | Are Ian Brown and Dee Snider both actors? |

## Marked Rows

| ID | Category | Delta | Gold | DAG | Direct | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 5a7df5635542990b8f503b0a | `other` | 0.134 | 2014 | 2014 | 2015 | What year did the sequel to a story that is told through a combination of narrative and vernacular photographs from the personal archives of collectors by Ransom Riggs come out? |
| 5ae3fd2c5542995dadf2428f | `both_systems_wrong` | 0.000 | no | yes | Yes | Are Ian Brown and Dee Snider both actors? |
| 5ac43c7b5542997ea680ca34 | `formatting` | -0.183 | Tamworth | Danglemah is between Tamworth and Walcha. | Tamworth and Walcha | Danglemah, New South Wales is between which town and the town on the south-eastern edge of the Northern Tablelands, New South Wales Australia? |
| 5abc1ca15542993a06baf88d | `pipeline_decomposition` | -0.809 | Vladimir Menshov | none | Vladimir Menshov | Which acter/director is known for depicting the Russian working class life and directed a film that won an Academy Award in 1981? |
| 5ae616ce5542992663a4f259 | `formatting` | -0.203 | Charles and Thomas Guard | ['Charles Guard', 'Thomas Guard'] | Charles and Thomas Guard | What are the names of the co-directors of the psychological horror film starring Emily Browning and Elizabeth Banks? |
| 5a886211554299206df2b24a | `bad_or_misaligned_gold` | 0.002 | 1958 Walt Disney Western adventure film | No | Yes | Were the films Tonka and 101 Dalmatians released in the same decade? |
| 5a8fdb955542990a98493572 | `wrong_factual` | -0.695 | WikiLeaks | TechCorp | WikiLeaks | One of the directors of Mediastan is well-known for founding what organisation in 2006? |
