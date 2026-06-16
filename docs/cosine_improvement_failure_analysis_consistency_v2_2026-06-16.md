# Benchmark Failure Analysis

- DAG run: `8302b93a-13b5-4d9c-8a5f-773a652dcc2e` / `mistral-30-consistency-v2-20260616`
- Direct run: `97e456de-700e-4345-841e-a14fa01ff344` / `mistral-30-consistency-v2-20260616`
- Compared rows: `30`
- DAG cosine: `0.9037`
- Direct cosine: `0.8759`
- DAG wins / direct wins / ties: `6` / `1` / `23`

## Error Buckets

- `both_systems_wrong`: 3
- `formatting`: 3
- `other`: 24

## Worst DAG Deltas

| ID | Category | Delta | Gold | DAG | Direct | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 5ae72c7d5542991e8301cb91 | `other` | -0.365 | Mac OS X 10.3 | 10.6 | Mac OS X 10.6 Snow Leopard | A sparse image is used by the FileVault feature in Mac OS X in versions later than which? |
| 5ac3e22d5542997ea680c976 | `other` | 0.000 | Hilary Duff | Hilary Duff | Hilary Duff | Michelle Lewis has written songs for which "Lizzie McGuire" actress? |
| 5a734e6d5542994cef4bc528 | `other` | 0.000 | The Beaver and Erie Canal | Beaver and Erie Canal | Beaver and Erie Canal | Which canal is located further North, Beaver and Erie Canal or the Dismal Swamp Canal? |
| 5ab5ec2a5542992aa134a3e3 | `formatting` | 0.000 | band | rock band | rock band | What kind of group does Takahiro Moriuchi and Doug Pinnick have in common? |
| 5adfa73b55429942ec259ad8 | `other` | 0.000 | Rockbridge County | Rockbridge County | Rockbridge County | In which county is the university, at which Roger Groot was the Class of 1975 Alumni Professor of Law, located? |
| 5ab2eff05542991669774140 | `other` | 0.000 | Greater Rochester International Airport | Greater Rochester International Airport | Greater Rochester International Airport | Which of the following is home to the 642nd Aviation Support Battalion: Greater Rochester International Airport or Valley International Airport? |
| 5adce2ee5542992c1e3a2461 | `other` | 0.000 | Tampa | Tampa | Tampa | What city that Lettuce Lake Park is just outside of is the largest in Hillsborough County, Florida? |
| 5a865d11554299211dda2b0d | `other` | 0.000 | Fort Saint Anthony | Fort Saint Anthony | Fort Saint Anthony | What is the original name of the place where The 1st Minnesota Light Artillery Battery mustered? |
| 5a7cef6955429909bec768ad | `other` | 0.000 | Crested Butte, Colorado | Crested Butte | Crested Butte | Which municipality of Gunnison County, Colorado is now called "the last great Colorado ski town" and is located near the Lucky Jack mine? |
| 5ae688065542996d980e7beb | `other` | 0.000 | 122,067 | 122,067 | 122,067 | What was the population at the 2010 census of the city which, along with Clayton, is covered by the Mount Diablo Unified School District? |
