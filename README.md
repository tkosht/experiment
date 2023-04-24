# auto topic repository
this repository is for making auto topic model from wikipedia data


---

# build container

```
make
```

you may be waiting for around 10 mins.


---

# work in container's bash

```
make bash
cd backend/
```


---

# works in the container

in this part, what you can do in the container:

- make wiki datasets in sqlite3
- train word2vec vectors with thease wiki datasets
- train topic model with trained word2vec vectors


## make wiki data

you can be in container, make wiki 
from tensorflow datasets, to sqlite3 database which data record is consist of a paragraph

```
make wikidata
```

Used Mem: around 5.1GiB, so you possibly run on the PC has 6GiB available CPU Memory.
around 1h 10 mins
5
## train wiki data vectors

```
make wordvector
```

Used Mem: around 11.0GiB, so you possibly run on the PC has 12GiB available CPU Memory.
around 2h 30 mins


## train topicmodel

```
make topicmodel
```

Used Mem: around 6.3GiB, so you possibly run on the PC has 7GiB available CPU Memory.
around 0h 5 mins


