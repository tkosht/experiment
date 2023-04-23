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


## train wiki data vectors

```
make wordvector
```


## train topicmodel

```
make topicmodel
```
