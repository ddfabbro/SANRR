# SANRR
A Python framework for Surrogate Assisted Nonrigid Registration (SANRR) using MIRTK.

## Requirements
- `numpy`
- `scikit-learn`
- `scikit-image`
- `PIL`
- `dlib`
- `pyDOE`
- `pyKriging`
- [Docker](https://docs.docker.com/install/)

## Fast getting started example: SANRR of [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html)

After installing the requirements, you can easily get started with SANRR of [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html) by following these 3 steps:

1. Clone repository
```
git clone --recursive https://github.com/ddfabbro/SANRR.git
```

2. Add it to `PYTHONPATH`
```
export PYTHONPATH=$PYTHONPATH:$(pwd)/SANRR
```

3. Pull MIRTK Docker image
```
docker pull biomedia/mirtk
```

4. Create a container named `mirtk` with shared volume
```
docker run -t -d --name mirtk --entrypoint /bin/bash -v $(pwd)/SANRR/mirtk_folder/:$(pwd)/SANRR/mirtk_folder/ biomedia/mirtk
```

5. Start SANRR
```
python SANRR/examples/register_fei_db.py
```
