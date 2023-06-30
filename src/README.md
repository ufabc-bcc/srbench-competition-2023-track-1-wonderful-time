# Reproduce Guide

I packed my packages using conda-pack in a linux system. I do not have a Window system to test on. So it is ideal to reproduce these steps in a linux machine or use the docker.

## Docker


### Build
```
docker build -t symmfea .
```

### Run
**NOTE:** The docker only serves as an environment. To run the script, you need to exec inside the container.

```
docker run -it -v $project:/workspace symmfea /bin/bash
#Now we are inside the container
source /tmp/symmfea/bin/activate
cd /workspace
#reproduce model for dataset1
python run1.py
#reproduce model for dataset2
python run2.py
#reproduce model for dataset3
python run3.py
```

## Conda
```
mkdir symmfea
tar -xzf /tmp/symmfea.tar.gz -C /tmp/symmfea
```

