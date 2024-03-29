# Reproduce Guide

I packed my packages using conda-pack in a linux system. I do not have a Window system to test on. So it is ideal to reproduce these steps in a linux machine or use the docker.

# Requirements
You need to install the compressed [conda environment file](https://drive.google.com/file/d/1l-yAM95Arzm5dgzUxpm6n_xyGbDW9ibT/view?usp=sharing) and put in the "src" folder. Because the environment file is quite large (about 800 MB), I put it in my Google Drive. If you encounter any difficulty because of it, feel free to contact me.

## Docker


### Build
```
docker build -t symmfea .
```

### Run
**NOTE:** The docker only serves as an environment. To run the script, you need to exec inside the container.

```
#$project is the root dir of the git project, which contains run1.py, run2.py, run3.py and the datasets
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
**NOTE:** you have to have conda. The packages have been test on linux but not guaranteed to run on other OSs.  
```
mkdir symmfea
tar -xzf /tmp/symmfea.tar.gz -C /tmp/symmfea
source /tmp/symmfea/bin/activate
conda-unpack

cd $project
#reproduce model for dataset1
python run1.py
#reproduce model for dataset2
python run2.py
#reproduce model for dataset3
python run3.py
```

## Script Explaination
runX.py file contains the script which I used to find the model for the corresponding Xth dataset. After running the python script, the sympy expression will be placed in the solution file. The parameters are also included in the file and they are self-explained. 

## Contact
For further information, feel free to contact me via email: haiminhnguyen2001@gmail.com.

