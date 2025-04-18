# Machine Learning and Pattern Recognition
Class of Machine Learning and Pattern Recognition of the Polytechnic University of Turin

# Repository Structure
This repository contains the code for the Machine Learning and Pattern Recognition course at the Polytechnic University of Turin. The code is organized into several folders, each containing different parts of the classwork. The main folders are:
- `labs`: This folder may contain the code for the labs of the course. Each lab is organized into its own folder, with a python code file.
- `work`: This folder contains jupyter notebooks and python code files for the labs and the projects.
- `work/utils`: This folder contains utility functions and classes used in the labs and projects.

# How to use this repository
You can clone this repository and run the code in the docker environment provided. To do this, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/ezmi234/machine-learing-and-pattern-recognition.git
```

2. Change into the directory:
```bash
cd machine-learing-and-pattern-recognition
```

3. Run the docker container:
```bash
docker-compose up -d
```
the first time it will build the image installing the `requirements.txt` file, if you want to force the build you can run:
```bash
docker-compose up -d --build
```

Then open a web browser and go to [http://localhost:8888](http://localhost:8888) to access the jupyter notebook.