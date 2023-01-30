# mono_pitch_tracker
A data driven mono pitch tracker

## Setup
We recommend using Conda to setup the project.
1. Clone this repo and its submodules:
```
git clone --recursive https://github.com/duotien/mono_pitch_tracker.git
```

2. Create a conda environment:
```
cd mono_pitch_tracker
conda env create -f environment.yml
conda activate mpt_2022
```

3. Use conda-develop to install the package to the environment (this help importing custom modules)

Note: use must still be in the mono_pitch_tracker
```
conda develop .
```

4. Do the same thing for medleydb
```
cd medleydb
conda develop .
```

5. Follow this guide to install pytorch: https://pytorch.org/get-started/locally/