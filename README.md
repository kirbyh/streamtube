# Streamtube
Minimalist, lightweight streamtube package to compute streamtubes

# Installation
To install this Python package follow one of the following methods.
### Direct installation from Github
To install directly from Github into the current Python environment, run:
```bash
pip install git+https://github.com/kirbyh/streamtube.git
```

### Install from cloned repository
Alternatively, clone the repository and added it to your virtual environment. 
```bash
git clone https://github.com/kirbyh/streamtube.git
```
or ssh:
```bash
git clone git@github.com:kirbyh/streamtube.git
```
then, install locally using pip using `pip install .` or with poetry using `poetry install .`

```bash
cd streamtube
poetry install .
```

## Usage
The `Streamtube` class is used to construct a streamtube. Given velocity fields $u$, $v$, and $w$, the `Streamtube` class integrates streamline trajectories from the given seed points and then computing a mask from the trajectories by computing overlap between the streamtube cross-section polygon and the flow field grid. 
```python
import streamtube
stream = streamtube.Streamtube(x, y, z, u, v, w)

# to compute trajectories, 3D mask, and return a mask: 
mask = stream.compute_mask(return_mask=True)
```

By default, `return_mask=False` and the result is stored in `stream.mask`. If a mask is not needed and only trajectories are required (much faster), then only trajectories can be computed: 
```python
traj = stream.compute_streamtube(return_trajectories=True)
```

Again, by default, `return_trajectories=False` and the trajectories are stored in `stream.trajectories`. 