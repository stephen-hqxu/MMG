# Music Generation by Matching Piano Roll

Year 4 MEng Group Project

This repository contains the code to generate a humanised music by matching two MIDI files in piano roll representation. The codebase is developed with *Python*. The project directory structure is pretty simple to navigate; specifically, you might be interested to take a look into the following folders:

- **Data** - Preprocess MIDI files and format the data into note and piano roll representation.
- **Model** - The main model of this project.

## Dependencies

We have only listed the major dependencies here, other commonly used packages such as *NumPy* and *Pandas* are also required. Please refer to *requirements.txt* to setup your local environment.

- We use [pretty_midi](https://github.com/craffel/pretty-midi) for MIDI file read and write.
- [PyTorch](https://github.com/pytorch/pytorch) is the framework for building our main model.

## Usage

After getting all dependencies and data resolved, you may train the model by running the following command from the project root directory.

```sh

python3 ./Run.py

```

Please note that this model is an experiment thus unoptimised and unusable. The memory consumption is very high (up to 1TB) that you need to run the training on a high-memory batch computing node. The inference code is unimplemented.