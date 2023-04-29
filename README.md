# Seminar Neuroinformatik
This project was created as a part of a seminar work in my bachelor studies. It consists of two notebooks that train autoencoders on different criteria.
Feel free to try it out.


## Getting started

```bash
$ git clone https://github.com/Tobi2K/SeminarNI.git
$ cd SeminarNI
$ pip install -r requirements.txt
```

## Configuring the trials
In the [Autoencoder Notebook](https://github.com/Tobi2K/SeminarNI/blob/main/Autoencoder.ipynb) you can select which comparisons to run by setting the corresponding variables to True.

The possible comparisons are:
- difference in number of autoencoder runs aka. epochs
- the size of the dense layer
- the number of dense layers
- added noise vs no noise 

In the [Denoising Autoencoder Notebook](https://github.com/Tobi2K/SeminarNI/blob/main/Denoising%20Autoencoder.ipynb) you can run a set of denoising autoencoders.


## Show loss trend
The loss of each epoch is tracked and can be displayed with TensorBoard.
```bash
# start TensorBoard
$ tensorboard --logdir=/path/to/project/SeminarNI/logs
```

The comparisons save their logs in separat folders, making separation easy:
```bash
# start TensorBoard but only show number of runs comparison
$ tensorboard --logdir=/path/to/project/SeminarNI/logs/epochs

# start TensorBoard but only show size of dense layer comparison
$ tensorboard --logdir=/path/to/project/SeminarNI/logs/layer_size

# start TensorBoard but only show number of dense layer comparison
$ tensorboard --logdir=/path/to/project/SeminarNI/logs/layer_count

# start TensorBoard but only show noise vs no noise comparison
$ tensorboard --logdir=/path/to/project/SeminarNI/logs/compare_noise
```

### Preexisting data
There already are existing [images](https://github.com/Tobi2K/SeminarNI/tree/main/images), [log-texts](https://github.com/Tobi2K/SeminarNI/tree/main/output_texts) as well as [TensorBoard logs](https://github.com/Tobi2K/SeminarNI/tree/main/logs) contained in this repository.

When running the configurations, only TensorBoard logs are saved automatically. Images and log-texts have to be saved manually.