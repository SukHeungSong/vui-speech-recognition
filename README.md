[//]: # "Image References"

[image1]: ./images/pipeline.png "ASR Pipeline"
## Project Overview

In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!  

![ASR Pipeline][image1]

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate your models. Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

## Project Instructions

### Getting Started

6. Obtain the `libav` package.
	- __Linux__: `sudo apt-get install libav-tools`
	- __Mac__: `brew install libav`
	- __Windows__: Browse to the [Libav website](https://libav.org/download/)
		- Scroll down to "Windows Nightly and Release Builds" and click on the appropriate link for your system (32-bit or 64-bit).
		- Click `nightly-gpl`.
		- Download most recent archive file.
		- Extract the file.  Move the `usr` directory to your C: drive.
		- Go back to your terminal window from above.
	```
	rename C:\usr avconv
    set PATH=C:\avconv\bin;%PATH%
	```

7. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	- __Linux__ or __Mac__: 
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	./flac_to_wav.sh
	```
	- __Windows__: Download two files ([file 1](http://www.openslr.org/resources/12/dev-clean.tar.gz) and [file 2](http://www.openslr.org/resources/12/test-clean.tar.gz)) via browser and save in the `AIND-VUI-Capstone` directory.  Extract them with an application that is compatible with `tar` and `gz` such as [7-zip](http://www.7-zip.org/) or [WinZip](http://www.winzip.com/). Convert the files from your terminal window.
	```
	move flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	powershell ./flac_to_wav.sh
	```

8. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```


### Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


<a id='rubric'></a>
## Project Rubric

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Submission Files      | The submission includes all required files.		|

#### STEP 2: Model 0: RNN

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Trained Model 0         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_0.pickle` are undefined.  The trained weights for the model specified in `simple_rnn_model` are stored in `model_0.h5`.   	|

#### STEP 2: Model 1: RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Completed `rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `rnn_model` module containing the correct architecture.   	|
| Trained Model 1         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_1.pickle` are undefined.  The trained weights for the model specified in `rnn_model` are stored in `model_1.h5`.   	|

#### STEP 2: Model 2: CNN + RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Completed `cnn_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `cnn_rnn_model` module containing the correct architecture.   	|
| Trained Model 2         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_2.pickle` are undefined.  The trained weights for the model specified in `cnn_rnn_model` are stored in `model_2.h5`.   	|

#### STEP 2: Model 3: Deeper RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Completed `deep_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `deep_rnn_model` module containing the correct architecture.   	|
| Trained Model 3         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_3.pickle` are undefined.  The trained weights for the model specified in `deep_rnn_model` are stored in `model_3.h5`.   	|

#### STEP 2: Model 4: Bidirectional RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Completed `bidirectional_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `bidirectional_rnn_model` module containing the correct architecture.   	|
| Trained Model 4         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_4.pickle` are undefined.  The trained weights for the model specified in `bidirectional_rnn_model` are stored in `model_4.h5`.   	|

#### STEP 2: Compare the Models

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Question 1         		| The submission includes a detailed analysis of why different models might perform better than others.   	|

#### STEP 2: Final Model

| Criteria       		|     Meets Specifications	        			            |
|:---------------------:|:---------------------------------------------------------:|
| Completed `final_model` Module         		| The submission includes a `sample_models.py` file with a completed `final_model` module containing a final architecture that is not identical to any of the previous architectures.   	|
| Trained Final Model        		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_end.pickle` are undefined.  The trained weights for the model specified in `final_model` are stored in `model_end.h5`.   	|
| Question 2         		| The submission includes a detailed description of how the final model architecture was designed.   	|

## Further More

#### (1) Add a Language Model to the Decoder

The performance of the decoding step can be greatly enhanced by incorporating a language model.  Build your own language model from scratch, or leverage a repository or toolkit that you find online to improve your predictions.

#### (2) Train on Bigger Data

In the project, you used some of the smaller downloads from the LibriSpeech corpus.  Try training your model on some larger datasets - instead of using `dev-clean.tar.gz`, download one of the larger training sets on the [website](http://www.openslr.org/12/).

#### (3) Try out Different Audio Features

In this project, you had the choice to use _either_ spectrogram or MFCC features.  Take the time to test the performance of _both_ of these features.  For a special challenge, train a network that uses raw audio waveforms!

## Special Thanks

We have borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
