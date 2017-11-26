# Contributing to this project

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:


## Opening an issue

I would love to see suggestions. Please:
- Pay attention to your writing (grammar, style, syntax).
- Take a look at existing issues in case you are opening a new issue: your problem might already be discussed or solved.
- If you have a question rather than an issue, and if this question is unrelated to the project or will not be helpful to the community around this project, please contact me in private on [LinkedIn](https://www.linkedin.com/in/chevalierg/) or [Facebook](https://www.facebook.com/Guillaume.Chevalier.Che) instead.


## Submitting a pull request for improving the code

I would love to see improvements to this code base, as long as things stays clean. By clean, I mean that you: 
- Pay attention to your writing (grammar, style, syntax).
- Write clean code (e.g.: no magic constants, clean names, no useless comments, clean functions... and on).
- Seek to apply the [boyscout rule](http://programmer.97things.oreilly.com/wiki/index.php/The_Boy_Scout_Rule).
- Agree to the conditions of the [Apache License 2.0](https://github.com/guillaume-chevalier/seq2seq-signal-prediction/blob/master/LICENSE).
- Respect other people's license(s) in case you use code or contribution is derived or taken from elsewhere on internet.
- Work with the `seq2seq.ipynb` file to do your edits, to then export that at the end to the `seq2seq.py` and `README.md` files (see instructions below).
- If you are submitting answers to the questions, opening a branch might be a good idea: if so, be sure to give a good name to the branch. Since this project might evolve, don't hesitate to leave comments or some information about the versions of the libraries you are using (e.g.: in the commit message). 

That's all. Thanks! 


### Working with the `seq2seq.ipynb` file

The instructions on how to open that file are in the `README.md` and in the `seq2seq.ipynb` file itself, too. See: https://github.com/guillaume-chevalier/seq2seq-signal-prediction#how-to-use-this-ipynb-python-notebook-

If your changes to the project are about the structure of the neural network and that it would significantly change the results, please regenerate the charts in the introduction and below, and also ensure to edit the surrounding text. If this is too long or that you don't have the required hardware to do that, you can still open the pull request to let me deal with that later, or maybe someone else will do it. In that case, please specify that the charts need to be regenerated. Those charts are located in the `images/` folder, which is different than the `seq2seq_files/` folder. 

Be sure that the images in the notebook still render, and that you add all the necessary modified files before commiting. 

Also, note that to run all the exercises properly, you may need to spend a few bucks by renting a GPU-enabled VM/instance on [AWS](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html#gpu-instance-families), [GCP](https://cloud.google.com/gpu/), [Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu), or [OVH](https://www.ovh.com/us/public-cloud/instances/prices/).

In case you want to learn some deep learning concepts to understand the code before being able to contribute, here are the [slides of my talk at the WAQ 2017](https://www.slideshare.net/webaquebec/guillaume-chevalier-deep-learning-avec-tensor-flow), in which I explain the RNN and the seq2seq neural network architectures. Despite the slides are in French, the figures are mostly in English, and nearly 60% of French words are similar to English words. Also, you may want to take a look at my [Awesome Deep Learning Resources](https://github.com/guillaume-chevalier/Awesome-Deep-Learning-Resources), which is a place where I keep track of the resources I liked the most for deep learning.


### Exporting the `seq2seq.ipynb` file to `README.md` and `seq2seq.py`

The conversion of the `seq2seq.ipynb` to the `README.md` is done automatically by running the last cell of the notebook. 
However, ensure that the generated README file contains the good surrounding metatada. For example, files in the [seq2seq_files/](https://github.com/guillaume-chevalier/seq2seq-signal-prediction/tree/master/seq2seq_files) folder might have changed, this folder is used by the generated README to store some of the images. Therefore, I recommend to delete this folder **before** generating the `README.md` file anew. Don't forget to add those new generated files to your commit. 

The conversion of the `seq2seq.ipynb` to the `README.md` is done almost manually. Inside the ipython IDE when in the browser running the notebook, you can use the `File > Export` menu item (or a sub-menu item with a similar name, under `File`) to then export the the `.py` python file format. Once exported, please review the exported file and comment out (or delete) the lines with calls to `get_ipython()`, because those commands will not run outsite of the ipython notebook environment. Please ensure that this `.py` file still runs properly. You can then run a `git diff *.py` in the console to verify your changes before adding and commiting files. 
