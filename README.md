# WhatsAppSpamRec
Filtering out the spam from pre-quiz/midsem/endsem chats present in various Group Discussions on Whatsapp

This program uses a 6 Layer LSTM and pre-trained 200-dimensional GloVe Word Embeddings, with PyTorch. 

It has been trained from using about 2000 messages from actual pre-exam texts in Subject-Discussion Whatsapp Groups at IIT-B.

Considering the small Training set available, I decided to not split it into Validation, Test set at the risk of Overfitting.

It had ~90% accuracy.

While it had been trained on only 2000 whatsapp texts, Its vocabulary includes a much larger set i.e. [The Intersection of much of 
Wikipedia in 2014 (used to create the 6M Words, 200D GloVe pre-trained embeddings!) and Gerard Ekembe Ngondi and Anne Kerr's Dictionary for Computer Science].


I initially planned it using TensorFlow(code for which is commented out in MakingTheModel->model.py), but it didn't operate well on my PC, due to which I moved to PyTorch, adapting 
the from [a link](https://www.kaggle.com/swarnabha/pytorch-text-classification-torchtext-lstm).

In order to use the model (on Linux!), you need to have the following Libraries for Python, an incomplete list being: (and obviously you need Python itself!)

*numpy
*pandas
*torch
*torchtext 
*spacy
*sklearn
*matplotlib
*spacy

(This is an incomplete list of libraries, which I could've recalled, there are many more as well, which you could find out after running it)

In order to use it: 

1. Go to the Whatsapp Chat you want to summarize and look out for the 'Export Chat' option.
2. It would create a '.txt' file which you could save on Google Drive or wherever you want, and get it into your computer, in the folder where you downloaded this repo with 'git clone'
3. Remove the notifications like: 'xyz joined the group', 'xyz was added to the group', 'xyz has left the group' etc. This could be done easily with automated searching in your favourite text editor
4. Make sure, you don't have spaces before the start for the first text etc.
5. Run 'python main.py' while you're in the concerned directory, and if you have all the required packages, it would do the job.


Email me at shubh5796@gmail.com , if you find a bug, or if you wanna collaborate as there's still lots of room for improvement!

