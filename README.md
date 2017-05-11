# DeepLie
Using LSTMs and Word Vectorisation to find difference in patterns of text in documents.

This model can be used to check if a speaker's preferences or views about a particular topic or agenda has changed over time or not.

LSTMs are used for training the model on huge text files of a particular speaker. The LSTM then generates its own interpretation of what the speaker meant about a particular topic. This can be achieved by giving the agenda topic as the seed to the LSTM. The generated text can be compared to the more recent speeches on the same topic. Using word vectorisation, it can be calculated how close the 2 speeches are. Eucledian distance between the centroids of the 2 speeches can also be calculated to get an idea of how far the 2 speeches are from each other, in terms of, views and opinions expressed.

## Example : 

### Origianl Text : 

Americans currently enrolled in the healthcare exchanges .
Secondly , we should help Americans purchase their own coverage , through the use of tax credits and expanded Health Savings Accounts  but it must be the plan they want , not the plan forced on them by the Government .
Thirdly , we should give our great State Governors the resources and flexibility they need with Medicaid to make sure no one is left out .Fourthly , we should implement legal reforms that protect patients and doctors from unnecessary costs that drive up the price of insurance  and work to bring down the artificially high price of drugs and bring them down immediately .
Finally , the time has come to give Americans the freedom to purchase health insurance across State lines  creating a truly competitive national marketplace that will bring cost way down and provide far better care .
Everything that is broken in our country can be fixed . Every problem can be solved . And every hurting family can find healing , and hope .

### Generated Text (using LSTMs) : 

americans currently enrolled in the healthcare exchanges .
secondly , we should help americans purchase coverage ,
through use tax credits and expand health save accounts  but it must be the plan they want ,
not the plan forced on them by the government .
thirdly , we should give our great state governors the resources its brame dorm the paford . in our country at riske america's fanital ,
this was about time oressed , we will be our success . we share one heart , one home , and our neeting fooded on the land of office.

## Word Vectorisation : 

Word vectors are calculated for the words present in the documents using pre trained word vector models. Gensim is used to load the pre trained weights. PCA is done on the vectors and brought down to 3 components for visualization purposes.

![](https://github.com/rohitsaha/DeepLie/blob/master/Data/Plotting.png)

<ul>
<li> Points which are marked in 'red' are word vectors that were derived from the original document.
<li> Points which are marked in 'blue' are word vectors that were derived from the generated document.
</ul>

