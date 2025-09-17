#  Automatic Short Answer Grading

Automatic short answer grading (ASAG) is the task of computationaly grading short answer to objective questions. It is different than AEG (automatic
essay grading) which is concerned with longer but less objective answers.

some methods used to achieve the objective include Prompt Engineering \[1]\[2], Word embedding similariy \[3] and sentence embedding similarity or finetuning these pretrained models\[4]. Out of these, the last two are present in this repositoy. 

# datasets
The data set used is edited version of [Mohler data set](http://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v1.0.tar.gz) provided by [rada mihalcea](http://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v1.0.tar.gz). 
The reason for the choice of dataset is two folds. First unlike other datasets the score are numerical, this makes it possible to train a regression model rather than a classifier. Sencondly the question are at university level and at the relevant subject of computer scinece.

the scores are between 0.0 (completely wrong) and 5.0 (completely correct).
 
![score distribution](./images/score-distribution.png 'score distribution')

The scores are negatively skewed.

It was accidently discovered that for most models if we scale the result into \[1,5] scale, the performace will be better (this probably can be explained by skewness of data).

However scaling can be considered a linear regression, it will be not be done so the result can be compared with papers' written in the subject.

## train, test, eval

the following will be the train, eval test split for finetuning:

train_all_row_k_is_11: 11 question and answers per question(all row attributes from main dataset)

eval_all_row_k_is_3: 3 row per question for evaluation

test_all_row_k_is_11: All the remaining rows per question, which in most cases *means* 16 = 30 - 11 - 3 (there are usually 30 answers for each question).


{train | eval | test}_triplets_row_k_is_14:  triplets of form (true answer, student answer , score)


# Evaluation methodology
The measure used are Root Mean Squared Error (RMSE) and and Pearson correlation coefficient ($\rho$). The former shows the amount of error we've had and the latter, the trainiblity of the predictions. The bigger the value of the $\rho$, the better the model that can be trained further on the result. Isotonic regreesion is frequenly used for this task, but here only linear and poly nominal regression is used.

# Models

## sentence embedding 

using cosine similarity between teacher's answer and student's. 
mxbai-embed-large and nomic-embed-text which are pretrained model available in ollama library were used.

the code is in ./code/embedding-cosine-similarity.ipynb

the prediction of each is available in /data/predictions


![distribution](./images/distribution_of_score_nomic_xbai.png 'distribution of score given by teachers adn two ollama models')

the mxbai-embed-large does perform better.

## finetuning sentence embedding

finetuning were done using [tutorial](https://huggingface.co/blog/train-sentence-transformers#trainer) with [SentenceTransformers library](https://sbert.net/index.html).

The results were impressive especially considering base model performance was worse than models available by ollama.

the selected loss function was CoSENTLoss (Cosine Sentence) which uses (sentence_A, sentence_B) pairs with a float for similarity score between 0 and 1.

this is ideal since we already have the given score of two sentences(student and teacher answer).

three polynomial regression were train on the predicted scores using 1 to 4 degree (${score}$ , ${score}^2$, ${score}^3$, ${score}^4$ features).

## results

                |           Model             |  RMSE  |    ρ    |
                |-----------------------------|--------|---------|
                | cosine similarity with xbai | 1.02   |  0.52   |
                | all-MiniLM-L6-v2 finetuned  | 0.97   |  0.72   |
                | all-MiniLM-L6-v2 finetuned  | 0.97   |  0.72   |  
                | degree 1 poly regression    | 0.80   |  0.73   |
                | degree 2 poly regression    | 0.78   |  0.76   |
                | degree 2 poly regression    | 0.80   |  0.76   |
                | degree 4 poly regression    | 0.75   |  0.76   |


## future work

- finetuning using stronger base model.

- perform punctuation the make student sentences more alike.

- handle special cases like empty answers.




## References

\[1] [Automatic Short Answer Grading in the LLM Era: Does GPT-4
with Prompt Engineering beat Traditional Models?](https://dl.acm.org/doi/pdf/10.1145/3706468.3706481)

\[2] [Automatic grading of short answers using Large Language
Models in software engineering coursesModels in software engineering courses](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=10267&context=sis_research)

\[3] [Comparative Evaluation of Pretrained Transfer Learning Models on
Automatic Short Answer Grading](https://arxiv.org/pdf/2009.01303)

\[4] [Finetuning Transformer Models to Build ASAG System](https://www.researchgate.net/publication/354950173_Finetuning_Transformer_Models_to_Build_ASAG_System)

