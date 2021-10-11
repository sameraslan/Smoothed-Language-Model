marhaba

###Q1
Perplexity = 2^(-l), l = (total log probability) / (number of words). Thus,
Perplexity per word of sample1: 68.53 \
Perplexity per word of sample2: 95.24 \
Perplexity per word of sample3: 89.60

If we train on the larger switchboard corpus, all the log probabilities are more negative, thus causing the perplexities for each of the sample files to be greater. This could be due to overfitting or unrepresentative training data.

###Q3(a)
All gen test files were classified as gen (error rate of 0%). 93.33% of spam test files were classified as gen (error rate of 93.33%). Overall this results in a dev file error rate of 84/270 (31.11%). 

###Q3(b)
Created vocab list from en.1k and sp.1k.
All english test files were classified as english (error rate of 0%). 100%% of spanish test files were classified as english (error rate of 100%%). Overall this results in a dev file error rate of 119/339 (49.79%).

###Q3(c)
A prior probability of gen as large as .22 will classify all the gens as spam.

###Q3(d)
For gen test files, the lowest cross entropy we could achieve was 9.04616 bits per token with a lambda value of .008. For spam test files, the lowest cross entropy we could achieve was 9.09572 bits per token with a lambda value of .005.

###Q3(e)
Total tokens calculated using wc -w = 87482
With lambda = .008: 9.0981 bits per token
With lambda = .005: 9.0964 bits per token
With lambda = .006: 9.0937 bits per token
With lambda = .007: 9.0949 bits per token
With lambda = .0063: 9.09369 bits per token
With lambda = .0061: 9.09363 bits per token

Thus, it seems that the lambda* used for smoothing both models that minimizes cross entropy is roughly .0061, resulting in a cross entropy of 9.09363 bits per token

###Q3(f)
Photos in the google doc. 

###Q3(h)


###Q4(a)
The UNIFORM estimate 1/V would be slightly greater than the ideal estimate since V (the denominator) would be one less, resulting in slight overfitting. This is also the case for add-lambda since the denominator would be smaller by lambda. Running fileprob with no OOV also causes an error because we're taking the log of novel words that are assigned a probability of zero.

###Q4(b)
The model would not be smoothing at all, leaving the add lambda estimate of p(z|x,y) = c(x,y,z)/c(x,y). This would overfit the training data and not account for novel words that are in the test set. Running fileprob with a lambda of 0 also causes an error because we're taking the log of zero probability novel words.

###Q4(c)
If c(xyz) = c(xyz') = 0, then it doesn't follow that phat(z | xy) = phat(z' | xy). When working it out by hand, we have the terms c(yz) and c(yz') which may not be equal to 0. This also applies when c(xyz) = c(xyz') = 1 (c(yz) and c(yz') may be greater than 1). The resulting \
phat(z | x y) = ((lambda * V) * c(yz) + (lambda * V * phat(z))) /\
((c(xy) + lambda * V) * (c(y) + lambda * V))

###Q4(d)
Increasing lambda decreases the probability of words that appear frequently in training, and increases the probability of OOV words. 


###Q8
We're trying to maximize p(U | w) (or the log of this probability) since it gives us an indication of how much the vector w resembles the audio utterance U. We're also trying to maximize p(w): the prior probability of the sentence looking like English using our trigram language model. Thus, what we are ultimately trying to maximize is the product of these two probabilities. This would allow us to combine the audio utterance knowledge and english similarity (prior trigram language model probability) knowledge when selecting a candidate. We can compute this using the log probability in the speech file to represent the estimate of p(U | w) and the product of the probability of each trigram in the sentence given the language model used for p(w). 