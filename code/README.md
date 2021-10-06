marhaba

###Q1
Perplexity = 2^(-l), l = (total log probability) / (number of words). Thus,
Perplexity per word of sample1: 68.53 \
Perplexity per word of sample2: 95.24 \
Perplexity per word of sample3: 89.60

If we train on the larger switchboard corpus, all the log probabilities are more negative, thus causing the perplexities for each of the sample files to be greater. This could be due to overfitting.

###Q3(a)
All gen test files were classified as gen (error rate of 0%). 93.33% of spam test files were classified as gen (error rate of 93.33%). Overall this results in a dev file error rate of 84/270 (31.11%). 

###Q3(b)
All english test files were classified as english (error rate of 0%). 100%% of spanish test files were classified as english (error rate of 100%%). Overall this results in a dev file error rate of 119/339 (49.79%).

###Q3(c)
A prior probability of gen as large as .22 will classify all the gens as spam.

###Q3(d)
For gen test files, the lowest cross entropy we could achieve was 9.04616 bits per token with a lambda value of .008. For spam test files, the lowest cross entropy we could achieve was 9.09572 bits per token with a lambda value of .005.

###Q3(e)
