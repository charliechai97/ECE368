import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    spamfiles = file_lists_by_category[0]
    hamfiles = file_lists_by_category[1]
    w = []
    for spamfile in spamfiles:
        w.extend(util.get_words_in_file(spamfile))
        
    for hamfile in hamfiles:
        w.extend(util.get_words_in_file(hamfile))
        
#    n_spam = len(spam_words)
#    n_ham = len(ham_words)
    spam_count = util.get_counts(spamfiles)
    ham_count = util.get_counts(hamfiles)
    
    n = len(w)
    dict_spam = {wi : 0 for wi in w}
    dict_ham = {wi : 0 for wi in w}
    for key in dict_spam:
        dict_spam[key] = (spam_count[key]+1)/(n+2) 
        dict_ham[key] = (ham_count[key]+1)/(n+2)
                  
    probabilities_by_category = (dict_spam,dict_ham)

    
    
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    
    probabilities_by_category: output of function learn_distributions
    
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    spam_prob = probabilities_by_category[0]
    ham_prob = probabilities_by_category[1]
    w = list(spam_prob.keys())
    words = util.get_words_in_file(filename)
#    x_spam = np.zeros(len(spam_words))
#    x_ham = np.zeros(len(ham_words))
    x = np.zeros(len(w))
#    
    itr = 0
    for wi in w:
        if wi in words:
            x[itr] = 1
        itr = itr + 1
#    p_spam = np.sum(x_spam*np.log(spam_prob.values()) + (1-x_spam)*np.log(1-spam_prob.values()))  
#    p_ham = np.sum(x_ham*np.log(ham_prob.values()) + (1-x_ham)*np.log(1-ham_prob.values()))
#    i = 0
#    p_spam = 0
#    for keys in spam_prob:
#        p_spam += x_spam[i]*np.log(spam_prob[keys]) + (1-x_spam[i])*np.log(1-spam_prob[keys])
#        i += 1
#    
#    i = 0
#    p_ham = 0
#    for keys in ham_prob:
#        p_ham += x_ham[i]*np.log(ham_prob[keys]) + (1-x_ham[i])*np.log(1-ham_prob[keys])
#        i += 1
#   
    spam_values = np.array(list(spam_prob.values()))
    ham_values = np.array(list(ham_prob.values()))
    p_ham = np.log(prior_by_category[1])
    p_spam = np.log(prior_by_category[0])
#    p_spam = np.sum(x_spam*np.log(spam_values) + (1-x_spam)*np.log(1-spam_values)) + np.log(prior_by_category[0])
    
    for i in range(0,len(x)):
        p_spam += x[i]*np.log(spam_values[i]) + (1-x[i])*np.log(1-spam_values[i])
        p_ham += x[i]*np.log(ham_values[i]) + (1-x[i])*np.log(1-ham_values[i])
    
    
    if p_spam > p_ham:
        classification = 'spam'
    else:
        classification = 'ham'
        
    posterior = [p_spam, p_ham]
    classify_result = (classification, posterior)

    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    log_posteriories = []
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        log_posteriories.append(log_posterior)
     
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    thresholds = np.arange(-1000, 1000, 1)
    type1_E = np.zeros(len(thresholds))
    type2_E = np.zeros(len(thresholds))
    
    for i in range(0,len(thresholds)):
        performance_measures = np.zeros([2,2])
        threshold = thresholds[i]
#        print(threshold)
        k = 0
        for filename in (util.get_files_in_folder(test_folder)):
            log_posterior = log_posteriories[k]
#            print(log_posterior)
            k += 1
            if log_posterior[0] >= log_posterior[1] + threshold:
                label = 'spam'
            else:
                label = 'ham'
                 
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

#        template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        type1_E[i] = totals[0] - correct[0]
        type2_E[i] = totals[1] - correct[1]
        
    print(type1_E)
    print(type2_E)
    plt.plot(type2_E,type1_E,'*')
    plt.savefig("nbc.pdf")

   
    ## HOW DO YOU DEFINE TYPE ERRORS AND DOES THE DENOMINATOR IN THE MAP RULE
    ## HAVE TO BE CALCULATED SINCE ITS THE SAME FOR BOTH CASES?

 