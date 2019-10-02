import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """
    
    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    poss_states = [None] * num_time_steps
    possible_prev_actions = ['left', 'right', 'up', 'down', 'stay']
    for i in range(0,num_time_steps):
        poss_states[i] = []
        if observations[i] == None:
            poss_states[i] = all_possible_hidden_states
        else:                
            [x,y] = observations[i]
            for action in possible_prev_actions:
                poss_states[i].append((x,y,action))
                poss_states[i].append((x-1,y,action))
                poss_states[i].append((x+1,y,action))
                poss_states[i].append((x,y+1,action))
                poss_states[i].append((x,y-1,action))
    # TODO: Compute the forward messages
    
    for i in range(0,num_time_steps):
        forward_messages[i] = rover.Distribution()
        if observations[i] == None:
            if i == 0:
                for state in poss_states[i]:
                    forward_messages[i][state] = prior_distribution[state]*1
            else:
                for state in poss_states[i]:
                    forward_messages[i][state] = 1*np.sum([forward_messages[i-1][zn1]*\
                                     transition_model(zn1)[state] for zn1 in poss_states[i-1]])
            forward_messages[i].renormalize()    
            
        else:
            if i == 0:
                for state in poss_states[i]:
                    forward_messages[i][state] = prior_distribution[state]*observation_model(state)[observations[0]]
            else:
                for state in poss_states[i]:
                    forward_messages[i][state] = observation_model(state)[observations[i]]*\
                        np.sum([forward_messages[i-1][zn1]*transition_model(zn1)[state] \
                             for zn1 in poss_states[i-1]])
            forward_messages[i].renormalize()    
    # TODO: Compute the backward messages
    
    for i in range(num_time_steps-1,-1,-1):
        backward_messages[i] = rover.Distribution()

        if i == num_time_steps-1:
            for state in all_possible_hidden_states:
                backward_messages[i][state] = 1
        else:
            if observations[i+1] == None:
                for state in poss_states[i]:
                    backward_messages[i][state] = np.sum([backward_messages[i+1][zn1]*\
                                  transition_model(state)[zn1] for zn1 in poss_states[i+1]])
            else:    
                for state in poss_states[i]:
                    backward_messages[i][state] = np.sum([backward_messages[i+1][zn1]*\
                    observation_model(zn1)[observations[i+1]]*transition_model(state)[zn1] \
                    for zn1 in poss_states[i+1]])
                    
            backward_messages[i].renormalize() 
#     TODO: Compute the marginals 
    for i in range(0,num_time_steps):
        marginals[i] = rover.Distribution()
#        [x,y] = observations[i]
#        possible_states = list(transition_model([x,y,'stay']).keys())
        for state in all_possible_hidden_states:

            marginals[i][state] = forward_messages[i][state]*backward_messages[i][state]
        marginals[i].renormalize()

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    poss_states = [None] * num_time_steps
    phi = [None] * (num_time_steps-1)
    possible_prev_actions = ['left', 'right', 'up', 'down', 'stay']
    estimated_hidden_states = []
    for i in range(0,num_time_steps):
        poss_states[i] = []
        if observations[i] == None:
            poss_states[i] = all_possible_hidden_states
        else:                
            [x,y] = observations[i]
            for action in possible_prev_actions:
                poss_states[i].append((x,y,action))
                poss_states[i].append((x-1,y,action))
                poss_states[i].append((x+1,y,action))
                poss_states[i].append((x,y+1,action))
                poss_states[i].append((x,y-1,action))
                
    for i in range(0,num_time_steps):
        w[i] = rover.Distribution()
        if observations[i] == None:
            if i == 0:
                for state in poss_states[i]:
                    w[0][state] = np.log(prior_distribution[state])
            else:
                phi[i-1] = dict()
                for state in poss_states[i]:
                    w[i][state] = np.max([np.log(transition_model(zn1)[state]) + w[i-1][zn1] for zn1 in poss_states[i-1]])
                    phi[i-1][state] = poss_states[i-1][np.argmax([np.log(transition_model(zn1)[state]) + w[i-1][zn1] for zn1 in poss_states[i-1]])]
        else:
            if i == 0:
                for state in poss_states[i]:
                    w[0][state] = np.log(prior_distribution[state]) + np.log(observation_model(state)[observations[0]])
            else:
                phi[i-1] = dict()
                for state in poss_states[i]:
                    w[i][state] = np.log(observation_model(state)[observations[i]]) + \
                        np.max([np.log(transition_model(zn1)[state]) + w[i-1][zn1] for zn1 in poss_states[i-1]])
                    phi[i-1][state] = poss_states[i-1][np.argmax([np.log(transition_model(zn1)[state]) + w[i-1][zn1] for zn1 in poss_states[i-1]])]
            
#        w[i].renormalize()    
    max_val = -1000
    for key in w[-1].keys():
        if w[-1][key]>max_val:
            max_val = w[-1][key]
            state_max = key
            
    estimated_hidden_states.append(state_max)
    for i in range(num_time_steps-2,-1,-1):
        state_max_new = phi[i][state_max]
        estimated_hidden_states.insert(0,state_max_new)
        state_max = state_max_new
        
    return estimated_hidden_states

def Pe_Viterbi(estimated_states):
    e_sum = 0
    for i in range(len(estimated_states)):
        if estimated_states[i]==hidden_states[i]:
            e_sum = e_sum + 1
    error = 1 - e_sum/100
    return error
    
    
def Pe_FwdBwd(marginals):
    e_sum = 0
    for i in range(len(marginals)):
        est_state = marginals[i].get_mode()
        if est_state==hidden_states[i]:
            e_sum =  e_sum + 1
    return 1-e_sum/100

def get_sequence(marginals):
    seq = []
    for i in range(len(marginals)):
        seq.append(marginals[i].get_mode())
        
    return seq
    
def check_seq(seq,transition_model):
    for i in range(1,len(seq)):
        if seq[i] not in transition_model(seq[i-1]):
            return 'Not Valid Sequence',seq[i-1],seq[i],i-1
            
    return 'Valid sequence'
    

if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
#    estimated_states = [None]*num_time_steps
#    marginals = [None]*num_time_steps
#    estimated_states = [None]*num_time_steps
#    marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
    
    print('The error for Viterbi-Alg: ', Pe_Viterbi(estimated_states))
    print('The error for the Forward-Backward Alg: ', Pe_FwdBwd(marginals))    
    fwdbwd_seq = get_sequence(marginals)
    print('Check the Forward-Backward sequence: ', check_seq(fwdbwd_seq,rover.transition_model))
    print('Check the Viterbi sequence:' ,check_seq(estimated_states,rover.transition_model))
    
