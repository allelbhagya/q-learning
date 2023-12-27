# q-learning
quality of a certain action in a given state

obtaining a function Q(s,a) that predicts the best action a in state s in order to maximize a cumulative reward

Q(s,a) = r{itermediate reward} + [(discount factor) * max-a' Q(s', a')] {max future reward}

### first-visit monte carlo policy evaluation

initialize 

pi <- policy to be evaluated
V <- arbritrary state-value function
returns(s) empty list, for all s belonging to S

repeat forever:
    (a) generate an episode using pi
    (b) for each state s appearing in the episode:
        R- return following the first occurence of s
        appebd R to returns(s)
        V(s) - average (Returns(s))