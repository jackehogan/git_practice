print('hello you')

# should this be the place for baysian opt?

import pandas as pd
import numpy as np
from matplotlib import pyplot

filepath = r'C:\Users\hogan\OneDrive\Documents\Clipboard\ChicagoShiftData.xlsx'
df = pd.read_excel(filepath)
print(df.columns)
dflvn = df[df['Agent Req']=='LVN']
# pyplot.scatter(dflvn['Average of Pay'], dflvn['Average of Net Rev'])
# pyplot.show()

from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm

# objective function
def objective(x,dflvn, noise=0.1):
	noise = np.random.normal(loc=0, scale=noise)
	return df[df['Average of Pay']==x]['Average of Net Rev'] + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)

# plot real observations vs surrogate function
def plot(X, y, model):
    # scatter plot of inputs and real objective function
    pyplot.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = np.asarray(np.arange(min(X), max(X), 0.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples)
    # show the plot
    pyplot.show()

# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs


# optimize the acquisition function
def opt_acquisition(X, y, model):
    # random search, generate random samples
    Xsamples = np.random.random(100)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = np.argmax(scores)
    return Xsamples[ix, 0]

# define the model
model = GaussianProcessRegressor()
X = np.asarray(dflvn['Average of Pay'])
y = np.asarray(dflvn['Average of Net Rev'])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# fit the model
model.fit(X, y)
# plot the surrogate function
plot(X, y, model)

# perform the optimization process
for i in range(100):
    # select the next point to sample
    x = opt_acquisition(X, y, model)
    # sample the point
    actual = objective(x,dflvn)
    # summarize the finding
    est, _ = surrogate(model, [[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # add the data to the dataset
    X = np.vstack((X, [[x]]))
    y = np.vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)

# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = np.argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))