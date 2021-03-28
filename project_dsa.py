import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
from statistics import stdev

file = r"C:/Users/pujit/Downloads/test.txt" #file location
x,x_err,y,y_err= np.loadtxt(file, unpack=True, usecols=[0,1,2,3])
n = len(x)
spearman_coeff,p_value = stats.spearmanr(x,y)
student_t,p_value = stats.ttest_ind(x,y)

def fisher_transformation(spearman_coeff):
    return np.arctanh(spearman_coeff)

def z_score(n,fisher):
    return fisher*np.sqrt((n-3)/1.060)
print("standard spearman coefficient is ",spearman_coeff)
print("standard z_score is ",z_score(len(x),fisher_transformation(spearman_coeff)))
"""
print(spearman_coeff)
print(student_t)
print(z_score)"""


def bootstrap(x,y):
    n = len(x)
    x_resample = []
    y_resample = []
    for i in range(n):
        rand = random.randint(0, n-1)
        x_resample.append(x[rand])
        y_resample.append(y[rand])
    return x_resample,y_resample

#x_r , y_r = bootstrap(x,y)

def perturbation(x,x_err,y,y_err):
    n = len(x)
    x_perturb = []
    y_perturb = []
    for i in range(n):
        rand = np.random.randn(1)
        x_perturb.append(x[i]+rand*x_err[i])
        rand = np.random.randn(1)
        y_perturb.append(y[i]+rand*y_err[i])
    return x_perturb,y_perturb

#x_p,y_p = perturbation(x,x_err,y,y_err)

def composite(x,x_err,y,y_err):
    n = len(x)
    x_composite = []
    y_composite = []
    for i in range(n):
        rand1 = random.randint(0, n-1)
        rand2 = np.random.randn(1)
        x_composite.append(x[rand1]+rand2*x_err[rand1])
        rand2 = np.random.randn(1)
        y_composite.append(y[rand1]+rand2*y_err[rand1])
    return x_composite,y_composite

#x_c,y_c = composite(x,x_err,y,y_err)

iterations = 1000

spearman_boot = []
spearman_perturb = []
spearman_composite = []
z_score_boot = []
z_score_composite = []
z_score_perturb = []
for i in range(iterations):
    x_r , y_r = bootstrap(x,y)
    spearman_coeff1,p_value = stats.spearmanr(x_r,y_r)
    spearman_boot.append(spearman_coeff1)
    z_score_boot.append(z_score(n,fisher_transformation(spearman_coeff1)))
    x_p,y_p = perturbation(x,x_err,y,y_err)
    spearman_coeff2,p_value = stats.spearmanr(x_p,y_p)
    spearman_perturb.append(spearman_coeff2)
    z_score_perturb.append(z_score(n,fisher_transformation(spearman_coeff2)))
    x_c,y_c = composite(x,x_err,y,y_err)
    spearman_coeff3,p_value = stats.spearmanr(x_c,y_c)
    spearman_composite.append(spearman_coeff3)
    z_score_composite.append(z_score(n,fisher_transformation(spearman_coeff3)))

plot1 = plt.figure (1)
plt.hist(spearman_boot,histtype ='step',label = 'bootstrap' )
plt.hist(spearman_perturb,histtype ='step',label = 'perturbation')
plt.hist(spearman_composite,histtype ='step',label = 'composite')
plt.xlabel("Spearman coefficient")
plt.ylabel("Frequency")
plt.legend()

plot2 = plt.figure(2)
plt.hist(z_score_boot,histtype ='step',label = 'bootstrap' )
plt.hist(z_score_perturb,histtype ='step',label = 'perturbation')
plt.hist(z_score_composite,histtype ='step',label = 'composite')
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.legend()
print("spearman coefficient using bootstrap method= {} ± {}".format(np.mean(spearman_boot), stdev(spearman_boot)))
print("spearman coefficient using Perturbation method= {} ± {}".format(np.mean(spearman_perturb), stdev(spearman_perturb)))
print("spearman coefficient using composite method= {} ± {}".format(np.mean(spearman_composite), stdev(spearman_composite)))
print("z_score using composite method= {} ± {}".format(np.mean(z_score_boot), stdev(z_score_boot)))
print("z_score using composite method= {} ± {}".format(np.mean(z_score_perturb), stdev(z_score_perturb)))
print("z_score using composite method= {} ± {}".format(np.mean(z_score_composite), stdev(z_score_composite)))