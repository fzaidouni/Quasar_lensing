#! /usr/bin/env python

import numpy as np
from math import *

from astroML.time_series import generate_damped_RW

import matplotlib.pyplot as plt

import argparse

import random



# e.g.
# LC_DRW.py -tau 90 -sf 0.2 -xmu 1. -sn r

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--time_initial'  , '-ti  ', type=float, default=-365.        ,help='Set the initial time (default: t=-365 [day])')
parser.add_argument('--time_final'    , '-tf  ', type=float, default=365.         ,help='Set the final time (default: t=365 [day])')
parser.add_argument('--delta_time'    , '-dt  ', type=float, default=0.1          ,help='Set the time interval (default: dt=0.1 [day])')
parser.add_argument('--fname'         , '-fn  ', type=str  , default='LC_DRW.txt' ,help='Set the output filename (default: LC_DRW.txt)')

parser.add_argument('--tau'           , '-tau ', type=float, default=300          ,help='Set the relaxation time (default: tau=300)')
parser.add_argument('--strc_func_inf' , '-sf  ', type=float, default=0.1          ,help='Set the structure function at infinity (default: SFinf=0.1)')
parser.add_argument('--xmean'         , '-xmu ', type=float, default=1.           ,help='Set the mean value of random walk (default: Xmean=1.)')
parser.add_argument('--ran_seed'      , '-sn  ', type=str  , default='123'        ,help='Set the random seed (r: random, snnn: random seed)')
parser.add_argument('--redshift_src'  , '-zs'  , type=float, default=0.5          ,help='Set the redshift of source (default: zs=0.5)')

parser.add_argument('--target_dir'    , '-td'  ,             default='.',          help='Set the output directory')


args = parser.parse_args()

ti = args.time_initial
tf = args.time_final
dt = args.delta_time
fn = args.fname

xmean = args.xmean
tau = args.tau
SFinf = args.strc_func_inf
zs = args.redshift_src
sn = args.ran_seed

stem_out = args.target_dir


if (sn == 'r'):
    np.random.seed()
else:
    sn = int(sn)
    np.random.seed(sn)

###########################################


########################
#      light curve     #
########################

for i in [1]:

    sn = np.random.randn(1)

    print('-------------------------------')
    print('       light curve (DRW)       ')
    print('-------------------------------')
    print('Xmean =',xmean)
    print('tau   =',tau,'day')
    print('zs    =',zs)
    print('SFinf =','{0:.3f}'.format(SFinf))
    print('sn    =',sn)

    #t_drive = np.arange(ti, tf, dt)
    t_drive = np.arange(ti, tf+dt, dt)
    n=5
    t_drive1 = t_drive+n
    t_drive2 = t_drive+n*2
    t_drive3 = t_drive+n*3

    f_drive = generate_damped_RW(t_drive, tau=tau, z=zs, SFinf=SFinf, xmean=xmean)

    f_drive = abs(f_drive)
    mean = np.mean(f_drive)
    std  = np.std( f_drive)

    print('Light Curve:','from',ti,'to',tf,'days')
    print('mean =','{0:.3f}'.format(mean),'std =','{0:.3f}'.format(std))

    fn = stem_out+'/'+fn

    #np.savetxt(fn,np.array([t_drive, f_drive]).T,fmt='%f')

    plt.plot(t_drive, f_drive,  color='black', label='DRW')
    plt.plot(t_drive1, f_drive, color='red', label='DRW1')
    plt.plot(t_drive2, f_drive, color='blue', label='DRW2')
    plt.plot(t_drive3, f_drive, color='green', label='DRW3')
    plt.xlabel('t (days)')
    plt.ylabel('Fraction Variation')
    plt.legend(loc=3)
    plt.show()


    plt.scatter(t_drive, f_drive,  color='black', label='DRW')

    plt.xlabel('t (days)')
    plt.ylabel('Fraction Variation')
    plt.legend(loc=3)

    sample_t = np.array([])
    sample_f = np.array([])
    i=1
    s=0
    N = 365*2+1
    while i<= N :
        if ((random.choice([1, 2, 3])) == 1):
            sample_t = np.append(sample_t,t_drive[i-1+s:i+9+s])
            sample_f = np.append(sample_f, f_drive[i-1+s:i+9+s])
        s+=9
        i+=1
    #print(np.sort(sample_t)[0:10], len(sample_t))
    #plt.scatter(sample_t, sample_f, color='red', label='DRW')

    t_final = np.array([])
    f_final = np.array([])
    l=0
    #print(int(len(sample_t)/10))
    for i in range(1,int(len(sample_t)/10)+1):
        # print(i)
        rand = random.randint(1,11)
        list_t = sample_t[0+10*(i-1):10*i]
        list_f = sample_f[0+10*(i-1):10*i]
        index = random.choice((range(len(list_t))))
        t_final = np.append(t_final,list_t[index])
        f_final = np.append(f_final,list_f[index])
        # print(list_[index])
    print(t_final,f_final)
    plt.scatter(t_final,f_final,color='blue')
    plt.show()




