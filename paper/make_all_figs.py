"""All figures of the paper, drawn with one style. Data: data.py, data2.py, data3.py; committor curves in
data_committor_curves.npy (read from the Step 23 notebook plot's markers, accurate to about 0.003)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
sys.path.insert(0,'.')
from data import CL, VEH, P44, A19, FGEN, kind
from data2 import R20, C20, NULL_RAW, NULL_CAL, ROB
from data3 import A22, LEV, MASS, FINAL, EXC_D, EXC_S, PLANT_ALL, PLANT_DEER, POWER
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.8,
    'font.family':'serif','mathtext.fontset':'cm','axes.linewidth':0.6,'savefig.bbox':'tight','savefig.pad_inches':0.05,
    'axes.spines.top':False,'axes.spines.right':False,'axes.titlepad':7,'legend.frameon':False})
COL={'va':'#1f6aa5','vv':'#e08214','aa':'#1b9e77'}; NAME={'va':'vehicle / animal','vv':'vehicle / vehicle','aa':'animal / animal'}; ORDER=['va','vv','aa']
K=np.array([kind(p) for p in P44])
def logticks(ax, axis, vals):
    A=ax.yaxis if axis=='y' else ax.xaxis; A.set_major_locator(FixedLocator(list(vals))); A.set_major_formatter(FixedFormatter([f'{v:g}' for v in vals])); A.set_minor_locator(NullLocator())
def save(fig, name): fig.savefig(f'figs/{name}.pdf'); plt.close(fig)

# ---------------- Figure 1: pitchfork vs fold ----------------
def branches(w, taus):
    eta=0.5*np.log(w/(1-w)); m=np.linspace(-1.6,1.6,64001); out=[]
    for t in taus:
        g=np.tanh(m/t+eta)-m; idx=np.where(np.sign(g[:-1])!=np.sign(g[1:]))[0]
        for i in idx:
            r=m[i]-g[i]*(m[i+1]-m[i])/(g[i+1]-g[i]); out.append((t,r,(1/t)*(1/np.cosh(r/t+eta)**2)-1<0))
    return np.array(out)
taus=np.linspace(0.08,1.6,1400)
fig,ax=plt.subplots(1,3,figsize=(7.4,2.5),gridspec_kw={'width_ratios':[1,1,1.05],'wspace':0.42})
for k,(w,title) in enumerate([(0.5,'(a) equal classes: pitchfork'),(0.75,'(b) unequal classes: fold')]):
    B=branches(w,taus); s=B[:,2]>0.5
    ax[k].scatter(B[s,0],B[s,1],s=0.6,c='k',lw=0); ax[k].scatter(B[~s,0],B[~s,1],s=0.3,c='#c0392b',lw=0)
    ax[k].set_xlabel(r'noise level $\tau$'); ax[k].set_title(title); ax[k].set_ylim(-1.5,1.55); ax[k].set_xlim(0.08,1.6); ax[k].set_yticks([-1,0,1])
    if k==0:
        ax[k].set_ylabel(r'order parameter $m$'); ax[k].axvline(1.0,color='grey',ls=':',lw=0.8)
        ax[k].plot([],[],'k-',label='peak'); ax[k].plot([],[],'-',color='#c0392b',label='dip'); ax[k].legend(loc='upper right',handlelength=1.2)
    else:
        tb=B[~s][:,0].max(); ax[k].axvline(tb,color='grey',ls=':',lw=0.8); ax[k].text(tb+0.04,1.42,r'$\tau_{\rm bif}$',fontsize=9,va='center')
        ax[k].text(1.12,0.62,'majority class\n(present from the start)',fontsize=7.5,ha='center',va='top')
        ax[k].annotate('minority class\nappears suddenly',xy=(tb,-0.75),xytext=(1.1,-1.08),fontsize=7.5,ha='center',va='center',arrowprops=dict(arrowstyle='->',lw=0.6))
xs=np.linspace(-3,3,1000); tb=branches(0.75,np.linspace(0.3,0.6,3000)); tb=tb[tb[:,2]<0.5][:,0].max()
for t,c,lab in [(1.2,'#8c8c8c',r'$\tau=1.2$ (before)'),(tb,'#c0392b',r'$\tau=\tau_{\rm bif}$ (at the split)'),(0.25,'k',r'$\tau=0.25$ (after)')]:
    p=0.75*np.exp(-(xs-1)**2/(2*t))+0.25*np.exp(-(xs+1)**2/(2*t)); ax[2].plot(xs,p/p.max(),color=c,label=lab,lw=1.2)
ax[2].set_xlabel(r'order parameter $m$'); ax[2].set_ylabel('density (scaled)'); ax[2].set_title(r'(c) the distribution in case (b)'); ax[2].set_yticks([]); ax[2].set_ylim(0,1.62)
ax[2].legend(loc='upper left',fontsize=7.4,handlelength=1.4)
save(fig,'fig1_fold')

# ---------------- Figure: committor decisions ----------------
CC=np.load('data_committor_curves.npy')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.75),gridspec_kw={'wspace':0.38})
ax[0].plot(LEV,CC[0],'o-',color='#2166ac',ms=4,lw=1.2,label='same superclass'); ax[0].plot(LEV,CC[1],'s-',color='#e08214',ms=4,lw=1.2,label='same class')
ax[0].axhline(0.52,color='#2166ac',ls=':',lw=0.8); ax[0].axhline(0.10,color='#e08214',ls=':',lw=0.8)
ax[0].text(0.105,0.495,'chance (superclass)',fontsize=7,color='#2166ac',ha='right',va='top'); ax[0].text(0.105,0.075,'chance (class)',fontsize=7,color='#e08214',ha='right',va='top')
ax[0].set_xscale('log'); ax[0].set_xlim(10,0.08); logticks(ax[0],'x',(8,4,2,1,0.5,0.25,0.1)); ax[0].set_ylim(0,1.05)
ax[0].set_xlabel(r'noise level of the paused image $\sigma$  (generation $\rightarrow$)'); ax[0].set_ylabel('fraction of 12 restarts that agree'); ax[0].set_title('(a) agreement of restarts with the original'); ax[0].legend(loc='lower right',bbox_to_anchor=(1.0,0.17))
cats=['va','vv','aa']; dec={0.05:[2.16,1.61,0.87],0.1:[3.15,2.60,1.52],0.2:[5.29,4.01,3.02]}; sb=[1.90,1.01,0.38]
x=np.arange(3)
for thr,mk,c in [(0.2,'^','#9ecae1'),(0.1,'o','#4292c6'),(0.05,'v','#08519c')]: ax[1].plot(x,dec[thr],mk+'-',color=c,ms=5,lw=1,label=f'decision, threshold {thr}')
ax[1].plot(x,sb,'D--',color='k',ms=4.5,lw=1,label=r'density transition $\sigma_{\rm bif}$')
ax[1].set_yscale('log'); logticks(ax[1],'y',(0.25,0.5,1,2,4,8)); ax[1].set_ylim(0.25,45); ax[1].set_xlim(-0.35,2.35)
ax[1].set_xticks(x); ax[1].set_xticklabels([NAME[c].replace(' / ',' /\n') for c in cats]); ax[1].set_ylabel(r'noise level (median over pairs)'); ax[1].set_title('(b) when pairs are decided'); ax[1].legend(loc='upper right',fontsize=7.2)
save(fig,'fig_committor')

# ---------------- Figure 2: census ----------------
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1.05,1],'wspace':0.32})
rng=np.random.default_rng(3)
for i,k in enumerate(ORDER):
    j=np.where(K==k)[0]; xx=i+rng.uniform(-0.2,0.2,len(j)); y=R20[j,0]
    ax[0].errorbar(xx,y,yerr=[y-R20[j,1],R20[j,2]-y],fmt='o',ms=3,color=COL[k],elinewidth=0.5,alpha=0.9); ax[0].hlines(np.median(y),i-0.32,i+0.32,color=COL[k],lw=1.8)
ax[0].set_yscale('log'); logticks(ax[0],'y',(0.2,0.5,1,2,5)); ax[0].set_xticks(range(3)); ax[0].set_xticklabels([NAME[k].replace(' / ',' /\n')+f'\n({(K==k).sum()} pairs)' for k in ORDER]); ax[0].set_xlim(-0.5,2.5)
ax[0].set_ylabel(r'density transition $\sigma_{\rm bif}$'); ax[0].set_title('(a) where each class pair splits')
h0,hl0,hh0=R20[:,3],R20[:,4],R20[:,5]; h1,hl1,hh1=C20[:,3],C20[:,4],C20[:,5]
ax[1].axvspan(-NULL_RAW,NULL_RAW,color='#dddddd',lw=0); ax[1].axhspan(-NULL_CAL,NULL_CAL,color='#eeeeee',lw=0)
for rob,c,z,lab in [(False,'#9a9a9a',1,'within the noise floor in at least one'),(True,'#b2182b',3,'lopsided in both, same sign')]:
    j=ROB==rob; ax[1].errorbar(h0[j],h1[j],xerr=[h0[j]-hl0[j],hh0[j]-h0[j]],yerr=[h1[j]-hl1[j],hh1[j]-h1[j]],fmt='o',ms=3,color=c,elinewidth=0.5,zorder=z,label=lab)
ax[1].plot([-0.36,0.34],[-0.36,0.34],'k--',lw=0.6); ax[1].set_xlim(-0.36,0.34); ax[1].set_ylim(-0.38,0.47)
ax[1].set_xlabel(r'$h$, classifier log-odds'); ax[1].set_ylabel(r'$h$, recalibrated log-odds'); ax[1].set_title(r'(b) lopsidedness $h$ under two sticks'); ax[1].legend(loc='upper left',fontsize=7.2)
save(fig,'fig2_census')

# ---------------- Figure 4: pretrained, raw ----------------
ref,reftr,mod,rat,lo,hi,hr,hm,cs,nt=A19.T
REF_OUR=np.array([0.101,0.099,0.100,0.101,0.100,0.097,0.101,0.099,0.101,0.101]); REF_VIT=np.array([0.100,0.101,0.100,0.102,0.101,0.097,0.101,0.099,0.101,0.099])
EMREP_VIT=np.array([0.099,0.130,0.082,0.089,0.079,0.095,0.100,0.113,0.093,0.119]); HEUN_OUR=np.array([0.112,0.111,0.082,0.096,0.086,0.094,0.102,0.115,0.106,0.095]); HEUN_VIT=np.array([0.110,0.109,0.086,0.103,0.085,0.083,0.104,0.115,0.107,0.096])
fig=plt.figure(figsize=(7.4,5.0)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],hspace=0.62,wspace=0.34)
a0=fig.add_subplot(gs[0,0]); a2=fig.add_subplot(gs[0,1]); a1=fig.add_subplot(gs[1,:])
yl=np.clip(mod-lo*ref,0,None); yh=np.clip(hi*ref-mod,0,None)
for k in ORDER:
    j=K==k; a0.errorbar(ref[j],mod[j],yerr=[yl[j],yh[j]],fmt='o',ms=3,color=COL[k],elinewidth=0.5,alpha=0.9,label=NAME[k])
lim=[0.15,5]; a0.plot(lim,lim,'k--',lw=0.6); a0.set_xscale('log'); a0.set_yscale('log'); a0.set_xlim(lim); a0.set_ylim(lim); logticks(a0,'x',(0.2,0.5,1,2,5)); logticks(a0,'y',(0.2,0.5,1,2,5))
a0.set_xlabel(r'$\sigma_{\rm bif}$ of the data'); a0.set_ylabel(r'$\sigma_{\rm bif}$ of the model'); a0.set_title('(a) where each pair splits'); a0.legend(loc='upper left',fontsize=7.2)
over=FGEN/REF_OUR; dov=np.array([np.log(over[a])-np.log(over[b]) for a,b in P44])
a2.scatter(dov,hm,s=13,c=[COL[k] for k in K],lw=0); a2.axhline(0,color='k',lw=0.5); a2.axvline(0,color='k',lw=0.5)
a2.set_xlabel(r'log over-production of class $a$ minus class $b$'); a2.set_ylabel(r'lopsidedness $h$ of the model'); a2.set_title('(b) the over-produced class wins the split')
o=np.argsort(-(FGEN/REF_OUR)); x=np.arange(10); w=0.2
for d,(vals,refv,lab,c) in enumerate([(FGEN,REF_OUR,'256-step SDE, our classifier','#b2182b'),(EMREP_VIT,REF_VIT,'256-step SDE, ViT','#ef8a62'),(HEUN_OUR,REF_OUR,'18-step Heun, our classifier','#2166ac'),(HEUN_VIT,REF_VIT,'18-step Heun, ViT','#67a9cf')]):
    a1.bar(x+(d-1.5)*w,(vals/refv)[o]-1,bottom=1,width=w,color=c,label=lab)
a1.axhline(1,color='k',lw=0.6); a1.set_xticks(x); a1.set_xticklabels([CL[i] for i in o]); a1.set_ylim(0.6,1.55); a1.set_yticks([0.6,0.8,1.0,1.2,1.4]); a1.set_xlim(-0.6,9.6)
a1.set_ylabel('generated / expected'); a1.set_title('(c) how often each class is generated'); a1.legend(loc='upper center',ncol=4,fontsize=7.4,bbox_to_anchor=(0.5,1.02),handlelength=1.2,columnspacing=1.2)
save(fig,'fig4_pretrained')

# ---------------- Figure: the direct test ----------------
fig=plt.figure(figsize=(7.4,4.6)); a0=fig.add_axes([0.07,0.11,0.30,0.76]); a1=fig.add_axes([0.66,0.585,0.32,0.30]); a2=fig.add_axes([0.66,0.11,0.32,0.30])
lr=np.log(A22[:,1]/A22[:,0]); d=np.abs(A22[:,6])**(2/3)-np.abs(A22[:,5])**(2/3)
for k in ORDER:
    j=K==k; a0.scatter(d[j],lr[j],s=15,c=COL[k],lw=0,label=NAME[k])
a0.axhline(0,color='k',lw=0.5); a0.axvline(0,color='k',lw=0.5)
b,a=np.polyfit(d,lr,1); xs=np.linspace(d.min(),d.max(),10); a0.plot(xs,a+b*xs,color='#555',lw=0.9)
a0.set_xlabel(r'error in $|h|^{2/3}$  (model $-$ data)'); a0.set_ylabel(r'error in $\log\sigma_{\rm bif}$  (model $-$ data)')
a0.set_title('(a) timing vs lopsidedness errors,\nuncorrected EDM'); a0.legend(loc='upper right',fontsize=7.2); a0.set_ylim(-1.0,1.05)
rows=['EDM as is, vs data','EDM vs mass-matched data','EDM conditional, vs data','EDM corrected, vs data','DDPM vs mass-matched data']
v=[0.106,0.974,0.87,0.889,0.985]; cols=['#b2182b','#ef8a62','#2166ac','#4393c3','#1b7837']; y=np.arange(5)[::-1]
a1.barh(y,v,0.62,color=cols); a1.axvline(0.981,color='#666',ls=':',lw=0.9)
a1.set_yticks(y); a1.set_yticklabels(rows); a1.set_xlim(0,1.0); a1.set_xticks([0,0.25,0.5,0.75,1.0])
a1.set_xlabel(r'agreement of $h$ with the data'); a1.set_title('(b) lopsidedness')
rows2=['EDM vs mass-matched data','EDM conditional, vs data','EDM corrected, vs mass-matched','DDPM vs mass-matched data']
pts=[(1.115,1.011,1.234,'#ef8a62'),(1.00,0.92,1.18,'#2166ac'),(1.069,0.902,1.296,'#4393c3'),(0.799,0.663,0.933,'#1b7837')]; y2=np.arange(4)[::-1]
a2.axvspan(0.98,1.29,color='#eeeeee',lw=0,zorder=0); a2.axvline(1,color='k',lw=0.5)
for yy,(m,lo_,hi_,c) in zip(y2,pts): a2.errorbar([m],[yy],xerr=[[m-lo_],[hi_-m]],fmt='o',color=c,capsize=2.5,ms=4.5)
a2.set_yticks(y2); a2.set_yticklabels(rows2); a2.set_xlim(0.6,1.4); a2.set_ylim(-0.6,3.6)
a2.set_xlabel('timing ratio (average over pairs)'); a2.set_title('(c) timing')
save(fig,'fig_direct')

# ---------------- Figure: planted errors ----------------
fig,ax=plt.subplots(1,2,figsize=(6.6,2.7),gridspec_kw={'wspace':0.38})
for P,lab,c,mk in [(PLANT_ALL,'all classes distorted','#2166ac','o'),(PLANT_DEER,'only deer distorted','#b2182b','s')]:
    xx=np.maximum(P[:,0],0.008); off=1.0 if c=='#2166ac' else 1.06
    ax[0].errorbar(xx*off,P[:,2],yerr=[P[:,2]-P[:,3],P[:,4]-P[:,2]],fmt=mk+'-',color=c,ms=3.8,capsize=2,label=lab); ax[1].plot(xx,P[:,1],mk+'-',color=c,ms=3.8,label=lab)
for a_ in ax: a_.set_xscale('log'); a_.set_xlabel('size of planted error (nats per class)'); a_.legend(loc='lower left' if a_ is ax[1] else 'upper left')
ax[0].axhline(1,color='k',lw=0.5); ax[0].set_ylabel('timing ratio, distorted / clean'); ax[0].set_title('(a) timing reacts early'); ax[0].set_ylim(0.6,1.9)
ax[1].set_ylabel(r'agreement of $h$ with clean data'); ax[1].set_title('(b) lopsidedness reacts late'); ax[1].set_ylim(0.55,1.03)
save(fig,'fig_plant')

# ---------------- Figure: class mass along trajectories, committor fidelity ----------------
fig,ax=plt.subplots(1,2,figsize=(7.6,3.1),gridspec_kw={'wspace':0.36}); cm=plt.get_cmap('tab10'); HL=(7,0,6,4)
for c in range(10):
    if c in HL: continue
    ax[0].plot(LEV,MASS[:,c]/0.1,'-',color='#bbbbbb',lw=0.7)
ax[0].plot([],[],'-',color='#bbbbbb',lw=0.7,label='other classes')
for c in HL: ax[0].plot(LEV,MASS[:,c]/0.1,'-',color=cm(c),lw=1.6,label=CL[c])
for c in range(10): ax[0].plot([0.068],[FINAL[c]/0.1],'o',color=cm(c) if c in HL else '#bbbbbb',ms=3.5,clip_on=False)
ax[0].axvline(2.42,color='#888',ls=':',lw=0.8); ax[0].text(2.42*1.06,1.66,'superclass\ndecided',fontsize=6.8,color='#555',ha='right',va='top')
ax[0].axvline(1.11,color='#888',ls=':',lw=0.8); ax[0].text(1.11/1.06,1.66,'class\ndecided',fontsize=6.8,color='#555',ha='left',va='top')
ax[0].set_xscale('log'); ax[0].set_xlim(9,0.065); logticks(ax[0],'x',(8,4,2,1,0.5,0.25,0.1)); ax[0].set_ylim(0.35,1.7)
ax[0].axhline(1,color='k',lw=0.5); ax[0].set_xlabel(r'noise level $\sigma$  (generation $\rightarrow$)'); ax[0].set_ylabel('class proportion / uniform')
ax[0].set_title("(a) class proportions along generation"); ax[0].legend(loc='upper center',bbox_to_anchor=(0.5,-0.2),fontsize=7.2,ncol=5,columnspacing=0.9,handlelength=1.3)
for E,lab,c in [(EXC_D,'class probabilities as is','#2166ac'),(EXC_S,'adjusted for class proportions','#e08214')]:
    ax[1].fill_between(LEV,E[:,1],E[:,2],color=c,alpha=0.16,lw=0); ax[1].plot(LEV,E[:,0],'o-',color=c,ms=3,lw=1.1,label=lab)
ax[1].axhline(0.088,color='#666',ls='--',lw=0.8,label="level at low noise (classifier's own error)")
ax[1].set_xscale('log'); ax[1].set_xlim(9,0.08); logticks(ax[1],'x',(8,4,2,1,0.5,0.25,0.1)); ax[1].set_ylim(0,0.25)
ax[1].set_xlabel(r'noise level $\sigma$  (generation $\rightarrow$)'); ax[1].set_ylabel('distance (total variation)'); ax[1].set_title('(b) model committor vs data'); ax[1].legend(loc='upper right',fontsize=7)
save(fig,'fig_massnoise')

# ---------------- Figure: corruptions (2 x 3) ----------------
fig,axs=plt.subplots(2,3,figsize=(7.4,4.7),gridspec_kw={'hspace':0.62,'wspace':0.42}); ax=axs.ravel()
series=[(['baseline','scale_0.995','scale_0.98','scale_0.95'],r'score scaled by $\lambda$ = 1, 0.995, 0.98, 0.95','#2166ac','o',-0.06),
        (['baseline','mismatch_1.1','mismatch_1.25'],r'noise level told as $\kappa\sigma$, $\kappa$ = 1, 1.1, 1.25','#e08214','s',0.06)]
for keys,lab,c,mk,off in series:
    P=np.array([POWER[k] for k in keys]); xs=np.arange(len(keys))+off
    ax[0].errorbar(xs,P[:,0],yerr=[P[:,0]-P[:,1],P[:,2]-P[:,0]],fmt=mk+'-',color=c,ms=3.8,capsize=2,lw=1)
    ax[1].errorbar(xs,P[:,3],yerr=[P[:,3]-P[:,4],P[:,5]-P[:,3]],fmt=mk+'-',color=c,ms=3.8,capsize=2,lw=1)
    ax[2].plot(xs,P[:,6],mk+'-',color=c,ms=3.8,lw=1); ax[3].plot(xs,P[:,7],mk+'-',color=c,ms=3.8,lw=1); ax[4].plot(xs,P[:,8],mk+'-',color=c,ms=3.8,lw=1,label=lab)
for a_,t,yl,ylim in [(ax[0],'(a) timing, as is',r'$\sigma_{\rm bif}$ ratio to baseline',(0.25,1.6)),(ax[1],'(b) timing, rebalanced',r'$\sigma_{\rm bif}$ ratio to baseline',(0.25,1.6)),
                     (ax[2],'(c) lopsidedness, rebalanced',r'agreement of $h$ with data',(0,1.05)),(ax[3],'(d) class imbalance','imbalance',(0,1.1)),(ax[4],'(e) image quality','FID (2,500 images)',(0,115))]:
    a_.set_title(t); a_.set_ylabel(yl); a_.set_ylim(*ylim); a_.set_xticks([0,1,2,3]); a_.set_xlabel('corruption level')
for a_ in (ax[0],ax[1]): a_.axhline(1,color='k',lw=0.5)
ax[5].axis('off'); h_,l_=ax[4].get_legend_handles_labels(); ax[5].legend(h_,l_,loc='center',fontsize=7.6,handlelength=1.8)
save(fig,'fig_power')

# ---------------- Appendix figure: symmetric null ----------------
null_raw=np.abs([-0.0096,-0.0468,-0.0446,-0.0111,-0.0196,-0.0034,0.0288,0.0072,-0.0131,0.0003,-0.0029,0.0288,-0.0268])
null_sym=np.abs([-0.0233,-0.0063,-0.0058,0.0021,-0.0011,0.0078,0.0243,0.0029,-0.0004,0.0086,0.0051,0.0012,0.0036])
real=np.abs(R20[:,3]); bins=np.linspace(0,0.30,31)
fig,ax=plt.subplots(figsize=(5.2,2.7))
ax.hist(real,bins=bins,color='#9ecae1',edgecolor='white',lw=0.5,label='44 real class pairs')
ax.hist(null_raw,bins=bins,histtype='step',color='#e08214',lw=1.6,label='13 exactly symmetric pairs')
ax.hist(null_sym,bins=bins,histtype='step',color='#1b7837',lw=1.4,ls='--',label='same, measuring stick made symmetric')
ax.axvline(0.047,color='#e08214',ls=':',lw=1); ax.text(0.051,7.4,'noise floor\n= 0.047',fontsize=7.4,color='#a0520c',va='top')
ax.set_xlabel(r'lopsidedness $|h|$'); ax.set_ylabel('number of pairs'); ax.set_xlim(0,0.30); ax.set_ylim(0,12.8); ax.legend(loc='upper right',fontsize=7.4)
save(fig,'fig3_null')
print('all figures written')
