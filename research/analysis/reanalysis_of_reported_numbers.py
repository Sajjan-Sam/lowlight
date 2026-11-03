import numpy as np
from scipy import stats
dets=['YOLOv8n','YOLOv10n','RT-DETR','FRCNN','DefDETR']
enh=['Raw','LIME','ZeroDCE++','SCI','Retinexformer','SNR','PairLIE','KinD++']
m50=np.array([
[.7332,.7309,.7626,.8176,.6697],
[.7233,.7240,.7593,.8219,.6578],
[.6925,.6996,.7205,.8036,.6321],
[.5892,.5952,.6615,.6031,.5184],
[.6675,.6809,.7009,.7838,.5904],
[.6707,.6776,.6823,.7642,.5613],
[.6008,.6258,.6696,.7558,.5271],
[.6394,.6483,.6729,.7641,.5620]])
m95=np.array([
[.4727,.4719,.4990,.5150,.4087],
[.4646,.4640,.4935,.5150,.4023],
[.4420,.4436,.4660,.5015,.3830],
[.3527,.3541,.4160,.3380,.2986],
[.4225,.4312,.4514,.4881,.3540],
[.4247,.4326,.4321,.4698,.3345],
[.3760,.3898,.4219,.4648,.3078],
[.4015,.4002,.4265,.4721,.3306]])
P=np.array([
[.8119,.7984,.8299,.8419,.8204],
[.7882,.8056,.8402,.8295,.7966],
[.7787,.7540,.8227,.8183,.7964],
[.7046,.7343,.8271,.7490,.7316],
[.7506,.7453,.8212,.8366,.7778],
[.7562,.7177,.8021,.8090,.7434],
[.6897,.7324,.8165,.7867,.7494],
[.7201,.7272,.8214,.8107,.7496]])
R=np.array([
[.6370,.6489,.7222,.7287,.6141],
[.6408,.6397,.7163,.7378,.6116],
[.6067,.6327,.6736,.7333,.5804],
[.5185,.5226,.5989,.5397,.4862],
[.5972,.6191,.6452,.6901,.5495],
[.6103,.6293,.6499,.6876,.5467],
[.5567,.5633,.6133,.6897,.4990],
[.5777,.5885,.6221,.6885,.5290]])
d50=m50[1:]-m50[0]; d95=m95[1:]-m95[0]
print("== Relative drop (%) mAP50 / mAP50-95, mean over detectors ==")
for i,e in enumerate(enh[1:]):
    r50=(d50[i]/m50[0]).mean()*100; r95=(d95[i]/m95[0]).mean()*100
    print(f"{e:14s} rel50={r50:6.1f}%  rel95={r95:6.1f}%  ratio95/50={r95/r50 if r50!=0 else float('nan'):.2f}")
print("\n== Per-detector mean relative drop over 7 enhancers (sensitivity) ==")
for j,d in enumerate(dets):
    print(f"{d:9s} mean rel50={(d50[:,j]/m50[0,j]).mean()*100:6.1f}%  excl SCI={(np.delete(d50[:,j],2)/m50[0,j]).mean()*100:6.1f}%  rel95={(d95[:,j]/m95[0,j]).mean()*100:6.1f}%")
print("\n== Ranking agreement of enhancers across detectors ==")
ranks=np.array([stats.rankdata(-m50[1:,j]) for j in range(5)])  # 5 x 7
k=5;n=7
Rsum=ranks.sum(0); S=((Rsum-Rsum.mean())**2).sum(); W=12*S/(k**2*(n**3-n))
chi2=k*(n-1)*W; p=1-stats.chi2.cdf(chi2,n-1)
print(f"Kendall's W = {W:.3f}, chi2={chi2:.2f}, p={p:.4f}")
taus=[]
for a in range(5):
    for b in range(a+1,5):
        t,_=stats.kendalltau(m50[1:,a],m50[1:,b]); taus.append(t); print(f"  tau({dets[a]},{dets[b]})={t:.2f}")
print("mean pairwise tau",np.mean(taus))
print("\n== Precision vs recall decomposition of the drop (mean over detectors) ==")
for i,e in enumerate(enh[1:]):
    dp=(P[i+1]-P[0]).mean(); dr=(R[i+1]-R[0]).mean()
    print(f"{e:14s} dP={dp:+.4f} dR={dr:+.4f}")
# P2/P3 decomposition (YOLOv8n)
print("\n== YOLOv8n decomposition: total P2 penalty = domain-shift (P3-P2) + information-loss (P1-P3) ==")
P1=.7332
for e,p2,p3 in [('LIME',.7233,.7315),('ZeroDCE++',.6925,.7292),('SCI',.5892,.6784),('Retinexformer',.6675,.7209),('SNR',.6707,.7071)]:
    tot=P1-p2; shift=p3-p2; loss=P1-p3
    print(f"{e:14s} total={tot:.4f}  shift={shift:.4f} ({shift/tot*100:4.0f}%)  irreversible={loss:.4f} ({loss/tot*100:4.0f}%)")
# size analysis
print("\n== Size ΔAP (YOLOv8n): correlation of small-gain with large-loss ==")
S=dict(LIME=(.004,-.006,-.007),ZDCE=(.023,-.030,-.031),RetF=(-.001,-.029,-.049),SNR=(-.018,-.048,-.039),KinD=(.019,-.032,-.079),PairLIE=(.006,-.078,-.101),SCI=(-.010,-.113,-.110))
raw=(.126,.349,.523)
for k_,v in S.items(): print(f"{k_:8s} rel S={v[0]/raw[0]*100:+5.1f}% M={v[1]/raw[1]*100:+5.1f}% L={v[2]/raw[2]*100:+5.1f}%")
# Rough statistical-significance estimate
print("\n== Rough SE of mAP50 on 737 test images ==")
# ExDark ~23.7k boxes / 7363 imgs ≈ 3.2 boxes/img → ~2.4k test boxes; empirical bootstrap SE of mAP50 at this scale is typically ~0.007-0.012
for se in (0.007,0.010,0.012):
    # paired difference SE is smaller due to correlation (same images): assume rho=0.8
    sd=np.sqrt(2*se**2*(1-0.8))
    print(f"SE={se}: paired-diff SE≈{sd:.4f}, 95% half-width≈{1.96*sd:.4f}")
