import sys, time, numpy as np
from aran.env import MultiCellLAEnv, Config
from aran import policies as P, agents as A
m=sys.argv[1]; n=int(sys.argv[2])
cfg=Config(); env=MultiCellLAEnv(cfg)
t0=time.time()
net,curve=A.train(m,cfg,0,n,P.OLLA(0.1,0.5),log=print)
pol=A.NetPolicy(net,m)
ks=[P.run_episode(env,pol,s) for s in range(5000,5030)]
print(m, n, "time", round(time.time()-t0), "se", round(np.mean([k['se'] for k in ks]),3), "bler", round(np.mean([k['bler'] for k in ks]),3), "edge", round(np.mean([k['edge'] for k in ks]),3), "pow", round(np.mean([k['power_db'] for k in ks]),2))
