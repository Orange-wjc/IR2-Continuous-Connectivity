import json,subprocess,socket
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
ROOT=REPO/'experiments/native_easy10_cycles_20261018_seed101'
assert json.loads((WORK/'preflight_checks.json').read_text())['passed']
assert not ROOT.exists(),'Do not duplicate or overwrite an existing diagnostic'
for unit in ['ir2-native-easy100-cycles-20261018','ir2-native-easy100-cycles-review-20261018','ir2-native-easy10-cycles-20261018','ir2-native-easy10-cycles-review-20261018']:
 state=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],stdout=subprocess.PIPE,text=True,check=True).stdout.strip()
 assert state not in ('active','activating','deactivating'),(unit,state)
assert (REPO/'experiments/native_easy100_cycles_20261018_seed101/STOP').exists()
with socket.socket() as sock:sock.bind(('127.0.0.1',6009))
common=['systemd-run','--user','--working-directory='+str(REPO)]
subprocess.run(common+['--unit=ir2-native-easy10-cycles-20261018','--description=Repeated ten original easy maps learnability diagnostic',
 '--property=Restart=on-failure','--property=RestartSec=30','--property=TimeoutStopSec=120',
 '--property=StartLimitIntervalSec=3600','--property=StartLimitBurst=4',
 '/bin/bash',str(WORK/'run_training.sh')],check=True)
subprocess.run(common+['--unit=ir2-native-easy10-cycles-review-20261018','--description=Persistent RAM and checkpoint guard for ten-map diagnostic',
 '--property=Restart=on-failure','--property=RestartSec=30',
 '/home/wyz/anaconda3/envs/ir2_train/bin/python',str(WORK/'monitor_training.py')],check=True)
subprocess.run(common+['--unit=ir2-native-easy10-tensorboard-20261018','--description=TensorBoard ten-map diagnostic localhost 6009',
 '--property=Restart=on-failure','--property=RestartSec=10','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
 '/home/wyz/anaconda3/envs/ir2_train/bin/tensorboard','--logdir',str(ROOT/'tensorboard'),
 '--host','127.0.0.1','--port','6009','--reload_interval','10'],check=True)
print('Started separate diagnostic training, RAM guard and TensorBoard on 6009; old training remains stopped.')
