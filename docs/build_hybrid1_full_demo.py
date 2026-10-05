"""Complete hybrid/1 reference run using real project sensing, not an RL policy."""
from collections import deque
from pathlib import Path
import base64
import json
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from sensor import sensor_work
from ss_realistic_model import SS_realistic_model

OUT=ROOT/'docs'/'hybrid1_full_demo'
COLORS=['#69baff','#ffc66f','#c59bff','#58e0b1']


def main():
    OUT.mkdir(exist_ok=True)
    raw=np.array(Image.open(ROOT/'DungeonMaps/test/hybrid/1.png').convert('L'))
    ground=np.where(raw>150,255,1); flat=ground.ravel();total_free=int(np.sum(ground==255))
    shared=np.full_like(ground,127); events=[];frames=[];sensing_cache={};signal_cache={}
    signal=SS_realistic_model(P_T=-20,threshold_ss=-70,gamma=2,gamma_obst=4,dist_o=35,PL_o=31,
                              X_g_min=0,X_g_max=0,K_min=0,K_max=0)
    nodes=[(x,y) for y in range(5,500,10) for x in range(5,500,10) if ground[y,x]==255]
    lookup={p:i for i,p in enumerate(nodes)};points=np.array(nodes)
    adjacency=[[] for _ in nodes]
    for i,(x,y) in enumerate(nodes):
        for dx,dy in [(10,0),(-10,0),(0,10),(0,-10)]:
            q=(x+dx,y+dy)
            if q in lookup and all(ground[y+dy*k//10,x+dx*k//10]==255 for k in range(11)):
                adjacency[i].append(lookup[q])
    def bfs(start):
        prev={start:None};todo=deque([start])
        while todo:
            current=todo.popleft()
            for j in adjacency[current]:
                if j not in prev:prev[j]=current;todo.append(j)
        return prev
    def path(prev,target):
        p=[target]
        while prev[p[-1]] is not None:p.append(prev[p[-1]])
        return p[::-1]
    def rssi(a,b):
        key=(tuple(a),tuple(b))
        if key not in signal_cache:signal_cache[key]=signal.compute_rssi(ground,np.array(a),np.array(b))
        return signal_cache[key]
    def q_values(pos):return [round(rssi(pos[a],pos[b]),6) for a in range(4) for b in range(a+1,4)]
    def connected(q):
        adj=[[] for _ in range(4)];k=0
        for a in range(4):
            for b in range(a+1,4):
                if q[k]>-70:adj[a].append(b);adj[b].append(a)
                k+=1
        seen={0};todo=[0]
        while todo:
            for j in adj[todo.pop()]:
                if j not in seen:seen.add(j);todo.append(j)
        return len(seen)==4
    def sense(pos,frame_index):
        changed=[]
        for p in pos:
            key=tuple(p)
            if key not in sensing_cache:
                belief=np.full_like(ground,127)
                sensor_work(np.array(p),80,belief,ground)
                sensing_cache[key]=np.flatnonzero(belief.ravel()!=127)
            ids=sensing_cache[key];new=ids[shared.ravel()[ids]==127]
            if len(new):
                shared.ravel()[new]=flat[new];changed.extend(new.tolist())
        if changed:events.append([frame_index,changed])
    def coverage():return float(np.count_nonzero(shared==255)/total_free*100)
    # Preserve the already validated separated opening and relay handoff.
    opening=json.loads((OUT/'opening_reference.json').read_text())
    phase_defs=opening['phases'][:]
    for k,f in enumerate(opening['frames']):
        pos=[tuple(p) for p in f['p']]
        if k%10==0 or k==len(opening['frames'])-1:sense(pos,k)
        q=q_values(pos);assert connected(q)
        frames.append({'p':pos,'q':q,'phase':f['phase'],'c':round(coverage(),5),'relay':phase_defs[f['phase']]['relay']})
    positions=[lookup[tuple(p)] for p in frames[-1]['p']]
    plans=[deque() for _ in range(4)];leader=0;mode=-1;stall=0;last_cov=coverage();travel=[0.0]*4
    def can_move(robot,target):
        a=nodes[positions[robot]];b=nodes[target]
        current=[nodes[i] for i in positions]
        for k in range(1,11):
            q=(a[0]+(b[0]-a[0])*k//10,a[1]+(b[1]-a[1])*k//10)
            candidate=current.copy();candidate[robot]=q
            if ground[q[1],q[0]]!=255 or not connected(q_values(candidate)):return False
        return True
    def move(robot,target,phase,relay):
        a=nodes[positions[robot]];b=nodes[target];current=[nodes[i] for i in positions]
        for k in range(1,11):
            current[robot]=(a[0]+(b[0]-a[0])*k//10,a[1]+(b[1]-a[1])*k//10)
            frames.append({'p':current.copy(),'q':q_values(current),'phase':phase,'c':round(coverage(),5),'relay':relay})
        positions[robot]=target
        sense([nodes[i] for i in positions],len(frames)-1)
        frames[-1]['c']=round(coverage(),5)
    def radio_safe_next(robot,destination):
        # Geometric shortest paths may cut through an RF shadow at a corner.
        # Search a different navigable route while the other poses are fixed.
        start=positions[robot];prev={start:None};todo=deque([start])
        original=positions[robot]
        while todo:
            current=todo.popleft()
            if current==destination:break
            for candidate in adjacency[current]:
                if candidate in prev:continue
                positions[robot]=current
                allowed=can_move(robot,candidate)
                positions[robot]=original
                if allowed:prev[candidate]=current;todo.append(candidate)
        positions[robot]=original
        if destination in prev:
            p=path(prev,destination)
            return p[1] if len(p)>1 else None
        return None
    print('Opening coverage using sensor.py:',round(coverage(),3),flush=True)
    for step in range(2200):
        cov=coverage()
        # The original project success threshold is 99%; seek >99.9% for this illustration.
        if cov>=99.9:break
        remaining=np.flatnonzero(shared[points[:,1],points[:,0]]!=255)
        if not len(remaining):
            # Target positions nearest the remaining unseen free pixels.
            y,x=np.where((ground==255)&(shared!=255))
            if not len(x):break
            remaining=np.unique([int(np.argmin(np.sum((points-[a,b])**2,axis=1))) for a,b in zip(x[::max(1,len(x)//50)],y[::max(1,len(y)//50)])])
        if not plans[leader]:
            choices=[]
            for j in range(4):
                prev=bfs(positions[j]);target=next((k for k in prev if k in remaining and k!=positions[j]),None)
                if target is not None:
                    p=path(prev,target);choices.append((len(p),j,p))
            if not choices:break
            _,leader,p=min(choices);plans=[deque() for _ in range(4)];plans[leader]=deque(p[1:])
        relay=(leader+3)%4
        newmode=0 if cov<75 else 1 if cov<95 else 2
        if newmode!=mode:
            mode=newmode
            titles=['继续推进：探索剩余分支','移动接应位置，探查远端区域','补查遗漏，达到完成标准']
            descriptions=['保留前面的分散片段，随后继续探查未观察区域。接应位置不再固定在中央。','根据剩余未知区域调整分工；转角处必要时靠近，通过后再寻找局部分散机会。','直接使用项目sensor.py检查尚未观察的空闲像素。进度来自实际传感器更新，不是预设百分比。']
            phase_defs.append({'title':titles[mode],'description':descriptions[mode],'relay':relay,'start':len(frames)})
        phase=len(phase_defs)-1
        blocked=False;moved=False
        if plans[leader]:
            target=plans[leader][0]
            if can_move(leader,target):move(leader,target,phase,relay);plans[leader].popleft();moved=True
            else:blocked=True
        for j in [(leader+i)%4 for i in range(1,4)]:
            prev=bfs(positions[j]);chosen=None
            if not blocked and j!=relay and stall<12:
                # Seek different nearby unobserved locations, when connectivity permits.
                for target in [k for k in prev if k in remaining and k!=positions[j]][:6]:
                    local=path(prev,target)
                    if 1<len(local)<=18 and can_move(j,local[1]):chosen=local[1];break
            if chosen is None:
                follow=path(prev,positions[leader])
                if len(follow)>1 and (blocked or stall>=12 or len(follow)>10):
                    if stall>=12:chosen=radio_safe_next(j,positions[leader])
                    if chosen is None and can_move(j,follow[1]):chosen=follow[1]
            if chosen is not None:move(j,chosen,phase,relay);moved=True
        if not moved:
            # Move downstream followers first when an articulation blocks a handoff.
            for j in reversed([(leader+i)%4 for i in range(1,4)]):
                p=path(bfs(positions[j]),positions[leader])
                if len(p)>1 and can_move(j,p[1]):move(j,p[1],phase,relay);moved=True
        if not moved:
            # At a narrow corner, a coordinated move can preserve the links
            # even when neither endpoint may move a full grid edge alone.
            from itertools import combinations
            options={}
            if plans[leader]:options[leader]=plans[leader][0]
            for j in range(4):
                if j!=leader:
                    p=path(bfs(positions[j]),positions[leader])
                    if len(p)>1:options[j]=p[1]
            for count in range(len(options),1,-1):
                if moved:break
                for group in combinations(options,count):
                    old=[nodes[i] for i in positions];proposal=[]
                    for k in range(1,11):
                        current=old.copy()
                        for j in group:
                            a=old[j];b=nodes[options[j]]
                            current[j]=(a[0]+(b[0]-a[0])*k//10,a[1]+(b[1]-a[1])*k//10)
                        if not connected(q_values(current)):break
                        proposal.append(current)
                    if len(proposal)==10:
                        for current in proposal:frames.append({'p':current,'q':q_values(current),'phase':phase,'c':round(coverage(),5),'relay':relay})
                        for j in group:
                            positions[j]=options[j]
                            if j==leader:plans[leader].popleft()
                        sense([nodes[i] for i in positions],len(frames)-1);frames[-1]['c']=round(coverage(),5)
                        moved=True;break
        stall=stall+1 if coverage()<=last_cov else 0;last_cov=coverage()
        if step%50==0:print('step',step,'sensor coverage',round(cov,4),'frames',len(frames),'stall',stall,flush=True)
        if not moved and stall<=250:
            # A different front may need to advance first to unlock a bridge.
            old_leader=leader
            for offset in range(1,5):
                j=(old_leader+offset)%4;prev=bfs(positions[j])
                target=next((k for k in prev if k in remaining and k!=positions[j]),None)
                if target is not None:
                    leader=j;plans=[deque() for _ in range(4)];plans[j]=deque(path(prev,target)[1:]);break
        if stall>250:
            print('Planner stalled; do not claim completion', 'positions',[nodes[i] for i in positions],'leader',leader,'target',list(plans[leader])[:2],flush=True);break
    final_cov=coverage();assert final_cov>=99,('Did not meet task criterion',final_cov)
    # Validate the complete artifact rather than relying only on planning gates.
    for f in frames:assert connected(f['q'])
    for a,b in zip(frames,frames[1:]):
        for j,(p,q) in enumerate(zip(a['p'],b['p'])):
            assert max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=1
            assert ground[q[1],q[0]]==255
            travel[j]+=float(np.linalg.norm(np.array(q)-p))
    for _ in range(100):frames.append(dict(frames[-1]))
    frames[-1]['c']=round(final_cov,5)
    rgb=np.zeros((500,500,3),dtype=np.uint8);rgb[:]=[109,129,155];rgb[ground==255]=[33,51,72];rgb[raw==208]=[255,220,108]
    bg=Image.fromarray(rgb);bg.save(OUT/'map.png')
    completion=bg.copy();draw=ImageDraw.Draw(completion)
    for j,color in enumerate(COLORS):draw.line([tuple(f['p'][j]) for f in frames],fill=color,width=1)
    unseen=(ground==255)&(shared!=255);arr=np.array(completion);arr[unseen]=[255,95,100];completion=Image.fromarray(arr)
    completion.save(OUT/'completed.png')
    data={'frames':frames,'events':events,'phases':phase_defs,'best':opening['best'],'coverage':final_cov,'free_pixels':total_free,
          'observed_free_pixels':int(np.sum(shared==255)),'unseen_free_pixels':int(np.sum(unseen)),
          'travel':travel,'source':'DungeonMaps/test/hybrid/1.png','complete':final_cov>=99}
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'))
    (OUT/'data.json').write_text(payload,encoding='utf-8')
    template=(ROOT/'docs/hybrid1_full_template.html').read_text().replace('__DATA__',payload)
    template=template.replace('__BG__','data:image/png;base64,'+base64.b64encode((OUT/'map.png').read_bytes()).decode())
    (OUT/'index.html').write_text(template,encoding='utf-8')
    print('PASS full run:',len(frames),'frames; connected throughout; sensor coverage',final_cov,'unseen free',int(np.sum(unseen)),flush=True)


if __name__=='__main__':main()
