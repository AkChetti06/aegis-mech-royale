"""Aegis Survivor v3 — defensive navigation and threat-aware Mech Royale agent.

Standalone standard-library agent for mechfight/1. Designed for placement against
coordinated house bots. No network access, model weights or extra dependencies.
"""
import heapq
import json
import math
import os
import random

import mechfight as mf

PI = math.pi
TAU = 2.0 * PI
EPS = 1e-7


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(x, y):
    d = math.hypot(x, y)
    return (x/d, y/d) if d > EPS else (0.0, 0.0)


def yawdiff(a, b):
    return (a-b+PI) % TAU - PI


def toward(a, b):
    return unit(b[0]-a[0], b[1]-a[1])


def step_aim(origin, target, velocity, speed=0.0, lead=0.0):
    ox, oy, oz = origin
    dx, dy, dz = target[0]-ox, target[1]-oy, target[2]-oz
    vx, vy, vz = velocity
    flight = 0.0
    if speed > 1:
        aa = vx*vx + vy*vy + vz*vz - speed*speed
        bb = 2*(dx*vx + dy*vy + dz*vz)
        cc = dx*dx + dy*dy + dz*dz
        disc = bb*bb - 4*aa*cc
        if disc >= 0 and abs(aa) > EPS:
            root = math.sqrt(disc)
            choices = [v for v in ((-bb-root)/(2*aa), (-bb+root)/(2*aa)) if v>0]
            if choices: flight = min(choices)
        if not flight: flight = math.sqrt(cc)/speed
    lead = clamp(flight+lead, 0, 1.65)
    dx += vx*lead; dy += vy*lead; dz += vz*lead
    return math.atan2(dy, dx), math.atan2(dz, max(EPS, math.hypot(dx,dy)))


def mortar_aim(origin, target, velocity, speed, gravity):
    dx = target[0]-origin[0]; dy=target[1]-origin[1]
    flight = max(0.08, math.hypot(dx,dy)/max(speed*0.8, 1))
    yaw = math.atan2(dy,dx); pitch=0.0
    for _ in range(5):
        xx = target[0]+velocity[0]*flight-origin[0]
        yy = target[1]+velocity[1]*flight-origin[1]
        zz = target[2]+velocity[2]*flight-origin[2]
        horiz = max(EPS, math.hypot(xx,yy))
        v2 = speed*speed
        disc = v2*v2 - gravity*(gravity*horiz*horiz+2*zz*v2)
        if disc<0: return None
        pitch = math.atan((v2-math.sqrt(disc))/max(EPS,gravity*horiz))
        flight = horiz/max(EPS, speed*math.cos(pitch))
        yaw = math.atan2(yy,xx)
        if flight>3.5: return None
    return yaw,pitch


class MapIndex:
    """Use actual map geometry for building collision, cover and escape routes."""
    def __init__(self, directory):
        self.cell=30.0
        self.features=[]
        self.grid={}
        self.bounds=(-550.,-380.,550.,380.)
        if not directory: return
        try:
            with open(os.path.join(directory,'v2','world.json'),encoding='utf-8') as fh:
                world=json.load(fh)
            terrain=world.get('terrain',{})
            nx=terrain.get('nx',1070);ny=terrain.get('ny',724)
            tx=terrain.get('minX',-535);ty=terrain.get('minY',-362)
            step=terrain.get('step',1)
            self.bounds=(float(tx),float(ty),float(tx+nx*step),float(ty+ny*step))
            for p in world.get('prisms',[]):
                ring=p.get('ring',[])
                if len(ring)<6 or len(ring)%2: continue
                poly=tuple((float(ring[j]),float(ring[j+1])) for j in range(0,len(ring),2))
                bb=(min(x for x,y in poly),min(y for x,y in poly),
                    max(x for x,y in poly),max(y for x,y in poly))
                holes=[]
                for hole in p.get('holes',[]):
                    if len(hole)>=6 and len(hole)%2==0:
                        holes.append(tuple((float(hole[j]),float(hole[j+1])) for j in range(0,len(hole),2)))
                idx=len(self.features)
                self.features.append((poly,tuple(holes),float(p.get('base',-100)),float(p.get('top',100)),bb))
                for i in range(math.floor((bb[0]-5)/self.cell),math.floor((bb[2]+5)/self.cell)+1):
                    for j in range(math.floor((bb[1]-5)/self.cell),math.floor((bb[3]+5)/self.cell)+1):
                        self.grid.setdefault((i,j),[]).append(idx)
        except (OSError,ValueError,TypeError,KeyError):
            self.features=[];self.grid={}

    @staticmethod
    def inside(poly,x,y):
        hit=False
        px,py=poly[-1]
        for qx,qy in poly:
            if (py>y)!=(qy>y) and x < px+(qx-px)*(y-py)/(qy-py):hit=not hit
            px,py=qx,qy
        return hit

    @staticmethod
    def touching(poly,x,y,r):
        px,py=poly[-1]
        for qx,qy in poly:
            vx=qx-px;vy=qy-py
            f=clamp(((x-px)*vx+(y-py)*vy)/max(EPS,vx*vx+vy*vy),0,1)
            if (x-px-f*vx)**2+(y-py-f*vy)**2 < r*r:return True
            px,py=qx,qy
        return False

    def blocked(self,x,y,z,r=3.4):
        if not self.features:return False
        if x<self.bounds[0]+5 or x>self.bounds[2]-5 or y<self.bounds[1]+5 or y>self.bounds[3]-5:return True
        for idx in self.grid.get((math.floor(x/self.cell),math.floor(y/self.cell)),()):
            poly,holes,base,top,bb=self.features[idx]
            if z+8 <= base or z+1 >= top: continue
            if x<bb[0]-r or x>bb[2]+r or y<bb[1]-r or y>bb[3]+r:continue
            if not (self.inside(poly,x,y) or self.touching(poly,x,y,r)):continue
            if any(self.inside(h,x,y) and not self.touching(h,x,y,r) for h in holes):continue
            return True
        return False

    def clear_line(self,a,b,z,r=3.3):
        if not self.features:return True
        d=dist(a,b)
        for k in range(1,max(1,math.ceil(d/7))+1):
            f=k/max(1,math.ceil(d/7))
            if self.blocked(a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,z,r):return False
        return True

    def cover_between(self,enemy,point,z):
        if not self.features:return False
        for k in (0.25,0.4,0.55,0.7,0.85):
            x=point[0]*(1-k)+enemy[0]*k
            y=point[1]*(1-k)+enemy[1]*k
            if self.blocked(x,y,z,1.0):return True
        return False

    def astar(self,start,end,z,zone,max_nodes=1300):
        """Short A* around walls; limited work per decision to protect 50 ms."""
        cell=12.0
        s=(round(start[0]/cell),round(start[1]/cell))
        g=(round(end[0]/cell),round(end[1]/cell))
        if s==g:return []
        c=zone.get('center',(0,0));r=float(zone.get('radius',600))
        # Route can briefly hug the edge, but should not head far into radiation.
        def free(node):
            x=node[0]*cell;y=node[1]*cell
            if dist((x,y),c)>r+7:return False
            return not self.blocked(x,y,z)
        Q=[(dist(s,g),0,s)]
        cost={s:0.0};parent={}
        seen=set();count=0
        neigh=((1,0,1),(-1,0,1),(0,1,1),(0,-1,1),(1,1,1.42),(1,-1,1.42),(-1,1,1.42),(-1,-1,1.42))
        best=s;best_h=dist(s,g)
        while Q and count<max_nodes:
            _,_,node=heapq.heappop(Q)
            if node in seen:continue
            seen.add(node);count+=1
            h=dist(node,g)
            if h<best_h:best=node;best_h=h
            if h<1.05:best=node;break
            for dx,dy,w in neigh:
                nxt=(node[0]+dx,node[1]+dy)
                if nxt in seen or not free(nxt):continue
                ng=cost[node]+w
                if ng>=cost.get(nxt,1e12):continue
                cost[nxt]=ng;parent[nxt]=node
                heapq.heappush(Q,(ng+dist(nxt,g),count,nxt))
        if best==s:return []
        path=[]
        while best!=s and best in parent:
            path.append((best[0]*cell,best[1]*cell));best=parent[best]
        path.reverse()
        return path


class Aegis:
    name="Aegis-v3-Runner"

    def __init__(self,hello):
        cfg=hello.get('config',{})
        self.config=cfg
        self.map=MapIndex(hello.get('map',{}).get('dir',''))
        self.rng=random.Random(int(hello.get('seed',1234)))
        self.pickup=float(cfg.get('loot',{}).get('pickup_radius',4))
        self.tracks={}
        self.sounds=[]
        self.damage_bearing=None
        self.damage_time=-999.
        self.goal=None
        self.goal_until=-1.
        self.route=[]
        self.last_plan=-999.
        self.last_goal=None
        self.last_pos=None
        self.last_progress=0.
        self.stuck=0
        self.last_jump=-999.
        self.jump_z=0.
        self.last_scan=0.
        self.last_target_id=None

    def upgrades(self,offer):
        def weight(choice):
            k=(str(choice.get('id',''))+' '+str(choice.get('desc',''))).lower()
            pref=(('armor',13),('regen',12),('damage taken',13),('walk_speed',12),
                  ('walk speed',12),('heal',11),('boost',10.5),('energy',10),
                  ('health',9),('speed',9),('gun_rate',7),('laser',7),('mortar',6),
                  ('gun',5),('damage',5),('sensor',3))
            return max([w for word,w in pref if word in k]+[1.])
        ch=offer.get('choices',[])
        return max(range(len(ch)),key=lambda i:weight(ch[i])) if ch else 0

    def remember(self,obs):
        t=float(obs.get('t',0))
        for m in obs.get('mechs',[]):
            if m.get('kind')!='mech':continue
            self.tracks[m['id']]=(tuple(m['pos']),tuple(m.get('vel',[0,0,0])),t)
        for idx,(_,_,seen) in list(self.tracks.items()):
            if t-seen>6:del self.tracks[idx]
        for ev in obs.get('events',[]):
            if ev.get('type')=='damaged' and ev.get('bearing') is not None:
                self.damage_bearing=float(ev['bearing']);self.damage_time=t
        ss=[]
        for s in obs.get('sounds',[]):
            if s.get('kind') in ('death','drop','crate','step','reload'):continue
            if s.get('bearing') is None:continue
            if s.get('dist')=='near':d=48.
            elif s.get('dist')=='mid':d=110.
            else:d=205.
            ss.append((float(s['bearing']),d,t))
        if ss:
            self.sounds=(ss+self.sounds)[:20]
        self.sounds=[x for x in self.sounds if t-x[2]<1.0]

    def threats(self,obs):
        t=float(obs.get('t',0))
        points=[]
        for id,(p,v,seen) in self.tracks.items():
            delta=clamp(t-seen,0,3)
            points.append((p[0]+v[0]*delta,p[1]+v[1]*delta,max(0.3,1.0-delta/7)))
        me=obs['self'];sx,sy=me['pos'][:2]
        for bearing,d,heard in self.sounds:
            if any(math.hypot(p[0]-sx,p[1]-sy)<65 for p in points):continue
            strength=0.4 if d<60 else 0.22
            points.append((sx+math.cos(bearing)*d,sy+math.sin(bearing)*d,strength))
        return points

    def nearest(self,pos,threats):
        return min((dist(pos,p[:2]) for p in threats if p[2]>0.35),default=1e6)

    def is_safe(self,p,zone,margin=0):
        c=zone.get('center',(0,0));r=float(zone.get('radius',560))
        return dist(p,c)<max(3,r-margin)

    def destination(self,obs,threats):
        """Find a survivable point in the *next* circle away from the squad."""
        me=obs['self'];pos=me['pos'];x,y,z=pos
        zone=obs['zone'];now=float(obs.get('t',0))
        curc=zone.get('center',(0,0));curR=max(1.,float(zone.get('radius',560)))
        nc=zone.get('next_center',curc);nr=max(0.,float(zone.get('next_radius',curR)))
        curmargin=curR-dist(pos,curc);nextmargin=nr-dist(pos,nc)
        nearest=self.nearest(pos,threats)
        until_start=float(zone.get('shrink_start',1e6))-now
        # If early and comfortably inside both circles, preserve heal time and energy.
        quiet=(nearest>150 and not obs.get('projectiles') and len(self.sounds)==0)
        health=me.get('hp',0)/max(1,me.get('max_hp',1))
        if quiet and nextmargin>65 and (health<0.93 or until_start>40):
            return tuple(pos[:2]),'quiet',curmargin,nextmargin,nearest
        if (self.goal and now<self.goal_until and curmargin>12 and nextmargin>15
                and dist(self.goal,nc)<max(5,nr-12)
                and self.is_safe(self.goal,zone,10)):
            if nearest>110 or (self.nearest(self.goal,threats)>nearest+25):
                return self.goal,'hold',curmargin,nextmargin,nearest
        # Candidate goals: center and safe interior ring. Best point is away from
        # active hunters, but not so close to the edge that it will be zoned out.
        c=nc;radius=clamp(nr*0.65,0,max(0,nr-22))
        points=[(c[0],c[1])]
        if nextmargin>28:points.append((x,y))
        for ring in (radius,radius*0.48):
            for k in range(16):
                a=k*TAU/16
                points.append((c[0]+ring*math.cos(a),c[1]+ring*math.sin(a)))
        best=None;best_score=-1e12
        for p in points:
            if not self.is_safe(p,zone,5) or self.map.blocked(p[0],p[1],z,4):continue
            enemy_cost=0.0
            if threats:
                # Squad proximity is nonlinear: one pursuer is manageable,
                # multiple nearby pursuers are catastrophic.
                nearby=sorted((dist(p,q[:2])/max(0.3,q[2]) for q in threats))[:4]
                enemy_cost=sum(max(0,175-d)**2/360 for d in nearby)
            travel=dist(pos,p)
            cover=0
            if threats and self.map.features:
                for q in sorted(threats,key=lambda q:dist(q,pos))[:3]:
                    if self.map.cover_between(q,p,z):cover+=9
            # Benefit of keeping a large margin against late storm shifts.
            margin=nr-dist(p,nc)
            score=-enemy_cost -0.17*travel + min(margin,70)*0.10+cover
            if curmargin<0:score-=0.3*travel
            if score>best_score:best_score=score;best=p
        if best is None:best=tuple(nc)
        self.goal=best
        self.goal_until=now+(2 if nearest<110 else 4)
        return best,'planned',curmargin,nextmargin,nearest

    def objective(self,obs,threats,curmargin,nextmargin,nearest):
        me=obs['self'];pos=me['pos'];z=pos[2];hp=me.get('hp',0)/max(1,me.get('max_hp',1))
        now=float(obs.get('t',0));zone=obs['zone'];nc=zone.get('next_center',(0,0));nr=float(zone.get('next_radius',500))
        # No farming while hunted; health regen and escape have priority.
        if nearest<145 or hp<0.80 or nextmargin<22:return None
        options=[]
        for c in obs.get('cubes',[]):
            d=dist(pos,c['pos'])
            if d<85 and abs(c['pos'][2]-z)<4 and dist(c['pos'],nc)<nr-15:
                options.append((80-.7*d,c['pos'][:2],'cube'))
        for p in obs.get('pickups',[]):
            if not any(w is None for w in me.get('weapons',[])):continue
            d=dist(pos,p['pos']);v={'laser':50,'mortar':48,'lightning':38}.get(p.get('kind'),0)
            if d<50 and abs(p['pos'][2]-z)<4 and dist(p['pos'],nc)<nr-12:
                options.append((v-.6*d,p['pos'][:2],'weapon'))
        # A free capture heals fully; don't fight a squad at a contested drop.
        for p in obs.get('drops',[]):
            d=dist(pos,p['pos'])
            if p.get('holder') not in (None,me.get('id')):continue
            if d<75 and float(p.get('lands_in',999))<16 and dist(p['pos'],nc)<nr-12:
                options.append((110-.65*d,p['pos'][:2],'drop'))
        if not options:return None
        score,p,kind=max(options)
        return tuple(p) if score>10 else None

    def select_target(self,obs,nearest):
        me=obs['self'];pos=me['pos']
        enemies=[m for m in obs.get('mechs',[]) if m.get('kind')=='mech']
        if enemies:
            # Face nearest threat to keep front armor active, even when fleeing.
            target=min(enemies,key=lambda m:dist(m['pos'],pos)-(18 if m.get('charging') else 0))
            return target,True
        if nearest<140:return None,False
        hp=me.get('hp',0)/max(1,me.get('max_hp',1))
        if hp<0.86:return None,False
        cubes=int(me.get('cubes',0))
        if cubes>5:return None,False
        targets=[]
        for m in obs.get('mechs',[]):
            if m.get('kind')=='npc' and dist(m['pos'],pos)<100:
                targets.append((dist(m['pos'],pos)*0.8 + 0.015*m.get('hp',350),m))
        for c in obs.get('crates',[]):
            if dist(c['pos'],pos)<105:
                m={'id':c.get('id'), 'kind':'crate','pos':c['pos'],'vel':[0,0,0]}
                targets.append((dist(c['pos'],pos)+22,m))
        return (min(targets,key=lambda t:t[0])[1],False) if targets else (None,False)

    def fight(self,obs,target,is_enemy,nearest):
        me=obs['self'];weapons=me.get('weapons',[]);pos=me['pos']
        fire=[False,False];reload=False
        origin=[pos[0],pos[1],pos[2]+me.get('stats',{}).get('muzzle_height',7)]
        actual=me.get('aim',[0,0]);aim=[actual[0],actual[1]]
        if target is None:
            if self.damage_bearing is not None and obs.get('t',0)-self.damage_time<4:
                aim=[self.damage_bearing,0]
            else:
                aim=[actual[0]+0.055,0]
        else:
            tpos=target['pos'];tvel=target.get('vel',[0,0,0]);kind=target.get('kind','mech')
            d=dist(pos,tpos)
            if kind=='mech':height=4.5*(1+0.05*target.get('cubes',0))
            elif kind=='npc':height=1.8
            else:height=1.7
            dest=(tpos[0],tpos[1],tpos[2]+height)
            gun=next(((i,w) for i,w in enumerate(weapons[:2]) if w and w.get('kind')=='gun'),None)
            laser=next(((i,w) for i,w in enumerate(weapons[:2]) if w and w.get('kind')=='laser'),None)
            mortar=next(((i,w) for i,w in enumerate(weapons[:2]) if w and w.get('kind')=='mortar'),None)
            lightning=next(((i,w) for i,w in enumerate(weapons[:2]) if w and w.get('kind')=='lightning'),None)
            # Avoid laser charge's 50% movement penalty when a squad is closing.
            laser_ok=laser and nearest>115 and d<245
            if laser and laser[1].get('charge',0)>0:laser_ok=True
            mortar_ok=mortar and 48<d<180 and nearest>75
            lightning_ok=lightning and d<40 and me.get('energy',0)>8
            gun_speed=float((gun[1].get('stats',{}).get('speed',252) if gun else 252))
            aim=step_aim(origin,dest,tvel,gun_speed,lead=0.08)
            if laser_ok:
                aim=step_aim(origin,dest,tvel,lead=0.08)
            elif mortar_ok:
                m=mortar_aim(origin,dest,tvel,
                             float(mortar[1].get('stats',{}).get('speed',72)),
                             float(mortar[1].get('stats',{}).get('gravity',22)))
                if m is not None:aim=m
                else:mortar_ok=False
            yawerr=abs(yawdiff(aim[0],actual[0]))
            pitcherr=abs(aim[1]-actual[1])
            aligned=yawerr<0.14 and pitcherr<0.16
            # House opponents share vision. Distant harassment only reveals us and
            # prevents passive HP regeneration, so shoot mechs to defend or finish.
            finish=is_enemy and target.get('hp',1e9)<270 and nearest<145
            safe_fire=(is_enemy and (nearest<90 or finish)) or not is_enemy
            if gun:
                i,w=gun
                ammo=w.get('ammo',12)
                if ammo < 9 and w.get('reload',0)<=0 and (not safe_fire or d>130):reload=True
                if ammo<=0 and w.get('reload',0)<=0:reload=True
                if safe_fire and not laser_ok and not mortar_ok and d < float(w.get('stats',{}).get('range',230))-6 and aligned:
                    fire[i]=True
            if laser:
                i,w=laser
                charge=float(w.get('charge',0));goal=float(w.get('stats',{}).get('charge',0.75))
                if laser_ok and safe_fire and d<float(w.get('stats',{}).get('range',280))-5:
                    fire[i]=(charge<max(0.18,goal-0.08) or not aligned)
                # A charging laser should be deliberately released before a retreat.
                elif charge>0:
                    fire[i]=False
            if mortar_ok and mortar and safe_fire and aligned:
                fire[mortar[0]]=True
            if lightning_ok and lightning and is_enemy and yawerr<0.30:
                fire[lightning[0]]=True
            # Other slots not firing: don't spray at mismatched aim angles.
        aim=[float(aim[0]),clamp(float(aim[1]),-1.35,1.35)]
        return aim,fire,reload

    def projectile_danger(self,obs,dirxy=(0.,0.)):
        pos=obs['self']['pos'];vel=obs['self'].get('vel',[0,0,0]);
        ownvx=vel[0]*0.55+dirxy[0]*8.0;ownvy=vel[1]*0.55+dirxy[1]*8.0
        danger=0.
        for p in obs.get('projectiles',[]):
            pp=p.get('pos',(0,0,0));pv=p.get('vel',(0,0,0))
            if len(pp)<3 or len(pv)<3:continue
            rx=pp[0]-pos[0];ry=pp[1]-pos[1]
            vx=pv[0]-ownvx;vy=pv[1]-ownvy
            ss=max(EPS,vx*vx+vy*vy)
            t=clamp(-(rx*vx+ry*vy)/ss,0,0.80)
            dx=rx+vx*t;dy=ry+vy*t
            z=pp[2]+pv[2]*t-(11*t*t if p.get('kind')=='mortar' else 0)
            if abs(z-pos[2]-4.5)>9:continue
            closest=math.hypot(dx,dy)
            if closest<17:
                w=(17-closest)**2/25 *(1.0-t/1.5)
                danger+=w*(2.2 if p.get('kind')=='mortar' else 1)
        return danger

    def steer(self,obs,dest,threats,nearest,curmargin,nextmargin):
        me=obs['self'];pos=me['pos'];x,y,z=pos
        zinfo=obs['zone'];now=float(obs.get('t',0))
        curc=zinfo.get('center',(0,0));nc=zinfo.get('next_center',curc)
        nr=float(zinfo.get('next_radius',0))
        # Compensate for movement delay and 0.1s decision interval.
        wx,wy=toward(pos,dest)
        if self.map.features and not self.map.clear_line(pos,dest,z) and dist(pos,dest)>18:
            if (not self.route or now-self.last_plan>3 or
                    (self.last_goal and dist(dest,self.last_goal)>40)):
                self.route=self.map.astar(pos,dest,z,zinfo)
                self.last_goal=dest;self.last_plan=now
            while self.route and dist(pos,self.route[0])<10:self.route.pop(0)
            if self.route:
                wx,wy=toward(pos,self.route[0])
        else:self.route=[]
        # Candidate movement: choose one of 16 headings, plus standing still.
        dirs=[(math.cos(a*TAU/16),math.sin(a*TAU/16)) for a in range(16)]
        dirs.append((0.,0.))
        best=(wx,wy);bestscore=-1e12
        close=nearest<105
        for dx,dy in dirs:
            px=x+dx*15;py=y+dy*15
            if self.map.blocked(px,py,z):continue
            # Additional short collision check prevents steering into wall faces.
            if self.map.blocked(x+dx*7,y+dy*7,z):continue
            post=(px,py)
            threat_pen=0.0
            for ex,ey,w in threats:
                d=dist(post,(ex,ey))
                if d<160:threat_pen += w*(max(0,160-d)/38)**2
            margin=float(zinfo.get('radius',550))-dist(post,curc)
            future=nr-dist(post,nc)
            storm_pen=15*max(0,10-margin)+4.5*max(0,16-future)
            toward_goal=(dist(pos,dest)-dist(post,dest))
            bullet=self.projectile_danger(obs,(dx,dy))
            score=0.25*toward_goal - threat_pen*4.2 - storm_pen - bullet*3.5
            # If there is no threat, don't cross the map for a marginally better path.
            if not close and dist(pos,dest)<5 and dx==dy==0:score+=4
            if close:
                # Running laterally confuses gun prediction; avoid standing still.
                if dx==dy==0:score-=7
                # Favor separation from a chasing squad and break line of sight.
                if threats:
                    nearest_after=min(dist(post,q[:2]) for q in threats)
                    score+=0.22*(nearest_after-nearest)
                if self.map.features:
                    score+=sum(2 for q in threats[:3] if self.map.cover_between(q,post,z))
            # Stability prevents jitter when direction is almost as good.
            v=me.get('vel',[0,0,0]);ux,uy=unit(v[0],v[1])
            score+=0.35*(dx*ux+dy*uy)
            if score>bestscore:bestscore=score;best=dx,dy
        # At home or healing, avoid revealing ourselves with footfalls.
        if nearest>160 and dist(pos,dest)<12 and nextmargin>25:
            best=(0.,0.)
        return best

    def act(self,obs):
        me=obs['self'];pos=me['pos'];now=float(obs.get('t',0))
        self.remember(obs)
        threats=self.threats(obs)
        goal,mode,curmargin,nextmargin,nearest=self.destination(obs,threats)
        objective=self.objective(obs,threats,curmargin,nextmargin,nearest)
        if objective is not None and nearest>150:goal=objective
        target,is_enemy=self.select_target(obs,nearest)
        aim,fire,reload=self.fight(obs,target,is_enemy,nearest)
        move=self.steer(obs,goal,threats,nearest,curmargin,nextmargin)
        energy=float(me.get('energy',0))
        evdanger=self.projectile_danger(obs,move)
        hurt=any(e.get('type')=='damaged' for e in obs.get('events',[]))
        danger=nearest<98 or evdanger>1.2 or hurt
        zone=obs['zone']
        time_to_shrink=float(zone.get('shrink_start',1e6))-now
        move_time=max(0,-nextmargin)/max(4.0,float(me.get('stats',{}).get('walk_speed',11.7)))
        storm_urgent=(curmargin<4 or (nextmargin<-10 and time_to_shrink<move_time+10))
        boost=bool((storm_urgent or danger) and energy>(12 if curmargin<0 else 24) and math.hypot(*move)>0.7)
        jump=False
        if self.last_pos is not None and now-self.last_progress>0.9:
            if dist(pos,self.last_pos)<2.0 and math.hypot(*move)>0.75:
                self.stuck+=1
            else:self.stuck=max(0,self.stuck-1)
        if now-self.last_progress>0.9:
            self.last_pos=tuple(pos[:2]);self.last_progress=now
        if me.get('grounded',True) and now-self.last_jump>4.0 and energy>42:
            if (self.stuck>=2 or (nearest<52 and evdanger>0.3)):
                jump=True;self.last_jump=now;self.jump_z=pos[2]
        if not me.get('grounded',True) and self.last_jump>0 and now-self.last_jump<2.0 and energy>35:
            if self.stuck>=2 or nearest<100:
                if pos[2]-self.jump_z < 26:
                    boost=True
        # A pickup can be taken with no detour if already in arm's reach.
        act={'tick':obs['tick'],'move':list(move),'boost':boost,'jump':jump,'aim':aim,'fire':fire,'reload':reload}
        if obs.get('offer'):
            act['pick']=self.upgrades(obs['offer'])
        weapons=me.get('weapons',[])
        owned={w.get('kind') for w in weapons if w}
        picks=sorted(obs.get('pickups',[]),key=lambda p:dist(p['pos'],pos))
        for p in picks:
            d=dist(p['pos'],pos)
            if d>max(0,self.pickup-0.3):break
            if abs(p['pos'][2]-pos[2])>4 or p.get('kind') in owned:continue
            empty=next((i for i,w in enumerate(weapons) if w is None),None)
            if empty is not None:
                act['take']=p['id'];break
            i=next((i for i,w in enumerate(weapons) if w and w.get('kind')=='gun'),None)
            if i is not None and p.get('kind') in ('laser','mortar'):
                act['take']=p['id'];act['drop']=i;break
        return act



class AegisV4(Aegis):
    """V4: objectives before first contact, quicker evasions, fewer wasted lives.

    Maintains v3 navigation fallback and official mechfight/1 protocol. All
    movement and weapon decisions are local and deterministic per observation.
    """
    name = "Aegis-v4-LastStand"

    def __init__(self, hello):
        super().__init__(hello)
        self.hide_goal = None
        self.hide_until = -1.0
        self.last_hp = None
        self.under_fire_until = -1.0
        self.last_boost_until = -1.0
        self.farm_goal = None
        self.farm_until = -1.0

    def cover_spot(self, pos, obs, threats):
        """Find a reachable point close to walls which masks long sightlines."""
        if not self.map.features:
            return None
        zone = obs['zone']
        nc = zone.get('next_center', zone.get('center', (0., 0.)))
        nr = float(zone.get('next_radius',zone.get('radius',550)))
        z = pos[2]
        best = None
        best_score = -1.e10
        for poly, holes, base, top, bb in self.map.features:
            if top < z+7 or base > z+8: continue
            cx = (bb[0]+bb[2])*0.5; cy = (bb[1]+bb[3])*0.5
            if dist(pos,(cx,cy)) > 115: continue
            # Stand slightly outside footprint, not in the wall or its holes.
            for vx,vy in (poly[::max(1,len(poly)//12)]):
                for ax,ay in ((vx-cx,vy-cy),(cx-vx,cy-vy)):
                    ux,uy=unit(ax,ay)
                    pt=(vx+ux*7,vy+uy*7)
                    if self.map.blocked(pt[0],pt[1],z,3.3):continue
                    if dist(pt,nc) > nr-28:continue
                    d=dist(pt,pos)
                    if d > 105 or d < 12:continue
                    if not self.is_safe(pt,zone,20):continue
                    # Prefer a wall *between us and known threats* and low travel cost.
                    shield=sum(22*q[2] for q in threats[:5] if self.map.cover_between(q,pt,z))
                    margin=min(60,nr-dist(pt,nc))
                    score=shield + 0.09*margin-0.19*d
                    if score>best_score:best=pt;best_score=score
        return best

    def objective(self, obs, threats, curmargin, nextmargin, nearest):
        me=obs['self'];pos=me['pos'];zone=obs['zone'];t=float(obs.get('t',0.))
        hp=me['hp']/max(1.,me['max_hp'])
        if nearest<165 or hp<0.78 or curmargin<25 or nextmargin<28:
            self.farm_goal=None
            return None
        nc=zone.get('next_center',zone.get('center',(0,0)))
        nr=float(zone.get('next_radius',500))
        # Loose cubes are the cheapest permanent survival upgrade (+max HP).
        candidates=[]
        for c in obs.get('cubes',()):
            p=c['pos'];d=dist(pos,p)
            if d>125 or abs(pos[2]-p[2])>4 or dist(p,nc)>nr-20:continue
            score=125-0.82*d+(45 if me.get('cubes',0)<3 else 0)
            candidates.append((score,tuple(p[:2])))
        for p in obs.get('pickups',()):
            if not any(w is None for w in me.get('weapons',())):break
            d=dist(pos,p['pos'])
            if d>90 or abs(pos[2]-p['pos'][2])>4:continue
            if dist(p['pos'],nc)>nr-20:continue
            score={'laser':87,'mortar':81,'lightning':65}.get(p.get('kind'),0)-0.66*d
            candidates.append((score,tuple(p['pos'][:2])))
        for p in obs.get('drops',()):
            d=dist(pos,p['pos'])
            if d>120 or p.get('holder') not in (None,me.get('id')):continue
            if dist(p['pos'],nc)>nr-20:continue
            landing=float(p.get('lands_in',999))
            if landing>18:continue
            score=155-0.73*d-0.50*landing
            candidates.append((score,tuple(p['pos'][:2])))
        # Unlike v3, APPROACH accessible crates/drones so shots yield cubes.
        # Only farm early and when we can afford to attract their attention.
        if t<150 and me.get('cubes',0)<7 and hp>0.88 and nearest>205:
            for m in obs.get('mechs',()):
                if m.get('kind')!='npc':continue
                p=m['pos'];d=dist(pos,p)
                if d>95 or dist(p,nc)>nr-23:continue
                candidates.append((75-0.50*d,tuple(p[:2])))
            for c in obs.get('crates',()):
                p=c['pos'];d=dist(pos,p)
                if d>105 or dist(p,nc)>nr-23 or abs(p[2]-pos[2])>7:continue
                candidates.append((56-0.50*d,tuple(p[:2])))
        if not candidates:return None
        best_score,best=max(candidates)
        if best_score<9:return None
        # Goal persistence avoids changing direction as noisy observations appear.
        if (self.farm_goal is not None and t<self.farm_until and
            dist(self.farm_goal,pos)>7 and
            dist(self.farm_goal,nc)<nr-20 and
            best_score<65):return self.farm_goal
        self.farm_goal=best;self.farm_until=t+1.7
        return best

    def destination(self,obs,threats):
        goal,mode,curmargin,nextmargin,nearest=super().destination(obs,threats)
        me=obs['self'];pos=me['pos'];t=float(obs.get('t',0))
        hp=me['hp']/max(1,me['max_hp'])
        # V3 would idle in the open at full health for the entire early hold.
        # Take accessible cover while quiet, provided the route stays well in zone.
        if mode=='quiet' and hp>0.92 and nextmargin>75 and t<190:
            if self.hide_goal and t<self.hide_until and dist(pos,self.hide_goal)>9:
                return self.hide_goal,'cover',curmargin,nextmargin,nearest
            h=self.cover_spot(pos,obs,threats)
            if h is not None and dist(pos,h)>16:
                self.hide_goal=h;self.hide_until=t+14
                return h,'cover',curmargin,nextmargin,nearest
        return goal,mode,curmargin,nextmargin,nearest

    def fight(self,obs,target,is_enemy,nearest):
        aim,fire,reload=super().fight(obs,target,is_enemy,nearest)
        if not target or not is_enemy:return aim,fire,reload
        me=obs['self'];pos=me['pos']
        d=dist(pos,target['pos'])
        hp=me['hp']/max(1.,me['max_hp'])
        # Shooting before contact can stop a pursuer, but reckless long-range
        # firing prevents regen and alerts the coordinated practice squad.
        if d<138 and (nearest<120 or hp>0.85):
            actual=me.get('aim',(0.,0.))
            if abs(yawdiff(actual[0],aim[0]))<0.12 and abs(actual[1]-aim[1])<0.13:
                for i,w in enumerate(me.get('weapons',())[:2]):
                    if w and w.get('kind')=='gun' and w.get('ammo',0)>0 and w.get('reload',0)<=0:
                        fire[i]=True
        return aim,fire,reload

    def act(self,obs):
        me=obs['self'];t=float(obs.get('t',0.));pos=me['pos']
        hp=float(me.get('hp',0.))
        if self.last_hp is not None and hp < self.last_hp-1:
            self.under_fire_until=t+4.5
        self.last_hp=hp
        act=super().act(obs)
        move=act['move'];energy=float(me.get('energy',0.))
        threats=self.threats(obs)
        near=self.nearest(pos,threats)
        pressured=t<self.under_fire_until or near<155
        # Earlier sustained escape bursts. V3 often boosted too late (<98m).
        if (pressured and energy>34 and math.hypot(*move)>.75 and
            (near<112 or t<self.under_fire_until or len(threats)>=3)):
            act['boost']=True
        # Conserve energy for unavoidable late-storm repositioning.
        if not pressured and me.get('in_zone',True) and energy<50:
            act['boost']=False
        if not me.get('in_zone',True) and energy>7 and math.hypot(*move)>0.2:
            act['boost']=True
        # Emergency vertical evasion when pinned under coordinated fire.
        nearby=sum(1 for q in threats if dist(pos,q[:2])<63 and q[2]>.35)
        if nearby>=3 and energy>56 and me.get('grounded',True) and t-self.last_jump>4:
            act['jump']=True;self.last_jump=t;self.jump_z=pos[2]
        if nearby>=2 and not me.get('grounded',True) and energy>33 and t-self.last_jump<1.6:
            act['boost']=True
        return act


mf.run(AegisV4)
