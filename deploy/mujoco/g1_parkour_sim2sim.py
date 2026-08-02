"""MuJoCo sim2sim smoke test for the frozen Instinct G1 parkour policy."""
from __future__ import annotations
import argparse,json,math,re,tempfile,time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import mujoco,numpy as np,onnxruntime as ort,yaml
ROOT=Path(__file__).resolve().parents[2]; POLICY=ROOT/'artifacts/low_level/g1_parkour'; URDF=ROOT/'external/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf'
JOINTS='left_shoulder_pitch_joint right_shoulder_pitch_joint waist_pitch_joint left_shoulder_roll_joint right_shoulder_roll_joint waist_roll_joint left_shoulder_yaw_joint right_shoulder_yaw_joint waist_yaw_joint left_elbow_joint right_elbow_joint left_hip_pitch_joint right_hip_pitch_joint left_wrist_roll_joint right_wrist_roll_joint left_hip_roll_joint right_hip_roll_joint left_wrist_pitch_joint right_wrist_pitch_joint left_hip_yaw_joint right_hip_yaw_joint left_wrist_yaw_joint right_wrist_yaw_joint left_knee_joint right_knee_joint left_ankle_pitch_joint right_ankle_pitch_joint left_ankle_roll_joint right_ankle_roll_joint'.split()
@dataclass(frozen=True)
class TerrainConfig:
 terrain: str = 'flat'
 start_x: float = 1.5
 width: float = 2.0
 stair_height: float = 0.12
 stair_depth: float = 0.50
 stair_steps: int = 4
 ramp_angle_deg: float = 12.0
 ramp_length: float = 3.0
 ramp_thickness: float = 0.10
 terrain_file: Path | None = None

def custom_terrain_geoms_xml(path):
 if path is None:
  raise ValueError('custom terrain requires --terrain_file')
 path=Path(path)
 data=json.loads(path.read_text())
 objects=data.get('objects')
 if not isinstance(objects,list) or not objects:
  raise ValueError('custom terrain file must contain a non-empty objects list')
 geoms=[]
 for index,obj in enumerate(objects):
  if obj.get('type','box') != 'box':
   raise ValueError('custom terrain currently supports box objects only')
  name=str(obj.get('name',f'object_{index}'))
  if re.fullmatch(r'[A-Za-z0-9_-]+',name) is None:
   raise ValueError(f'invalid terrain object name: {name}')
  position=np.asarray(obj['position'],dtype=float)
  size=np.asarray(obj['size'],dtype=float)
  euler=np.asarray(obj.get('rotation_rpy_deg',[0,0,0]),dtype=float)
  color=np.asarray(obj.get('color',[0.42,0.46,0.50,1.0]),dtype=float)
  if position.shape != (3,) or size.shape != (3,) or np.any(size <= 0):
   raise ValueError(f'invalid position or size for terrain object: {name}')
  if euler.shape != (3,) or color.shape != (4,):
   raise ValueError(f'invalid rotation or color for terrain object: {name}')
  geoms.append(
   f'<geom name="terrain_{name}" type="box" pos="{" ".join(map(str,position))}" '
   f'size="{" ".join(map(str,size*0.5))}" euler="{" ".join(map(str,euler))}" '
   f'group="0" rgba="{" ".join(map(str,color))}"/>'
  )
 return ''.join(geoms)

def terrain_geoms_xml(cfg):
 if cfg.terrain == 'flat':
  return ''
 if cfg.terrain == 'custom':
  return custom_terrain_geoms_xml(cfg.terrain_file)
 if cfg.width <= 0:
  raise ValueError('terrain width must be positive')
 if cfg.terrain == 'stairs':
  if cfg.stair_height <= 0 or cfg.stair_depth <= 0 or cfg.stair_steps <= 0:
   raise ValueError('stair dimensions and step count must be positive')
  geoms=[]
  for i in range(cfg.stair_steps):
   top=(i+1)*cfg.stair_height
   x=cfg.start_x+(i+0.5)*cfg.stair_depth
   geoms.append(
    f'<geom name="terrain_stair_{i}" type="box" '
    f'pos="{x:.6f} 0 {top*0.5:.6f}" '
    f'size="{cfg.stair_depth*0.5:.6f} {cfg.width*0.5:.6f} {top*0.5:.6f}" '
    f'group="0" rgba="0.42 0.46 0.50 1"/>'
   )
  return ''.join(geoms)
 if cfg.terrain == 'ramp':
  if cfg.ramp_length <= 0 or cfg.ramp_thickness <= 0:
   raise ValueError('ramp length and thickness must be positive')
  angle=math.radians(cfg.ramp_angle_deg)
  if not 0 < angle < math.radians(45):
   raise ValueError('ramp angle must be between 0 and 45 degrees')
  x=cfg.start_x+0.5*cfg.ramp_length*math.cos(angle)
  z=0.5*cfg.ramp_length*math.sin(angle)+0.5*cfg.ramp_thickness*math.cos(angle)
  return (
   f'<geom name="terrain_ramp" type="box" '
   f'pos="{x:.6f} 0 {z:.6f}" '
   f'size="{cfg.ramp_length*0.5:.6f} {cfg.width*0.5:.6f} {cfg.ramp_thickness*0.5:.6f}" '
   f'euler="0 {-cfg.ramp_angle_deg:.6f} 0" group="0" rgba="0.42 0.46 0.50 1"/>'
  )
 raise ValueError(f'unsupported terrain: {cfg.terrain}')

def terrain_config_from_args(a):
 return TerrainConfig(
  terrain=a.terrain,start_x=a.terrain_start_x,width=a.terrain_width,
  stair_height=a.stair_height,stair_depth=a.stair_depth,stair_steps=a.stair_steps,
  ramp_angle_deg=a.ramp_angle_deg,ramp_length=a.ramp_length,terrain_file=a.terrain_file,
 )
def mapped(names,cfg):
 out=np.zeros(len(names),np.float32)
 for pat,val in cfg.items():
  for i,name in enumerate(names):
   if re.search(pat,name): out[i]=float(val)
 return out
def load_params(path):
 cfg=yaml.unsafe_load((path/'params/env.yaml').read_text()); robot=cfg['scene']['robot']; out={'default':mapped(JOINTS,robot['init_state']['joint_pos']),'scale':mapped(JOINTS,cfg['actions']['joint_pos']['scale'])}
 for field,key in [('stiffness','kp'),('damping','kd'),('effort_limit_sim','effort'),('armature','armature')]:
  arr=np.zeros(29,np.float32)
  for ac in robot['actuators'].values():
   ids=[i for i,n in enumerate(JOINTS) if any(re.search(x,n) for x in ac['joint_names_expr'])]; val=ac[field]
   arr[ids]=mapped([JOINTS[i] for i in ids],val) if isinstance(val,dict) else float(val)
  out[key]=arr
 return out
def build_model(path,dt,terrain_cfg=None):
 terrain_xml=terrain_geoms_xml(terrain_cfg or TerrainConfig())
 if path.suffix.lower() != ".urdf":
  loaded=mujoco.MjModel.from_xml_path(str(path))
  with tempfile.TemporaryDirectory() as td:
   saved=Path(td)/"robot.xml"; mujoco.mj_saveLastXML(str(saved),loaded); xml=saved.read_text()
   xml=re.sub(r'file="([^"]+)"',lambda m:'file="'+str((path.parent/"meshes"/m.group(1)).resolve())+'"',xml)
   xml=xml.replace('<worldbody>','<worldbody><geom name="floor" type="plane" size="20 20 0.1" friction="1 0.005 0.0001" group="0"/>'+terrain_xml,1)
   model=mujoco.MjModel.from_xml_string(xml)
   model.opt.timestep=dt; model.opt.gravity[:]=[0,0,-9.81]; model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
   return model
 # Add the floating joint before URDF import so MuJoCo preserves root-link inertia.
 urdf=path.read_text()
 urdf=re.sub(r'filename="([^"]+)"',lambda m:'filename="'+str((path.parent/m.group(1)).resolve())+'"',urdf)
 floating='<link name="mujoco_world"/><joint name="floating_base_joint" type="floating"><parent link="mujoco_world"/><child link="torso_link"/></joint>'
 urdf=urdf.replace('</robot>',floating+'</robot>')
 with tempfile.TemporaryDirectory() as td:
  source=Path(td)/'floating.urdf'; source.write_text(urdf)
  imported=mujoco.MjModel.from_xml_path(str(source)); mjcf=Path(td)/'floating.xml'; mujoco.mj_saveLastXML(str(mjcf),imported); xml=mjcf.read_text()
  opening=f'''<option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><geom friction="1 0.005 0.0001" condim="3"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="0"/>{terrain_xml}'''
  return mujoco.MjModel.from_xml_string(xml.replace('<worldbody>',opening,1))
def ids(model):
 q,d=[],[]
 for name in JOINTS:
  j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
  if j<0: raise RuntimeError('missing '+name)
  q.append(model.jnt_qposadr[j]); d.append(model.jnt_dofadr[j])
 return np.array(q),np.array(d)
def body_obs(model,data,bid):
 v=np.zeros(6); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,v,1); g=data.xmat[bid].reshape(3,3).T@np.array([0.,0.,-1.]); return v[:3].astype(np.float32),g.astype(np.float32)
def run(a):
 p=load_params(a.policy_dir); model=build_model(a.urdf,a.dt,terrain_config_from_args(a)); data=mujoco.MjData(model); qi,di=ids(model); bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link'); model.dof_armature[di]=p['armature']
 actor=ort.InferenceSession(str(a.policy_dir/'exported/actor.onnx'),providers=['CPUExecutionProvider']); enc=ort.InferenceSession(str(a.policy_dir/'exported/0-depth_encoder.onnx'),providers=['CPUExecutionProvider'])
 data.qpos[:7]=[0,0,a.height,1,0,0,0]; data.qpos[qi]=p['default']; mujoco.mj_forward(model,data); last=np.zeros(29,np.float32); ang,grav=body_obs(model,data,bid); cmd=np.array(a.command,np.float32)
 vals={'ang':ang*.25,'grav':grav,'cmd':np.zeros(3,np.float32),'q':np.zeros(29,np.float32),'dq':np.zeros(29,np.float32),'act':last}; hist={k:deque([v.copy() for _ in range(8)],maxlen=8) for k,v in vals.items()}; depth=np.ones((1,8,18,32),np.float32)
 if a.depth_npz is not None:
  depth_m=np.load(a.depth_npz)['depth_m'].astype(np.float32)
  if depth_m.shape != (36,64): raise ValueError('expected depth_m shape (36,64), got '+str(depth_m.shape))
  frame=np.clip(depth_m[18:36,16:48],0.,2.5)/2.5
  depth=np.repeat(frame[None,None],8,axis=1)
 latent=enc.run(None,{enc.get_inputs()[0].name:depth})[0].reshape(-1); target=p['default'].copy(); nxt=0.; rec=[]; fell=False; first_action=None
 viewer=None
 if a.viewer:
  from mujoco import viewer as mj_viewer
  viewer=mj_viewer.launch_passive(model,data)
 while data.time<a.duration:
  if data.time+1e-9>=nxt:
   ang,grav=body_obs(model,data,bid); active=cmd if data.time>=a.command_start else np.zeros(3,np.float32); vals={'ang':ang*.25,'grav':grav,'cmd':active,'q':data.qpos[qi].astype(np.float32)-p['default'],'dq':data.qvel[di].astype(np.float32)*.05,'act':last}
   for k,v in vals.items(): hist[k].append(v.copy())
   obs=np.concatenate([np.stack(hist[k]).reshape(-1) for k in hist]+[latent]).reshape(1,-1).astype(np.float32)
   if obs.shape!=(1,896): raise RuntimeError('actor obs '+str(obs.shape))
   if not a.disable_policy:
    last=actor.run(None,{actor.get_inputs()[0].name:obs})[0].reshape(-1).astype(np.float32)
    if first_action is None: first_action=last.copy()
    target=p['default']+p['scale']*last
   nxt+=1/a.policy_hz
  tau=np.clip(p['kp']*(target-data.qpos[qi])-p['kd']*data.qvel[di],-p['effort'],p['effort']); data.qfrc_applied[di]=tau; mujoco.mj_step(model,data)
  if viewer is not None:
   viewer.sync()
   time.sleep(a.dt)
  if not rec or data.time-rec[-1][0]>=.02-1e-9:
   _,g=body_obs(model,data,bid); tilt=float(np.degrees(np.arccos(np.clip(-g[2],-1,1)))); rec.append((data.time,data.qpos[2],tilt,np.max(np.abs(data.qvel[di])),np.max(np.abs(tau))))
   if data.qpos[2]<a.fall_height or tilt>a.fall_tilt: fell=True; break
 if viewer is not None: viewer.close()
 r=np.array(rec); out={'duration_s':float(data.time),'command':cmd.tolist(),'policy_enabled':not a.disable_policy,'fell':fell,'initial_action':None if first_action is None else first_action.tolist(),'initial_action_abs_max':None if first_action is None else float(np.max(np.abs(first_action))),'final_position':data.qpos[:3].tolist(),'height_min':float(r[:,1].min()),'height_final':float(r[-1,1]),'tilt_deg_max':float(r[:,2].max()),'tilt_deg_final':float(r[-1,2]),'max_abs_joint_velocity':float(r[:,3].max()),'max_abs_torque':float(r[:,4].max()),'samples':len(rec)}; a.json_out.parent.mkdir(parents=True,exist_ok=True); a.json_out.write_text(json.dumps(out,indent=2)+'\n'); return out
def cli():
 p=argparse.ArgumentParser(); p.add_argument('--policy_dir',type=Path,default=POLICY); p.add_argument('--urdf',type=Path,default=URDF); p.add_argument('--duration',type=float,default=8.); p.add_argument('--command_start',type=float,default=3.); p.add_argument('--command',nargs=3,type=float,default=(0.,0.,0.)); p.add_argument('--dt',type=float,default=.005); p.add_argument('--policy_hz',type=float,default=50.); p.add_argument('--height',type=float,default=.9); p.add_argument('--disable_policy',action='store_true'); p.add_argument('--viewer',action='store_true'); p.add_argument('--depth_npz',type=Path,default=None); p.add_argument('--fall_height',type=float,default=.45); p.add_argument('--fall_tilt',type=float,default=60.); p.add_argument('--json_out',type=Path,default=ROOT/'artifacts/mujoco/low_level_smoke.json')
 p.add_argument('--terrain',choices=('flat','stairs','ramp','custom'),default='flat'); p.add_argument('--terrain_file',type=Path,default=ROOT/'deploy/mujoco/terrains/rich_course.json'); p.add_argument('--terrain_start_x',type=float,default=1.5); p.add_argument('--terrain_width',type=float,default=2.0)
 p.add_argument('--stair_height',type=float,default=0.12); p.add_argument('--stair_depth',type=float,default=0.50); p.add_argument('--stair_steps',type=int,default=4)
 p.add_argument('--ramp_angle_deg',type=float,default=12.0); p.add_argument('--ramp_length',type=float,default=3.0); return p.parse_args()
if __name__=='__main__': print(json.dumps(run(cli()),indent=2))
