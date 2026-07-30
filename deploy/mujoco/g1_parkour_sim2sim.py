"""MuJoCo sim2sim smoke test for the frozen Instinct G1 parkour policy."""
from __future__ import annotations
import argparse,json,re,tempfile,time
from collections import deque
from pathlib import Path
import mujoco,numpy as np,onnxruntime as ort,yaml
ROOT=Path(__file__).resolve().parents[2]; POLICY=ROOT/'artifacts/low_level/g1_parkour'; URDF=ROOT/'external/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf'
JOINTS='left_shoulder_pitch_joint right_shoulder_pitch_joint waist_pitch_joint left_shoulder_roll_joint right_shoulder_roll_joint waist_roll_joint left_shoulder_yaw_joint right_shoulder_yaw_joint waist_yaw_joint left_elbow_joint right_elbow_joint left_hip_pitch_joint right_hip_pitch_joint left_wrist_roll_joint right_wrist_roll_joint left_hip_roll_joint right_hip_roll_joint left_wrist_pitch_joint right_wrist_pitch_joint left_hip_yaw_joint right_hip_yaw_joint left_wrist_yaw_joint right_wrist_yaw_joint left_knee_joint right_knee_joint left_ankle_pitch_joint right_ankle_pitch_joint left_ankle_roll_joint right_ankle_roll_joint'.split()
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
def build_model(path,dt):
 if path.suffix.lower() != ".urdf":
  loaded=mujoco.MjModel.from_xml_path(str(path))
  with tempfile.TemporaryDirectory() as td:
   saved=Path(td)/"robot.xml"; mujoco.mj_saveLastXML(str(saved),loaded); xml=saved.read_text()
   xml=re.sub(r'file="([^"]+)"',lambda m:'file="'+str((path.parent/"meshes"/m.group(1)).resolve())+'"',xml)
   xml=xml.replace('<worldbody>','<worldbody><geom name="floor" type="plane" size="20 20 0.1" friction="1 0.005 0.0001"/>',1)
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
    <geom name="floor" type="plane" size="20 20 0.1"/>'''
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
 p=load_params(a.policy_dir); model=build_model(a.urdf,a.dt); data=mujoco.MjData(model); qi,di=ids(model); bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link'); model.dof_armature[di]=p['armature']
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
 p=argparse.ArgumentParser(); p.add_argument('--policy_dir',type=Path,default=POLICY); p.add_argument('--urdf',type=Path,default=URDF); p.add_argument('--duration',type=float,default=8.); p.add_argument('--command_start',type=float,default=3.); p.add_argument('--command',nargs=3,type=float,default=(0.,0.,0.)); p.add_argument('--dt',type=float,default=.005); p.add_argument('--policy_hz',type=float,default=50.); p.add_argument('--height',type=float,default=.9); p.add_argument('--disable_policy',action='store_true'); p.add_argument('--viewer',action='store_true'); p.add_argument('--depth_npz',type=Path,default=None); p.add_argument('--fall_height',type=float,default=.45); p.add_argument('--fall_tilt',type=float,default=60.); p.add_argument('--json_out',type=Path,default=ROOT/'artifacts/mujoco/low_level_smoke.json'); return p.parse_args()
if __name__=='__main__': print(json.dumps(run(cli()),indent=2))
