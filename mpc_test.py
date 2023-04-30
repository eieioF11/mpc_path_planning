import casadi
import pathplanner.a_star as a_star
import numpy as np
import matplotlib.pyplot as plt


class mpc_config(object):
    horizon = 10  # ホライゾン長さ
    dt = 0.01  # 離散化ステップ
    # 機体スペック
    max_velocity = 1.0  # [m/s]
    max_angular = 1.0  # [m/s]
    max_acceleration = 0.98  # [m/s^2]
    max_angular_acceleration = 0.98  # [rad/s^2]
    # 重み
    state_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 状態への重み Q
    control_weight = [1.0, 1.0, 1.0]  # 制御入力への重み R
    final_state_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 終端の状態への重み Q_final
    # final_control_weight = [1.0, 1.0, 1.0]  # 終端の制御入力への重み R_final
    # モデルパラメータ
    xy_vel_time_constant = 0.0
    theta_vel_time_constant = 0.0

    def __init__(self):
        pass


opti_ = casadi.Opti()

config_ = mpc_config()

# 運動モデル
# 静止座標系
# 入力量: 各軸方向の速度入力、実際の機体速度はこの入力に遅れが生じたものであるとする
# 状態量: [velocity_x, velocity_y, velocity_theta, position_x, position_y, position_theta]^T
class Kinematic:
    # 状態量
    velocity_ = casadi.MX.sym("velocity", 3)
    position_ = casadi.MX.sym("position", 3)
    state_ = casadi.vertcat(velocity_, position_)
    # 入力量
    control_ = casadi.MX.sym("control", 3)
    def __init__(self,config):
        self.lpf_xy_gain_ = config_.dt / (config_.dt + config_.xy_vel_time_constant)
        self.lpf_theta_gain_ = config_.dt / (config_.dt + config_.theta_vel_time_constant)
    def set_variable(self,vel_n,pos_n,control_n):
        self.velocity_ = casadi.MX.sym("velocity", vel_n)
        self.position_ = casadi.MX.sym("position",pos_n)
        self.state_ = casadi.vertcat(self.velocity_, self.position_)
        self.control_ = casadi.MX.sym("control", control_n)
    def model(self):
        pass
# 離散化した全方位移動モデル
class OmniDirectional(Kinematic):
    def __init__(self,config):
        super().__init__(config)
        self.set_variable(3,3,3)
    def model(self):
        MJ = casadi.vertcat(self.lpf_xy_gain_,self.lpf_xy_gain_, self.lpf_theta_gain_)
        MMJ = casadi.MX.ones(3) - MJ
        velocity_next = casadi.times(MMJ, self.velocity_) + casadi.times(MJ, self.control_)
        position_next = self.position_ + config_.dt * velocity_next
        state_next = casadi.vertcat(velocity_next, position_next)
        return casadi.Function("OmniDirectional", [self.state_, self.control_], [state_next])

# 制約条件
def guard_circle_subject(xy,center,size,comp):#円の制約
    dx = casadi.times(casadi.MX(xy) - casadi.MX(center),casadi.vertcat(casadi.MX({2./size[0],2./size[1]})))
    lh = dx**2
    if comp == "<" or comp == "keep in":
        return lh < 1
    if comp == ">" or comp == "keep out":
        return lh > 1
    if comp == "<=":
        return lh <= 1
    if comp == ">=":
        return lh >= 1
    return lh

def guard_rect_subject(xy,center,size,comp):#四角の制約
    dx = casadi.times(casadi.MX(xy) - casadi.MX(center),casadi.vertcat(casadi.MX({1./size[0],1./size[1]})))
    lh = casadi.MX(abs(dx[0]+dx[1])+abs(dx[0]-dx[1]))
    if comp == "<" or comp == "keep in":
        return lh < 1
    if comp == ">" or comp == "keep out":
        return lh > 1
    if comp == "<=":
        return lh <= 1
    if comp == ">=":
        return lh >= 1
    return lh

# 最適化変数
u_sol_ = casadi.DM.zeros(3, config_.horizon)
x_sol_ = casadi.DM.zeros(6, config_.horizon + 1)
x_init_ = casadi.DM.zeros(x_sol_.size())
U = opti_.variable(3, config_.horizon)
X = opti_.variable(6, config_.horizon + 1)
X_target_ = opti_.parameter(6, 1)
current_state_ = opti_.parameter(6, 1)
# 重み
R_vec = casadi.DM(config_.control_weight)
Q_vec = casadi.DM(config_.state_weight)
Q_final_vec = casadi.DM(config_.final_state_weight)

Q = casadi.diag(Q_vec)
Q_final = casadi.diag(Q_final_vec)
R = casadi.diag(R_vec)

print(R)
# 評価関数
obj = casadi.MX.zeros(1)
for i in range(config_.horizon):
    dx_i = X[:, i + 1] - X_target_
    du = U[:, i]
    obj += casadi.mtimes(casadi.mtimes(dx_i.T, Q), dx_i)
    obj += casadi.mtimes(casadi.mtimes(du.T, R), du)
dx_final = X[:,config_.horizon] - X_target_
obj += casadi.mtimes(casadi.mtimes(dx_final.T, Q_final), dx_final)
print(obj)
opti_.minimize(obj)
# 制約条件を定義
kinematic=OmniDirectional(config_)
kinematic_model = kinematic.model()
opti_.subject_to(X[:, 0] == current_state_)  # 初期状態
for i in range(config_.horizon):
    # 対象が従う運動モデル
    opti_.subject_to(X[:, i + 1] == kinematic_model(X[:, i], U[:, i])[0])
    # 加速度制約
    max_vel_diff_sqr = (config_.max_acceleration * config_.dt)**2
    diff_vel = X[0:2, i + 1] - X[0:2, i]
    opti_.subject_to(diff_vel**2 <= max_vel_diff_sqr)
    diff_angular = X[2, i + 1] - X[2, i]
    opti_.subject_to(-config_.max_angular_acceleration * config_.dt <= diff_angular)
    opti_.subject_to(diff_angular <= config_.max_angular_acceleration * config_.dt)
    # 速度制約
    diff_xy = X[3:5, i + 1] - X[3:5, i]
    opti_.subject_to(diff_xy**2 <= config_.max_velocity)
    diff_yaw = X[5, i + 1] - X[5, i]
    opti_.subject_to(-config_.max_angular <= diff_yaw)
    opti_.subject_to(diff_yaw <= config_.max_angular)

# 最適化ソルバを設定
opti_.solver("ipopt")

# 初期値設定
current_position = [0.,0.,0.]
current_velocity = [0.,0.,0.]
target_position =  [9.,9.,0.]

current_pos = casadi.DM(current_position)
current_vel = casadi.DM(current_velocity)

dm_current_state_ = casadi.vertcat(current_velocity,current_position)
dm_X_target = casadi.DM([0.,0.,0.]+target_position)
opti_.set_value(current_state_,dm_current_state_)

x_init_ = casadi.DM.zeros(x_init_.size())
u_init = casadi.DM.zeros(U.size())

# grid_path
path = np.array([])
grid=np.array([
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0]
])
map_resolution = 1
origin = [0,0]
def path_length(path):
    length=0.
    for i in range(len(path)-1):
        diff = path[i+1]-path[i]
        length+=np.sqrt(abs(diff[0])+abs(diff[1]))
    return length
#変換
def conversion_grid_pos(map_pos):
    return [int(map_pos[0]/map_resolution+origin[0]),int(map_pos[1]/map_resolution+origin[1])]
def conversion_map_pos(grid_pos):
    return [(grid_pos[0]-origin[0])*map_resolution,(grid_pos[1]-origin[1])*map_resolution]
#線形補間
def lerp(path,t):
    if t<0.0:
        t=0
    passed_len = 0.
    p=np.array([path[-1][0],path[-1][1]])
    for i in range(len(path)-1):
        diff = path[i+1]-path[i]
        l=np.sqrt(abs(diff[0])+abs(diff[1]))
        if (passed_len+l) > t:
            t-=passed_len
            t/=l
            p=path[i]+diff*t
            return p
        passed_len+=l
    return p

grid_path_planner = a_star.PathPlanner(grid,False)
init,path=grid_path_planner.a_star(np.array([int(current_position[0]),int(current_position[1])]),np.array([int(target_position[0]),int(target_position[1])]))
if init!=-1:
    path=np.vstack((init,path))#初期位置をpathに追加
    #結果表示
    print(path)
    plt.imshow(grid)
    plt.plot(path[:,1],path[:,0])
    plt.show()
# 初期値代入
path_len = path_length(path)
path_pos = casadi.DM.zeros(6)
init_path_result = []
for i in range(config_.horizon):
    if init!=-1:
        #grid path
        yaw_t = i / (config_.horizon -1.0)
        step = path_len / (config_.horizon -1.0)

        max_step = config_.max_velocity * config_.dt / map_resolution
        step = min([step,max_step])

        glen = step * i
        if glen > path_len:
            glen = path_len

        grid_path_pos = lerp(path,glen)
        print(grid_path_pos,glen)
        grid_path_pos=conversion_map_pos(grid_path_pos)
        yaw = current_position[2] + (target_position[2]-current_position[2])*yaw_t
        init_path_result.append([grid_path_pos[0],grid_path_pos[1],yaw])
        pos = casadi.DM([grid_path_pos[0],grid_path_pos[1],yaw])
        vel = (pos-x_init_[3:6,i])/config_.dt
        print(vel,pos)
        path_pos = casadi.vertcat(vel,pos)
        u_init[:,i] = vel
        x_init_[:,i+1] = path_pos
    else :
        #線形補間
        print('error')
        u_init[:,i] = casadi.DM.zeros(3)
        x_init_[:,i+1]=dm_current_state_ + (dm_X_target - dm_current_state_)*(i/(config_.horizon-1))
        print(x_init_[:,i])

init_path_result=np.array(init_path_result)

opti_.set_initial(U,u_init)
opti_.set_initial(X,x_init_)

# end_diff = (dm_X_target - x_init_[:,config_.horizon-1])[3:5]
# print("end_diff")
# end_diff_vec = np.array(end_diff.T)[0]
# print(end_diff_vec)
# if np.linalg.norm(end_diff_vec, ord=0)>0.5:
#     dm_X_target = x_init_[:,config_.horizon-1]

opti_.set_value(X_target_,dm_X_target)


# 最適化
sol = opti_.solve()

print(f'評価関数：{sol.value(obj)}')
print(f'X: {sol.value(X)}')
print(f'U: {sol.value(U)}')
# 解をプロット
x1_opt = np.array(sol.value(X)[0::6])
x2_opt = np.array(sol.value(X)[1::6])
x3_opt = np.array(sol.value(X)[2::6])
u1_opt  = np.array(sol.value(X)[3::6])
u2_opt  = np.array(sol.value(X)[4::6])
u3_opt  = np.array(sol.value(X)[5::6])
print(u1_opt[0],u2_opt[0],u3_opt[0])
plt.figure(1)
plt.clf()
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.plot(init_path_result[:,0], init_path_result[:,1], '-')
plt.plot(x1_opt, x2_opt, '.')
plt.grid()
plt.show()
