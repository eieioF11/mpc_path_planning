import casadi


class mpc_config(object):
    horizon = 5  # ホライゾン長さ
    dt = 0.1  # 離散化ステップ
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


lpf_xy_gain_ = config_.dt / (config_.dt + config_.xy_vel_time_constant)
lpf_theta_gain_ = config_.dt / (config_.dt + config_.theta_vel_time_constant)


# 離散化した全方位移動モデル
# 静止座標系
# 入力量: 各軸方向の速度入力、実際の機体速度はこの入力に遅れが生じたものであるとする
# 状態量: [vel_x, vel_y, vel_theta, pos_x, pos_y, pos_theta]^T
def gen_kinematic_model():
    # 状態量
    vel = casadi.MX.sym("vel", 3)
    pos = casadi.MX.sym("pos", 3)
    state = casadi.vertcat(vel, pos)
    # 入力量
    control = casadi.MX.sym("control", 3)

    MJ = casadi.vertcat(lpf_xy_gain_, lpf_xy_gain_, lpf_theta_gain_)
    MMJ = casadi.MX.ones(3) - MJ
    vel_next = casadi.times(MMJ, vel) + casadi.times(MJ, control)
    pos_next = pos + config_.dt * vel_next
    state_next = casadi.vertcat(vel_next, pos_next)
    return casadi.Function("kinematic_model", [state, control], [state_next])


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
dx_final = X[config_.horizon] - X_target_
obj += casadi.mtimes(casadi.mtimes(dx_final.T, Q_final), dx_final)
print(obj)
opti_.minimize(obj)
# 制約条件を定義
kinematic_model = gen_kinematic_model()
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
target_position = [9.,9.,0.]

current_pos = casadi.DM(current_position)
current_vel = casadi.DM(current_velocity)

dm_current_state_ = casadi.vertcat(current_velocity,current_position)
dm_X_target = casadi.DM(target_position+[0.,0.,0.])
opti_.set_value(current_state_,dm_current_state_)

x_init_ = casadi.DM.zeros(x_init_.size())



# 最適化
# sol = opti_.solve()

# print(f'評価関数：{sol.value(obj)}')
# print(f'X: {sol.value(X)}')
# print(f'U: {sol.value(U)}')
