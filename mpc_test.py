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

    def __init__(self):
        pass


opti = casadi.Opti()

config_ = mpc_config()
# 最適化変数
u_sol_ = casadi.DM.zeros(3, config_.horizon)
x_sol_ = casadi.DM.zeros(6, config_.horizon + 1)
x_init_ = casadi.DM.zeros(x_sol_.size())
U = opti.variable(3, config_.horizon)
X = opti.variable(6, config_.horizon + 1)
X_target_ = opti.parameter(6, 1)
curr_state_ = opti.parameter(6, 1)
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
    dx_i = X[:,i + 1] - X_target_
    du = U[:,i]
    obj += casadi.mtimes(casadi.mtimes(dx_i.T, Q), dx_i)
    obj += casadi.mtimes(casadi.mtimes(du.T, R), du)
    print(dx_i)
    print(du)
dx_final = X[config_.horizon] - X_target_
obj += casadi.mtimes(casadi.mtimes(dx_final.T, Q_final), dx_final)
print(obj)
opti.minimize(obj)
# 制約条件を定義
# opti.subject_to( x1*x2 >= 1 )
# opti.subject_to( x1 >=0 )
# opti.subject_to( x2 >=0 )

# 最適化ソルバを設定
opti.solver("ipopt")
# 最適化
#sol = opti.solve()

# print(f'評価関数：{sol.value(obj)}')
# print(f'X: {sol.value(X)}')
# print(f'U: {sol.value(U)}')
