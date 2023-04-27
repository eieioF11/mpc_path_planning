
from casadi import *
import numpy as np
import math
import matplotlib.pyplot as plt

class MPC(object):
    # 問題設定
    T = 30.0     # ホライゾン長さ
    N = 10     # ホライゾン離散化グリッド数
    dt = T / N  # 離散化ステップ
    nx = 3      # 状態空間の次元
    nu = 3      # 制御入力の次元
    Vmax = 2.0
    #重み
    Q = [1.0, 1.0, 0.01] # 状態への重み
    R = [1.0, 1.0, 1.0]  # 制御入力への重み
    def __init__(self):
        pass

    def cost(self,inipos,iniv,path):#inipos=[x0,y0,yaw0] path=[[x,y,yaw],...]
        J = 0 # コスト関数
        for k in range(self.N):
            Uk = MX.sym('U_' + str(k), self.nu) # 時間ステージ k の制御入力 uk を表す変数
            self.w   += [Uk]                    # uk を最適化変数 list に追加
            self.lbw += [-self.Vmax,-self.Vmax,-self.Vmax]        # uk の lower-bound
            self.ubw += [self.Vmax,self.Vmax,self.Vmax]           # uk の upper-bound
            self.w0  += iniv                 # uk の初期推定解

            #運動方程式
            x   = self.Xk[0]    # X座標[m]
            y   = self.Xk[1]    # Y座標[m]
            yaw = self.Xk[2]    # ロボット角度[rad]
            #制御入力
            vx      = Uk[0]   # vx[m/s]
            vy      = Uk[1]   # vy[m/s]
            angular = Uk[2]   # w[rad/s]
            # ステージコストのパラメータ
            self.k_max+=1
            x_ref = path[k]           # 目標状態
            L = 0 # ステージコスト
            for i in range(self.nx):
                L += 0.5 * self.Q[i] * (self.Xk[i]-x_ref[i])**2
            for i in range(self.nu):
                L += 0.5 * self.R[i] * Uk[i]**2
            J = J + L * self.dt # コスト関数にステージコストを追加

            # Forward Euler による離散化状態方程式
            Xk_next = vertcat(x + vx * self.dt,
                            y + vy * self.dt,
                            yaw + angular * self.dt)
            Xk1 = MX.sym('X_' + str(k+1), self.nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
            self.w   += [Xk1]                       # xk+1 を最適化変数 list に追加
            self.lbw += [-inf, -inf, -inf]    # xk+1 の lower-bound （指定しない要素は -inf）
            self.ubw += [inf, inf, inf]       # xk+1 の upper-bound （指定しない要素は inf）
            self.w0  += inipos       # xk+1 の初期推定解

            # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
            self.g   += [Xk_next-Xk1]
            self.lbg += [0.0,0.0,0.0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
            self.ubg += [0.0,0.0,0.0] # 等式制約は lower-bo und と upper-bound を同じ値にすることで設定
            self.Xk = Xk1

        # 終端コストのパラメータ
        Vf = 0                            # 終端コスト
        for i in range(self.nx):
            Vf += 0.5 * self.Q[i] * (self.Xk[i]-x_ref[i])**2
        for i in range(self.nu):
            Vf += 0.5 * self.R[i] * Uk[i]**2
        J = J + Vf
        return J
    def solve(self,inipos,iniv,path):#inipos=[x0,y0,yaw0] path=[[x,y,yaw],...]
        # 以下で非線形計画問題(NLP)を定式化
        self.w   = []  # 最適化変数を格納する list
        self.w0  = []  # 最適化変数(w)の初期推定解を格納する list
        self.lbw = []  # 最適化変数(w)の lower bound を格納する list
        self.ubw = []  # 最適化変数(w)の upper bound を格納する list
        self.g = []    # 制約（等式制約，不等式制約どちらも）を格納する list
        self.lbg = []  # 制約関数(g)の lower bound を格納する list
        self.ubg = []  # 制約関数(g)の upper bound を格納する list

        self.Xk = MX.sym('X0', self.nx) # 初期時刻の状態ベクトル x0

        self.w += [self.Xk]             # x0 を 最適化変数 list (w) に追加
        # 初期状態は given という条件を等式制約として考慮
        self.lbw += inipos # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
        self.ubw += inipos # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
        self.w0  += inipos # x0 の初期推定解
        self.k_max=0
        #現在位置から一番近い経路の点取得
        d=(path[:,0]-inipos[0])**2+(path[:,1]-inipos[1])**2
        min_d=np.argmin(d)
        path_=list(path[min_d:,:])#経路の現在位置から一番近い経路の点からゴールまでを抜き出し
        #サイズNのリストのあまりをゴールの座標で埋める
        for i in range((self.N-len(path[min_d:,:]))):
            path_.append(path[-1,:])
        path_=np.array(path_)
        #評価関数式作成
        J=self.cost(inipos,iniv,path_)
        # 非線形計画問題(NLP)
        nlp = {'f': J, 'x': vertcat(*self.w), 'g': vertcat(*self.g)}
        # Ipopt ソルバー，最小バリアパラメータを0.001に設定
        solver = nlpsol('solver', 'ipopt', nlp, {'ipopt':{'mu_min':0.001}})

        # NLPを解く
        sol = solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        w_opt = sol['x'].full().flatten()
        print(self.k_max)
        tgrid = np.array([self.dt*k for k in range(self.k_max+1)])
        return w_opt, tgrid

# 目標状態(経路)
path = []
for i in range(10):
    path.append([i,1,0.0])
path=np.array(path)
#初期位置
inipos=[0.0,2.0,0.0]

#Test
def main():
    mpc = MPC()
    w_opt, tgrid = mpc.solve(inipos,[0,0,0],path)
    # 解をプロット
    x1_opt = np.array(w_opt[0::6])
    x2_opt = np.array(w_opt[1::6])
    x3_opt = np.array(w_opt[2::6])
    u1_opt  = np.array(w_opt[3::6])
    u2_opt  = np.array(w_opt[4::6])
    u3_opt  = np.array(w_opt[5::6])
    print(u1_opt[0],u2_opt[0],u3_opt[0])
    plt.figure(1)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(tgrid, x1_opt, '-')
    plt.plot(tgrid, x2_opt, '-')
    plt.plot(tgrid, x3_opt, '-')
    plt.step(tgrid, np.array(vertcat(DM.nan(1), u1_opt)), '-.')
    plt.step(tgrid, np.array(vertcat(DM.nan(1), u2_opt)), '-.')
    plt.step(tgrid, np.array(vertcat(DM.nan(1), u3_opt)), '-.')
    plt.xlabel('t')
    plt.legend(['x','y', 'yaw', 'vx','vy','angular'])
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.xlim([0,10])
    #plt.ylim([0,2.0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(path[:,0], path[:,1], '-')
    plt.plot(x1_opt, x2_opt, '.')
    plt.grid()
    plt.show()

#if __name__ == "__main__":
#    try:
#        main()
#    except KeyboardInterrupt:
#        sys.exit()