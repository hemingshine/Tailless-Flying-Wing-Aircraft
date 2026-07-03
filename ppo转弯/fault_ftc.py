#coding=utf-8
"""
fault_ftc.py —— 内环自适应容错增量补偿 (你的"积分反步 + RBF + 滑模"思想的移植)

设计为【慢速自适应增量】，叠加在 RL/PID 基线舵面指令之上：
  · 积分项     : 抹平舵效损失造成的稳态误差(积分反步思想)
  · RBF 神经项 : 在线学习并补偿非线性/突发故障(σ-modification 保稳)
  · 滑模鲁棒项 : tanh 柔化的强镇压，吃掉残余抖动
基线(RL)负责快速名义跟踪；本模块慢、且带泄漏，名义飞行时基本休眠，
只在持续性故障下逐渐顶出补偿，不与 RL 抢快动态 —— 即 L1/MRAC 式增广。
"""
import math
import numpy as np


class AxisFTC:
    def __init__(self, sign, c=2.0, ki=1.5, eta=2.0, phi=1.2,
                 gamma=2.0, sigma=0.1, i_clip=12.0, out_clip=12.0, w_clip=30.0,
                 e_dead=0.8, s_dead=1.5, rate_tau=0.05):
        # sign: 该轴"正姿态误差需要的舵偏符号"。俯仰=-1(抬头需负flap)，滚转=+1。
        self.sign = sign
        self.c, self.ki, self.eta, self.phi = c, ki, eta, phi
        self.gamma, self.sigma = gamma, sigma
        self.i_clip, self.out_clip, self.w_clip = i_clip, out_clip, w_clip
        # ★ 抗抖振三件套：误差死区(名义真休眠)、滑模死区(tanh不在小s处嗡)、角速率低通时间常数
        self.e_dead, self.s_dead, self.rate_tau = e_dead, s_dead, rate_tau
        self.rate_f = 0.0     # 低通后的角速率
        e_c = np.linspace(-8.0, 8.0, 5)
        s_c = np.linspace(-30.0, 30.0, 5)
        self.centers = np.array(np.meshgrid(e_c, s_c)).T.reshape(-1, 2)
        self.width = 6.0
        self.W = np.zeros(len(self.centers))
        self.I = 0.0
        self.enabled = True

    def reset(self):
        self.I = 0.0
        self.W[:] = 0.0
        self.rate_f = 0.0

    def compute(self, e, rate, dt):
        """e: 姿态误差(deg, target-actual); rate: 该轴机体角速率(deg/s, 俯仰用q/滚转用p)。
        返回叠加到该轴舵偏的补偿量(deg)。"""
        if not self.enabled:
            return 0.0
        # ★ 角速率低通：不把高频速率灌进滑模面(抖振主驱动之一)
        a = min(1.0, dt / self.rate_tau)
        self.rate_f += (rate - self.rate_f) * a
        # 滑模面：误差 + 速率阻尼 (用滤波后的速率)
        s = self.c * e - self.rate_f
        s_b = float(np.clip(s, -30.0, 30.0))
        # ★ 误差死区：名义小误差时不累积、缓慢泄漏，FTC 真正休眠不顶舵
        if abs(e) > self.e_dead:
            self.I = float(np.clip(self.I + e * dt, -self.i_clip, self.i_clip))
            learn = 1.0
        else:
            self.I *= max(0.0, 1.0 - 0.5 * dt)   # 名义缓慢回零
            learn = 0.0
        # RBF 在线学习 (σ-modification 鲁棒自适应；名义只泄漏不学习)
        x = np.array([e, s_b])
        h = np.exp(-np.sum((self.centers - x) ** 2, axis=1) / (2 * self.width ** 2))
        self.W += self.gamma * (learn * s_b * h - self.sigma * self.W) * dt
        self.W = np.clip(self.W, -self.w_clip, self.w_clip)
        f_nn = float(np.dot(self.W, h))
        # ★ 滑模鲁棒项：带死区的 tanh —— |s|<s_dead 不输出(名义不嗡)，故障下全力镇压
        if abs(s) > self.s_dead:
            s_eff = s - math.copysign(self.s_dead, s)
            robust = self.eta * math.tanh(s_eff / self.phi)
        else:
            robust = 0.0
        comp = self.ki * self.I + f_nn + robust
        return self.sign * float(np.clip(comp, -self.out_clip, self.out_clip))


class InnerFTC:
    """三轴容错增量管理器。在内环 step 里，对 RL 给出的 delta_e/delta_a/delta_r 增广。"""
    def __init__(self, pitch=True, roll=True, yaw=False):
        # 俯仰: 抬头需 d_flap<0 => sign=-1; 滚转: 右滚需 +ail => sign=+1
        self.pitch = AxisFTC(sign=-1.0) if pitch else None
        self.roll = AxisFTC(sign=+1.0, c=2.5, ki=1.0, eta=1.5) if roll else None
        # 偏航舵(扰流板)又弱又强耦合，默认关闭；如需可单独标定 sign
        self.yaw = AxisFTC(sign=-1.0, c=1.0, ki=0.5, eta=1.0, out_clip=20.0) if yaw else None

    def reset(self):
        for a in (self.pitch, self.roll, self.yaw):
            if a:
                a.reset()

    def augment(self, delta_e, delta_a, delta_r, e_theta, q, e_phi, p, e_beta, r, dt):
        if self.pitch:
            delta_e += self.pitch.compute(e_theta, q, dt)
        if self.roll:
            delta_a += self.roll.compute(e_phi, p, dt)
        if self.yaw:
            delta_r += self.yaw.compute(e_beta, r, dt)
        return delta_e, delta_a, delta_r