import math
import numpy as np
# 导入你刚才那个文件里的基础类
from rtt_generate import HybridAeroDatabase, EngineDatabase, FlightSimulator6DOF

if __name__ == "__main__":
    aircraft_params = {
        'S': 88.58, 'b': 18.9, 'c_bar': 4.6, 'mass': 14000,
        'Ixx': 313220.6, 'Iyy': 273435.3, 'Izz': 392903.9, 'Ixz': 0.0
    }

    flight_db = HybridAeroDatabase()
    flight_db._load_from_pickle('X47B.pkl')
    engine_db = EngineDatabase()
    engine_db.load1('engine.pkl')

    sim = FlightSimulator6DOF(flight_db, engine_db, aircraft_params)
    sim.set_initial_state(h_m=2000.0, V_mps=250.0, theta_deg=0.0)
    
    dt = 0.02
    print("\n=== 开始松杆试飞 (使用 State05 中立舵面) ===")
    print(f"初始状态: 高度 2000m, 速度 250m/s, 俯仰角 3.0°\n")
    
    for step in range(4000): # 10秒
        sim.step_rk4(dt, 'state05')
        
        u, w = sim.state[3], sim.state[5]
        V = math.sqrt(sim.state[3]**2 + sim.state[4]**2 + sim.state[5]**2)
        alpha = math.degrees(math.atan2(w, u)) if u != 0 else 0
        h = -sim.state[2]
        pitch = math.degrees(sim.state[7])
        
        # 每 1 秒打印一次状态
        if step % 50 == 0:
            print(f"Time: {step*dt:.1f}s | 高度: {h:.1f}m | 速度: {V:.1f}m/s | 迎角: {alpha:.2f}° | 俯仰: {pitch:.2f}°")
            
        # 坠毁检测
        if alpha > 15.0:
            print(f"❌ 坠毁！迎角 {alpha:.2f}° 突破上限 15° (时间: {step*dt:.2f}s)")
            break
        if alpha < -10.0:
            print(f"❌ 坠毁！迎角 {alpha:.2f}° 突破下限 -10° (时间: {step*dt:.2f}s)")
            break
        if h < 1500.0:
            print(f"❌ 坠毁！高度 {h:.1f}m 跌破下限 1500m (时间: {step*dt:.2f}s)")
            break

    else:
        print("\n✅ 试飞成功！飞机在不打舵的情况下安全存活了 10 秒。")