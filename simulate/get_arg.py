#coding=utf-8
def estimate_inertia(mass_kg, span_m, length_m, aircraft_type='flying_wing'):
    """
    根据经验公式估算飞机的转动惯量
    """
    if aircraft_type == 'flying_wing':
        # 针对 X-47B 等飞翼布局的经验系数
        kx = 0.25  # 基于展长
        ky = 0.38  # 基于机长
        kz = 0.28  # 基于展长
        
        Rx = kx * span_m
        Ry = ky * length_m
        Rz = kz * span_m
        
    elif aircraft_type == 'fighter':
        # 针对常规战斗机
        kx = 0.30  # 基于展长
        ky = 0.40  # 基于机长
        kz = 0.45  # 基于机长
        
        Rx = kx * span_m
        Ry = ky * length_m
        Rz = kz * length_m
        
    else: # airliner / transport
        kx = 0.28
        ky = 0.38
        kz = 0.45
        Rx = kx * span_m
        Ry = ky * length_m
        Rz = kz * length_m

    Ixx = mass_kg * (Rx ** 2)
    Iyy = mass_kg * (Ry ** 2)
    Izz = mass_kg * (Rz ** 2)
    Ixz = 0.0 # 初始假设为 0
    
    return {'Ixx': Ixx, 'Iyy': Iyy, 'Izz': Izz, 'Ixz': Ixz}

# ===== X-47B 估算示例 =====
# 数据参考开源资料：空重约 6350kg，满载约 20000kg。这里假设当前测试质量为 14000kg
# 翼展约 18.9m，机长约 11.6m

x47b_mass = 14000.0  
x47b_span = 18.92    
x47b_length = 11.63  

inertia = estimate_inertia(x47b_mass, x47b_span, x47b_length, 'flying_wing')

print("估算出的 X-47B 转动惯量 (kg*m^2):")
for k, v in inertia.items():
    print(f"{k}: {v:.1f}")

# 你可以直接把打印出来的结果填入 FlightSimulator6DOF 的 global_params 中！