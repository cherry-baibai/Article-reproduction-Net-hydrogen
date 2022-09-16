from gurobipy import *
import matplotlib.pyplot as plt

#为啥基本模型无解捏，参数数值改过，分析可能是以基于时间的模型有点问题，换成基于距离的模型trytry
#基于时间的模型还是有问题，基于距离的模型在基于此的参数的基础上是可以运行的（出问题无解的点在于对速度的约束条件时应该保证初始和最终速度为1而非0）
#所以改动的两个点1.最初和最后速度的初始条件 2.能量的限制的下限为负无穷

#import the parameters information
Distance = 18000 #m
Time_total = 500 #s
H = 0 #don't consider the gradient temporary
cap = 40
M_Total = 178000 #kg
N = 101
N_V = 9 #speed is divided into N_v(used in alpha/beta)
delta_d = int(Distance/(N-1))
#delta_t = Time_total/(N-1)
delta_h = 0
Acc_max_a = 1.2 #m/s2
Acc_max_b = -1.2 #m/s2
A = 3.6449 #kn
B = 0.001710 #kn/(m/s)
C = 0.01134 #kn/(m2/s2)
Fb_Max = 250000 #N 这个可以导入实际数据来同样进行线性分段，如同PWL_SPE的处理方式
Ft_Max = 250000 #N
P_b_max = 9376000 #W
P_t_max = 9376000 #W
P_fc_max = 250000 #W
P_ESD_max = 400000 #W
n_m = 0.6 #motor efficiency
n_ESD = 0.95 #ESD efficiency
n_fc_max = 0.84 #maximum efficiency of fuel cell
g = 9.8
v_max = 80 #mps
v_min = 1 #mps

i = list(range(1,N)) #1-100
ii = list(range(0,N)) #0-100

#piecewise linearisation accuracy(creat a speed list)
delta_speed = v_max / (N_V - 1)
PWL_SPE = [v_min] #from 0 to 33,piece 35
pre = 0
for index in range(N_V-1):
    pre = pre + delta_speed
    PWL_SPE.append(pre)
#speed limitation
v_limit = []
for index in range(0, 25):
    v_limit.append(70) #此时设定的速度的限制的数值导致公式（6）的速度限制是一个废的条件，后续改进
for index in range(25, 29):
    v_limit.append(50)
for index in range(29, N):
    v_limit.append(70)

#energy storage parametewrs(不知道单位)
E_cap = 1000000*cap +1
PESD = 400000 #it is Pd_max and Pc_max (to constrain the distributed energy)
H_heat_value = 100000 #kJ/kg

#modelling (this is based on time, the formulation is a little different to the model which is based on distance)
m = Model('hydrogen_power')
#delta_d_i= m.addVars(i, vtype=GRB.CONTINUOUS, name='Elapsed distance')
delta_t_i= m.addVars(i,lb=0.0, vtype=GRB.CONTINUOUS, name='Elapsed time')
#delta_t_i1d = m.addVars(i,lb=0.0, vtype=GRB.CONTINUOUS, name='1/Elapsed time ')

f_i_drag = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Average drag force')

v_i = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name='speed')
v_i2 = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name='square of speed')
v_ave_i = m.addVars(i,lb=0.0, vtype=GRB.CONTINUOUS, name='Average speed')
v_ave_i2 = m.addVars(i,lb=0.0, vtype=GRB.CONTINUOUS, name='square of average speed')
v_ave_i1d = m.addVars(i,lb=0.0, vtype=GRB.CONTINUOUS, name='1/average speed')

E_i_seg = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Applied force')
#the former is basic model parameters, the latter is fuel cell hybrid parameters
E_i_fc = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='energy from fc')
E_i_dis = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='energy discharge to ESD')
E_i_ch = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='energy charge to ESD') #it is positive because in formulation there exist a minus sign
E_init = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name="the initial energy of the ESD")#the initial is relative which is updated after every stage

P_i_ESD = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the net power of ESD')#this part P is corresponding to the mass, use P or M is same
P_i_fc = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the power of fc')

m_i_ESD = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the net hydrogen mass of ESD')
m_i_fc = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the hydrogen mass from fc')

lambda_i = m.addVars(i, vtype=GRB.BINARY, name='to judge the energy comes back or need')
SOE_i = m.addVars(i, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='the state of energy')
Cr_i_fc = m.addVars(i, vtype=GRB.CONTINUOUS, name='hydrogen consumption rate')

alpha = m.addVars(N, N_V, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='a')#N_V is the dimension of special speed at x for speed
beta = m.addVars(N-1, N_V, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='b')#N is the dimension of all the speed through the whole distance for average speed
#equation of SOS2 variable constrain
for index in ii:
    m.addSOS(GRB.SOS_TYPE2, [alpha[index, j] for j in range(N_V)],)
for index in i:
    m.addSOS(GRB.SOS_TYPE2, [beta[index-1, j] for j in range(N_V)],) #beta的下标只能从0开始
m.addConstrs((alpha.sum(index, '*') == 1 for index in ii), name='SOS2 property_α ')
m.addConstrs((beta.sum(index-1, '*') == 1 for index in i), name='SOS2 property_β')

#sos2 express the velocity
for index in ii:
    m.addConstr(v_i[index] == (quicksum(PWL_SPE[j] * alpha[index, j] for j in range(N_V))), name='candidate speed')
    m.addConstr(v_i2[index] == (quicksum(PWL_SPE[j]**2 * alpha[index, j] for j in range(N_V))), name='candidate speed square')

for index in i:
    m.addConstr(v_ave_i[index] == (quicksum(beta[index-1, j] * PWL_SPE[j] for j in range(N_V))), name='candidate average speed')
    m.addConstr(v_ave_i2[index] == (quicksum(beta[index-1, j] * PWL_SPE[j]**2 for j in range(N_V))), name='candidate average speed square')
    m.addConstr(v_ave_i1d[index] == (quicksum(1 / PWL_SPE[j] * beta[index-1, j] for j in range(N_V))), name='candidate 1/average speed')

#Equation(1):travel distance(this model is based on time not on distance)
for index in i:
    m.addConstr(delta_t_i[index] == delta_d * v_ave_i1d[index], name='For Δdi')
    #m.addConstr(v_ave_i[index] == delta_d * delta_t_i1d[index] , name='For Δdi1')

m.addConstr(quicksum(delta_t_i) <= Time_total, name='total distance') #是不是时间和距离同时限定为等号所以过约束了导致模型无解？

#Equation(2):the average speed
for index in i:
    m.addConstr(v_ave_i[index] * 2 == v_i[index] + v_i[index - 1], name='average speed')

#Equation(4):Davis equation
for index in i:
    m.addConstr(f_i_drag[index] == 1000*(A + B * v_ave_i[index] + C * v_ave_i2[index]), name='Davis equation')

#Equation(5): acceleration constrain
for index in i:
    m.addRange(0.5 * (v_i2[index] - v_i2[index-1]) / delta_d, Acc_max_b, Acc_max_a, name='acceleration constrain')

#Equation(6): velocity constrain
for index in range(0, N-1):
    m.addConstr(v_i[index] <= v_limit[index], name='velocity constrain upper')
    m.addConstr(v_i[index] >= 0, name='velocity constrain lower')
m.addConstr(v_i[0] == 1, name="起点")
m.addConstr(v_i[N-1] == 1, name="终点")

#Equation(7): conservation of energy
for index in i:
    m.addConstr(E_i_seg[index]*n_m - M_Total * g * delta_h - 0.5 * M_Total * (v_i2[index] - v_i2[index-1]) - f_i_drag[index] * delta_d >= 0, name='conservation of energy1')
    m.addConstr(E_i_seg[index]/n_m - M_Total * g * delta_h - 0.5 * M_Total * (v_i2[index] - v_i2[index-1]) - f_i_drag[index] * delta_d >= 0, name='conservation of energy2')

#Equation(9):energy constrain
for index in i:
    m.addConstr(E_i_seg[index] >= -Fb_Max * delta_d * n_m, name='braking fore constrain')
    m.addConstr(E_i_seg[index] <= Ft_Max * delta_d/n_m, name='traction force constrain')#using divide because the obtain force is smaller than the power grid provided
    m.addConstr(E_i_seg[index] >= -P_b_max * delta_t_i[index] * n_m, name='braking power constrain')
    m.addConstr(E_i_seg[index] <= P_t_max * delta_t_i[index] / n_m, name='traction power constrain')

#energy sum
E_total = quicksum(E_i_seg)

# the model for the energy distribution
#here is the co-optimization, maybe the problem will be occured in Cri_fc and P_ifc ,delta_t will be a certain value in sequence optimization
#while it is an uncertain value in co-optimization
#Equation(10,11,12) the relation between distruted energy and the all energy
for index in i:
    m.addConstr(
        E_i_seg[index] <= lambda_i[index] * (E_i_fc[index] + E_i_dis[index] * n_ESD) - (1 - lambda_i[index]) * E_i_ch[
            index] / n_ESD) #the total energy distribution

    m.addConstr(E_i_ch[index] <= (1-lambda_i[index]) * 10000000000) #charge and dischage only one can exist
    m.addConstr(E_i_fc[index] <= lambda_i[index] * 10000000000)
    m.addConstr(E_i_dis[index] <= lambda_i[index] * 10000000000)

    m.addConstr(E_i_fc[index] <= P_fc_max * delta_t_i[index])#the distributed energy constrain
    m.addConstr(E_i_ch[index] <= PESD * delta_t_i[index])
    m.addConstr(E_i_dis[index] <= PESD * delta_t_i[index])

#Equation(13) SOE expression (0-1 constrain has already written in the definition of SOE)
E_init[0] = E_cap
for index in i:
    m.addConstr(SOE_i[index] == (E_init[index-1] + E_i_ch[index] - E_i_dis[index]) / E_cap)
    m.addConstr(E_init[index] == SOE_i[index] * E_cap)

#Equation(14) build the connection between Cri,fc and Pi,fc and calculate the m_fc
for index in i:
    m.addConstr(P_i_fc[index] == E_i_fc[index] * v_ave_i[index] / delta_d)
    m.addGenConstrPWL(P_i_fc[index], Cr_i_fc[index], [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
                      [0, 0.3571, 0.64935, 0.931677, 1.19, 1.5133, 1.84729, 2.19298, 2.5975, 3.032345, 3.571428],
                      name='Power-Efficiency_Characteristic')
    m.addConstr(m_i_fc[index] == Cr_i_fc[index] * delta_t_i[index])
m_fc = quicksum(m_i_fc)

#Equation(15) calculate the m_ESD
m_ESD = ((1 - SOE_i[N-1]) * E_cap) / H_heat_value / n_fc_max



#objective function
obj = m_fc + m_ESD
m.setObjective(obj, GRB.MINIMIZE)



m.optimize()

#plot the graph
v_point = []
for index in ii:
    v_point.append(v_i[index].x * 3.6)
plt.plot(range(0,N),v_point)
plt.show()

















