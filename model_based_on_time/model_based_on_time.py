from gurobipy import *
import matplotlib.pyplot as plt
import numpy


#为啥基本模型无解捏，参数数值改过，分析可能是以基于时间的模型有点问题，换成基于距离的模型trytry
#基于时间的模型在能量守恒的地方由于F_drag与Δd相乘会造成非线性问题，因此在最开始对于F_drag的定义的时候就计算出其能量的值

#import the parameters information
Distance = 10000 #m
Time_total = 450 #s
cap = 40
M_Total = 72700 + 50*cap #kg
N = 41 #这个莫名奇妙不知道为啥会影响总的’氢耗量‘
N_V = 35 #speed is divided into N_v(used in alpha/beta)
#delta_d = int(Distance/(N-1))
delta_t = Time_total/(N-1)

Acc_max_a = 1 #m/s2
Acc_max_b = -1 #m/s2
A = 1.5 #kn
B = 0.006 #kn/(m/s)
C = 0.0067 #kn/(m2/s2)
Fb_Max = 80000 #N 这个可以导入实际数据来同样进行线性分段，如同PWL_SPE的处理方式
Ft_Max = 80000 #N
P_b_max = 600000 #W
P_t_max = 600000 #W
P_fc_max = 250000 #W
P_ESD_max = 400000 #W
n_m = 0.9 #motor efficiency
n_ESD = 0.88 #ESD efficiency
<<<<<<< HEAD
n_fc_max = 0.9 #maximum efficiency of fuel cell
=======
n_fc_max = 0.6 #maximum efficiency of fuel cell
>>>>>>> 537de9ac93f795124ed0a00b3291a402cad56c96
g = 9.8
v_max = 33 #mps
v_min = 0.1 #mps

i = list(range(1,N)) #1-40
ii = list(range(0,N)) #0-40

#piecewise linearisation accuracy(creat a speed list)
 #以下的是我自己的速度数据，现在开始用学长的速度数据,改的地方1.速度基本向量2.初始与结束速度的数值设定
delta_speed = v_max / (N_V - 1)
PWL_SPE = [v_min] #from 0 to 33,piece 35
pre = 0
for index in range(N_V-1):
    pre = pre + delta_speed
    PWL_SPE.append(pre)
'''
#speed limitation
v_limit = []
for index in range(0, 25):
    v_limit.append(70) #此时设定的速度的限制的数值导致公式（6）的速度限制是一个废的条件，后续改进
for index in range(25, 29):
    v_limit.append(50)
for index in range(29, N):
    v_limit.append(70)
'''
#energy storage parametewrs(不知道单位)
<<<<<<< HEAD
E_cap = 1000000*cap +1 #J
PESD = 400000 #w
=======
E_cap = 1000000*cap +1
PESD = 400000
>>>>>>> 537de9ac93f795124ed0a00b3291a402cad56c96
H_heat_value = 140000 #J/g

#modelling (this is based on time, the formulation is a little different to the model which is based on distance)
m = Model('hydrogen_power')
delta_d_i= m.addVars(i,lb=0, vtype=GRB.CONTINUOUS, name='Elapsed distance')
add_distance = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name='plus delta d')
H_slop = m.addVars(ii, lb=-100, vtype=GRB.CONTINUOUS, name='the slop/gradient')
delta_h = m.addVars(i, lb=-100, vtype=GRB.CONTINUOUS, name='the delta hight')

f_i_drag = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='Average drag force')

v_i = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name='speed')
v_i2 = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name='square of speed')
v_ave_i = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='Average speed')
v_ave_i2 = m.addVars(i, lb=0,vtype=GRB.CONTINUOUS, name='square of average speed')
v_ave_i3 = m.addVars(i, lb=0,vtype=GRB.CONTINUOUS, name='Cubic of average speed')
v_ave_i1d = m.addVars(i, lb=0,vtype=GRB.CONTINUOUS, name='1/average speed')
v_limit = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name='the limited speed')

E_i_seg = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Applied force')
E_i_drag = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='Average drag force energy')
E_store = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name='store')

#the former is basic model parameters, the latter is fuel cell hybrid parameters
E_i_fc = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='energy from fc')
E_i_dis = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='energy discharge to ESD')
E_i_ch = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='energy charge to ESD') #it is positive because in formulation there exist a minus sign
#E_init = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name="the initial energy of the ESD")#the initial is relative which is updated after every stage
E_store = m.addVars(ii, lb=0, vtype=GRB.CONTINUOUS, name='store')

P_i_ESD = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the net power of ESD')#this part P is corresponding to the mass, use P or M is same
P_i_fc = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the power of fc')

m_i_ESD = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the net hydrogen mass of ESD')
m_i_fc = m.addVars(i, lb=0, vtype=GRB.CONTINUOUS, name='the hydrogen mass from fc')
m_fc = m.addVar(vtype=GRB.CONTINUOUS, name='m_fc')
m_ESD = m.addVar(vtype=GRB.CONTINUOUS, name='m_ESD')

lambda_i = m.addVars(i, vtype=GRB.BINARY, name='to judge the energy comes back or need')
SOE_i = m.addVars(ii, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='the state of energy')
Cr_i_fc = m.addVars(i, vtype=GRB.CONTINUOUS, name='hydrogen consumption rate')
#n_i_fc = m.addVars(i, vtype=GRB.CONTINUOUS, name='the efficency of fc')

alpha = m.addVars(N, N_V, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='a')#N_V is the dimension of special speed at x for speed
beta = m.addVars(N-1, N_V, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='b')#N is the dimension of all the speed through the whole distance for average speed
#equation of SOS2 variable constrain
for index in ii:
    m.addSOS(GRB.SOS_TYPE2, [alpha[index, j] for j in range(N_V)])
for index in i:
    m.addSOS(GRB.SOS_TYPE2, [beta[index-1, j] for j in range(N_V)]) #beta的下标只能从0开始
m.addConstrs((alpha.sum(index, '*') == 1 for index in ii), name='SOS2 property_α ')
m.addConstrs((beta.sum(index-1, '*') == 1 for index in i), name='SOS2 property_β')

#sos2 express the velocity
for index in ii:
    m.addConstr(v_i[index] == (quicksum(PWL_SPE[j] * alpha[index, j] for j in range(N_V))), name='candidate speed')
    m.addConstr(v_i2[index] == (quicksum(PWL_SPE[j]**2 * alpha[index, j] for j in range(N_V))), name='candidate speed square')

for index in i:
    m.addConstr(v_ave_i[index] == (quicksum(beta[index-1, j] * PWL_SPE[j] for j in range(N_V))), name='candidate average speed')
    m.addConstr(v_ave_i2[index] == (quicksum(beta[index-1, j] * PWL_SPE[j]**2 for j in range(N_V))), name='candidate average speed square')
    m.addConstr(v_ave_i3[index] == (quicksum(beta[index - 1, j] * PWL_SPE[j] * PWL_SPE[j] * PWL_SPE[j] for j in range(N_V))), name='candidate average speed square')
    #m.addConstr(v_ave_i1d[index] == (quicksum(1 / PWL_SPE[j] * beta[index-1, j] for j in range(N_V))), name='candidate 1/average speed')

# creat the corresponding relationship between add_distance and v_limit and delta_h
m.addConstr(add_distance[0] == 0)
for index in i:
    m.addConstr(add_distance[index] == quicksum(delta_d_i[j] for j in range(1,index+1)))

#the limit for speed
for index in ii:
    m.addGenConstrPWL(add_distance[index], v_limit[index],  [0,4299,4300, 5700,5701, Distance], [50, 50, 20, 20, 60, 60], name='v_limit and distance relation')
    m.addConstr(v_i[index] <= v_limit[index])

#the limit for delta_H
for index in ii:
    m.addGenConstrPWL(add_distance[index], H_slop[index], [0, 2000, 3000, 4000, 8000,Distance], [0, 0, 2, 2, -2,-2], name='the gradient and distance relationship')
for index in i:
    m.addConstr(delta_h[index] == H_slop[index] - H_slop[index-1])


#Equation(1):travel distance(this model is based on time not on distance)
for index in i:
    m.addConstr(delta_d_i[index] == v_ave_i[index] * delta_t, name='For Δdi')
m.addConstr(quicksum(delta_d_i) == Distance, name='total distance')

#Equation(2):the average speed
for index in i:
    m.addConstr(v_ave_i[index] * 2 == v_i[index] + v_i[index - 1], name='average speed')

#Equation(4):Davis equation
for index in i:
    m.addConstr(f_i_drag[index] == 1000 * (A + B * v_ave_i[index] + C * v_ave_i2[index]), name='Davis equation')
    m.addConstr(E_i_drag[index] == 1000 * delta_t * (A * v_ave_i[index] + B * v_ave_i2[index] + C * v_ave_i3[index]), name='Davis equation energy')

#Equation(5): acceleration constrain
for index in i:
    m.addRange((v_i[index] - v_i[index-1]) / delta_t, Acc_max_b, Acc_max_a, name='acceleration constrain')

#Equation(6): velocity constrain
for index in range(0,N-1):
    m.addConstr(v_i[index] <= v_limit[index], name='velocity constrain upper')
    #m.addConstr(v_i[index] >= 0, name='velocity constrain lower')
m.addConstr(v_i[0] == 0.1, name="起点")
m.addConstr(v_i[N-1] == 0.1, name="终点")

#Equation(7): conservation of energy（未考虑能量来源的效率系数，后续改进添加）
for index in i:
    m.addConstr(E_i_seg[index]*n_m - M_Total * g * delta_h[index] - 0.5 * M_Total * (v_i2[index] - v_i2[index-1]) - E_i_drag[index] >= 0, name='conservation of energy1')
    m.addConstr(E_i_seg[index]/n_m - M_Total * g * delta_h[index] - 0.5 * M_Total * (v_i2[index] - v_i2[index-1]) - E_i_drag[index] >= 0, name='conservation of energy2')



#Equation(9):energy constrain
for index in i:
    m.addConstr(E_i_seg[index] >= -Fb_Max * delta_d_i[index] * n_m, name='braking fore constrain')
    m.addConstr(E_i_seg[index] <= Ft_Max * delta_d_i[index]/n_m, name='traction force constrain')#using divide because the obtain force is smaller than the power grid provided
    m.addConstr(E_i_seg[index] >= -P_b_max * delta_t * n_m, name='braking power constrain')
    m.addConstr(E_i_seg[index] <= P_t_max * delta_t / n_m, name='traction power constrain')

#energy sum
E_total = quicksum(E_i_seg)

#iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
for index in i:
    m.addConstr(
        E_i_seg[index] <= lambda_i[index] * (E_i_fc[index] + E_i_dis[index] * n_ESD) - (1 - lambda_i[index]) * (E_i_ch[
            index] / n_ESD)) #the total energy distribution
    m.addConstr(E_i_ch[index] <= (1-lambda_i[index]) * 100000000000) #charge and dischage only one can exist
    m.addConstr(E_i_fc[index] <= lambda_i[index] * 100000000000)
    m.addConstr(E_i_dis[index] <= lambda_i[index] * 100000000000)

for index in i:
    m.addConstr(E_i_fc[index] <= P_fc_max * delta_t )#the distributed energy constrain
    m.addConstr(E_i_ch[index] <= PESD * delta_t )
    m.addConstr(E_i_dis[index] <= PESD * delta_t )

#Equation(13) SOE expression (0-1 constrain has already written in the definition of SOE)
E_init = E_cap
m.addConstr(E_store[0] == E_init)
for index in range(1, N):
    m.addConstr(E_store[index] == E_store[index-1] + E_i_ch[index] - E_i_dis[index])
    m.addConstr(E_store[index] <= E_cap)
for index in range(0,N):
    m.addConstr(SOE_i[index] == E_store[index] / E_cap)

#Equation(14) build the connection between Cri,fc and Pi,fc and calculate the m_fc
for index in i:
    m.addConstr(P_i_fc[index] == E_i_fc[index] /(delta_t * 1000))
    m.addGenConstrPWL(P_i_fc[index], Cr_i_fc[index], [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
                      [0, 0.3571, 0.64935, 0.931677, 1.19, 1.5133, 1.84729, 2.19298, 2.5975, 3.032345, 3.571428],
                      name='Power-hydrogen_comsuption_Characteristic')
    '''
    m.addGenConstrPWL(P_i_fc[index], n_i_fc[index], [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
                      [0, 70.0, 77.0, 80.5, 84.0, 82.6, 81.20, 79.0, 76.9, 74.2, 70.0],
                      name='Power-Efficiency_Characteristic')
    '''
    m.addConstr(m_i_fc[index] == Cr_i_fc[index] * delta_t)
m.addConstr(m_fc == quicksum(m_i_fc))

#Equation(15) calculate the m_ESD
<<<<<<< HEAD
m.addConstr(m_ESD == ((1 - SOE_i[N-1]) * E_cap) / n_fc_max / H_heat_value) #g
=======
m_ESD = ((1 - SOE_i[N-1]) * E_cap) /n_fc_max/H_heat_value #g
>>>>>>> 537de9ac93f795124ed0a00b3291a402cad56c96

#objective function
obj = m_fc + m_ESD #g

#iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii




#objective function
m.setObjective(obj, GRB.MINIMIZE)
m.Params.MIPGap = 0.2
m.optimize()

#plot the graph function
def plotspeed():
    v_point = []
    Time_plot = []
    v_limit_plot = []
    H_slop_plot = []
    for index in ii:
        v_point.append(v_i[index].x * 3.6)
        v_limit_plot.append(v_limit[index].x * 3.6)
        H_slop_plot.append(H_slop[index].x)
        Time_plot.append(delta_t * index)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(0, Time_total)
    ax1.plot(Time_plot, v_point, label='Speed Trajectory',color='black')
    ax1.plot(Time_plot, v_limit_plot, label='Speed limit', color='red', linestyle='--')
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Speed(km/h)")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.plot(Time_plot, H_slop_plot, label='slop', color='y',linestyle='-.')
    ax2.fill_between(x=Time_plot, y1=-10, y2=H_slop_plot, facecolor='grey', alpha=0.3)
    ax2.set_ylabel("Height(m)",color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.set_ylim(-10,50)
    ax1.grid()
    ax1.legend(loc='upper right')
    plt.show()

plotspeed()

def power_plot_function(): #注意时间和功率的维度问题，将第一个功率数据copy一份添加一个维度
    P_fc_plot = [E_i_fc[1].x / delta_t / 1000]
    Time_plot = [0]
    P_ESD_plot = [(E_i_dis[1].x - E_i_ch[1].x) / delta_t / 1000]
    P_seg_plot = [E_i_seg[1].x/delta_t/1000]
    P_ch_plot = [-E_i_ch[1].x/delta_t/1000]
    P_dis_plot = [E_i_dis[1].x / delta_t / 1000]
    E_seg_calculate = []
    for index in i: #1-40
        P_fc_plot.append(E_i_fc[index].x / delta_t / 1000)
        P_ESD_plot.append((E_i_dis[index].x - E_i_ch[index].x) / delta_t / 1000)
        P_seg_plot.append(E_i_seg[index].x/delta_t/1000)#两者是等价的P_seg_plot2.append((E_i_fc[index].x + E_i_dis[index].x * n_ESD - E_i_ch[index].x / n_ESD)/delta_t/1000)
        E_seg_calculate.append(E_i_seg[index].x)
        P_ch_plot.append(-E_i_ch[index].x/delta_t/1000)
        P_dis_plot.append(E_i_dis[index].x / delta_t / 1000)
        Time_plot.append((index+1) * delta_t)
    fig = plt.figure(2)
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(0, Time_total)
    ax1.step(Time_plot, P_fc_plot, color = 'b', label="FC power")
    #ax1.step(Time_plot, P_ESD_plot, label="ESD power")# it is covered by charge and discharge lines in graph
    ax1.step(Time_plot, P_seg_plot, color = 'm',label="the required/get power")
    ax1.step(Time_plot, P_ch_plot, color = 'g', label="ESD charge power")
    ax1.step(Time_plot, P_dis_plot, color = 'c', label="ESD discharge power")
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Power(kw)")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylim(0,100)
    ax1.grid()
    print('The total energy is ',(sum(E_seg_calculate)))
    #plt.show()
def SOC_plot_function():
    SOE = []
    Time_plot = []
    for index in ii:
        SOE.append(SOE_i[index].x)
        Time_plot.append(index * delta_t)
    fig = plt.figure(3)
    plt.xlim(0, Time_total)
    plt.plot(Time_plot, SOE)
    plt.xlabel("Time(s)")
    plt.ylabel("SOE")
    plt.grid()
    plt.show()
def Cr_plot():
    Time_plot = [0]
    Cr_fc_plot = [Cr_i_fc[1].x]
    m_i_fc_plot = [m_i_fc[1].x]
    for index in i:
        Time_plot.append(index * delta_t)
        #n_fc_plot.append(P_i_fc[index].x/Cr_i_fc[index].x/H_heat_value)
        Cr_fc_plot.append(Cr_i_fc[index].x)
        m_i_fc_plot.append(m_i_fc[index].x)
    fig = plt.figure(4)
    plt.xlim(0, Time_total)
    #plt.plot(Time_plot, Cr_fc_plot)
    plt.plot(Time_plot, m_i_fc_plot)
    plt.xlabel("Time(s)")
    plt.ylabel("Cr(g/t)")
    plt.grid()
    plt.show()
def n_fc_plot():
    Time_plot = [0]
    P_fc_plot = [E_i_fc[1].x / delta_t / 1000]
    n_fc_num = [E_i_fc[1].x / m_i_fc[1].x / 1000]
    for index in i:
        if m_i_fc[index].x != 0:
            n_fc_num.append(E_i_fc[index].x / m_i_fc[index].x / 1000)
        if m_i_fc[index].x == 0:
            n_fc_num.append(20)
        Time_plot.append(index * delta_t)
        P_fc_plot.append(E_i_fc[index].x / delta_t / 1000)
    fig = plt.figure(5)
    ax1 = fig.add_subplot(111)
    plot1 = ax1.step(Time_plot, n_fc_num, label='Efficiency of fc')
    ax1.set_xlim(0, Time_total)
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Fuel Cell efficiency(%)")
    ax1.grid()
    ax2 = ax1.twinx()
    plot2 = ax2.step(Time_plot, P_fc_plot, color="orange", label="FC power")
    ax2.set_ylabel("Power(kw)", color='orange')
    ax2.tick_params(axis='y', colors='orange')
    # 画lengend
    lines = plot1 + plot2
    ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(1, 1))
    plt.show()
'''
def force_plot():
    F_seg_plot = [E_i_seg[1].x / delta_d_i[1].x / 1000]
    Time_plot = [0]
    for index in i:
        Time_plot.append((index + 1) * delta_t)
        F_seg_plot.append(E_i_seg[index].x / delta_d_i[index].x / 1000)
    plt.step(Time_plot, F_seg_plot)
    plt.show()
force_plot()
'''



n_fc_plot()
Cr_plot()
power_plot_function()
SOC_plot_function()
<<<<<<< HEAD
print('the hydrogen consumption is ',(m_fc.x+((1 - SOE_i[N-1].x) * E_cap) / H_heat_value / n_fc_max),'based on time')


#calculate the total energy
E_seg_calculate = []
for index in i:
    E_seg_calculate.append(E_i_seg[index].x)
print('The total energy is ',(sum(E_seg_calculate)),'based on time')
=======
print('the hydrogen consumption is ',(m_fc.x+((1 - SOE_i[N-1].x) * E_cap) / H_heat_value / n_fc_max))
>>>>>>> 537de9ac93f795124ed0a00b3291a402cad56c96
