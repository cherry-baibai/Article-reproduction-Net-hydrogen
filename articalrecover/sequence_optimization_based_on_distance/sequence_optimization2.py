from gurobipy import *
import matplotlib.pyplot as plt
import xlrd

#读取第一步优化而获得的数据
#attention,最开始E_i_seg是从1到100，现在它的标号变成了0到99
def read_excel_xls(path):
    parameter_numbers_ave = 6
    v_ave_i = []
    v_ave_i1d = []
    v_ave_i2 = []
    delta_t_i = []
    f_i_drag = []
    E_i_seg = []
    v_i = []
    v_i2 = []
    workbook = xlrd.open_workbook(path) #打开工作簿
    sheet = workbook.sheet_names() #获取工作簿中所有的表格
    worksheet = workbook.sheet_by_name(sheet[0])#获取所有表格中的第一个表格
    for i in range(0, parameter_numbers_ave):
        for j in range(1, worksheet.ncols-1):
            if i == 0:
                v_ave_i.append(worksheet.cell_value(i,j))
            if i == 1:
                v_ave_i1d.append(worksheet.cell_value(i,j))
            if i == 2:
                v_ave_i2.append(worksheet.cell_value(i,j))
            if i == 3:
                delta_t_i.append(worksheet.cell_value(i, j))
            if i == 4:
                f_i_drag.append(worksheet.cell_value(i, j))
            if i == 5:
                E_i_seg.append(worksheet.cell_value(i, j))
    for i in range(parameter_numbers_ave, worksheet.nrows):
        for j in range(1, worksheet.ncols):
            if i == 6:
                v_i.append(worksheet.cell_value(i, j))
            if i == 7:
                v_i2.append(worksheet.cell_value(i, j))
    return v_ave_i,v_ave_i1d,v_ave_i2,delta_t_i,f_i_drag,E_i_seg, v_i, v_i2
v_ave_i, v_ave_i1d, v_ave_i2, delta_t_i, f_i_drag, E_i_seg, v_i, v_i2 = read_excel_xls('the_first_optimal_consequence.xls')

#import the parameters information
Distance = 1800 #m
Time_total = 130 #s
H = 0 #don't consider the gradient temporary
cap = 40
M_Total = 72700 + 50*cap #kg
N = 41
N_V = 35 #speed is divided into N_v(used in alpha/beta)
delta_d = int(Distance/(N-1))
#delta_t = Time_total/(N-1)
delta_h = 0
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
n_fc_max = 0.84 #maximum efficiency of fuel cell
g = 9.8
v_max = 33 #mps
v_min = 0.1 #mps

i = list(range(0,N-1)) #0-39 len=40
ii = list(range(0,N)) #0-40 len=41

#energy storage parametewrs(不知道单位)
E_cap = 1000000*cap +1
PESD = 400000 #it is Pd_max and Pc_max (to constrain the distributed energy)
H_heat_value = 100000 #kJ/kg
#modelling (this is based on time, the formulation is a little different to the model which is based on distance)
m = Model('hydrogen_power')

#delta_d_i= m.addVars(i, vtype=GRB.CONTINUOUS, name='Elapsed distance')

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

lambda_i = m.addVars(i, vtype=GRB.BINARY, name='to judge the energy comes back or need')
SOE_i = m.addVars(ii, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='the state of energy')
Cr_i_fc = m.addVars(i, vtype=GRB.CONTINUOUS, name='hydrogen consumption rate')

for index in i:
    m.addConstr(
        E_i_seg[index] <= lambda_i[index] * (E_i_fc[index] + E_i_dis[index] * n_ESD) - (1 - lambda_i[index]) * (E_i_ch[
            index] / n_ESD)) #the total energy distribution
    m.addConstr(E_i_ch[index] <= (1-lambda_i[index]) * 100000000000) #charge and dischage only one can exist
    m.addConstr(E_i_fc[index] <= lambda_i[index] * 100000000000)
    m.addConstr(E_i_dis[index] <= lambda_i[index] * 100000000000)

for index in i:
    m.addConstr(E_i_fc[index] <= P_fc_max * delta_t_i[index] * 1000)#the distributed energy constrain
    m.addConstr(E_i_ch[index] <= PESD * delta_t_i[index] * 1000)
    m.addConstr(E_i_dis[index] <= PESD * delta_t_i[index] * 1000)

#Equation(13) SOE expression (0-1 constrain has already written in the definition of SOE)
E_init = E_cap
m.addConstr(E_store[0] == E_init)
for index in range(1, N):
    m.addConstr(E_store[index] == E_store[index-1] + E_i_ch[index-1] - E_i_dis[index-1])
    m.addConstr(E_store[index] <= E_cap)
for index in range(0,N):
    m.addConstr(SOE_i[index] == E_store[index] / E_cap)

#Equation(14) build the connection between Cri,fc and Pi,fc and calculate the m_fc
for index in i:
    m.addConstr(P_i_fc[index] == E_i_fc[index] /(delta_t_i[index] * 1000))
    m.addGenConstrPWL(P_i_fc[index], Cr_i_fc[index], [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
                      [0, 0.3571, 0.64935, 0.931677, 1.19, 1.5133, 1.84729, 2.19298, 2.5975, 3.032345, 3.571428],
                      name='Power-Efficiency_Characteristic')
    m.addConstr(m_i_fc[index] == Cr_i_fc[index] * delta_t_i[index])
m.addConstr(m_fc == quicksum(m_i_fc))

#Equation(15) calculate the m_ESD
m_ESD = ((1 - SOE_i[N-1]) * E_cap) / H_heat_value / n_fc_max

#objective function
obj = m_fc + m_ESD
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
'''
m.computeIIS()
m.write("model1.ilp")
'''
#plot the graph define the plot function
'''
Energy_from_fuel_cell = []
for index in range(0,N-1):
    Energy_from_fuel_cell.append(E_i_fc[index].x)
'''
def plotspeed(ax2):
    v_point = []
    distance_plot = []
    for index in ii:
        v_point.append(v_i[index] * 3.6)
        distance_plot.append(delta_d * index)
    ax2.set_xlim(0, Distance)
    ax2.plot(distance_plot, v_point, 'r', label='Speed Trajectory')
    ax2.set_xlabel("Distance(m)")
    ax2.set_ylabel("Speed(km/h)", color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.grid()
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 100)
    plt.show()

def power_plot_function():
    P_fc_plot = [E_i_fc[0].x / delta_t_i[0] / 1000]
    distance_plot = [0]
    P_ESD_plot = [(E_i_dis[0].x - E_i_ch[0].x) / delta_t_i[0] / 1000]
    P_seg_plot = [E_i_seg[0] / delta_t_i[0] / 1000]
    P_ch_plot = [-E_i_ch[0].x / delta_t_i[0] / 1000]
    P_dis_plot = [E_i_dis[0].x / delta_t_i[0] / 1000]
    for index in i: #0-39
        P_fc_plot.append(E_i_fc[index].x / delta_t_i[index] / 1000)
        P_ESD_plot.append((E_i_dis[index].x - E_i_ch[index].x) / delta_t_i[index] / 1000)
        P_seg_plot.append(E_i_seg[index] / delta_t_i[index] / 1000)  # 两者是等价的P_seg_plot2.append((E_i_fc[index].x + E_i_dis[index].x * n_ESD - E_i_ch[index].x / n_ESD)/delta_t/1000)
        P_ch_plot.append(-E_i_ch[index].x / delta_t_i[index] / 1000)
        P_dis_plot.append(E_i_dis[index].x / delta_t_i[index] / 1000)
        distance_plot.append((index+1) * delta_d)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(0, Distance)
    ax1.step(distance_plot, P_fc_plot,color = 'b', label="FC power")
    #plt.step(distance_plot, P_ESD_plot, label="ESD power")
    ax1.step(distance_plot, P_seg_plot, color='m', label="the required/get power")
    ax1.step(distance_plot, P_ch_plot, color='g', label="ESD charge power")
    ax1.step(distance_plot, P_dis_plot, color='c', label="ESD discharge power")

    ax1.set_xlabel("Distance(m)")
    ax1.set_ylabel("Power(kw)")
    ax1.legend()
    ax1.grid()
    ax2 = ax1.twinx()
    plotspeed(ax2)
    plt.show()


def SOC_plot_function():
    SOE = []
    distance_plot = []
    for index in ii:
        SOE.append(SOE_i[index].x)
        distance_plot.append(index * delta_d)
    plt.figure()
    plt.xlim(0, Distance)
    plt.plot(distance_plot, SOE)
    plt.xlabel("Distance(m)")
    plt.ylabel("SOE")
    plt.grid()
    plt.show()

def Cr_plot():
    distance_plot = []
    Cr_fc_plot = [Cr_i_fc[0].x]
    for index in ii:
        distance_plot.append(index * delta_d)
    for index in i:
        Cr_fc_plot.append(Cr_i_fc[index].x)
    plt.xlim(0, Distance)
    plt.plot(distance_plot, Cr_fc_plot)
    plt.xlabel("Distance(m)")
    plt.ylabel("Cr(kg/t)")
    plt.grid()
    plt.show()

def n_fc_plot():
    distance_plot = []
    n_fc_num = [E_i_fc[1].x / m_i_fc[1].x / 1000]
    P_fc_plot = [E_i_fc[0].x / delta_t_i[0] / 1000]
    for index in ii:
        distance_plot.append(index * delta_d)
    for index in i:
        if m_i_fc[index].x != 0:
            n_fc_num.append(E_i_fc[index].x / m_i_fc[index].x /1000)
        if m_i_fc[index].x == 0:
            n_fc_num.append(20)
        P_fc_plot.append(E_i_fc[index].x / delta_t_i[index] / 1000)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot1 = ax1.step(distance_plot, n_fc_num, label='Efficiency of fc')
    ax1.set_xlim(0, Distance)
    ax1.set_ylim(15, 90)
    ax1.set_xlabel("Distance(m)")
    ax1.set_ylabel("Fuel Cell efficiency(%)")
    ax1.grid()

    ax2 = ax1.twinx()
    plot2 = ax2.step(distance_plot, P_fc_plot, color="orange", label="FC power")
    ax2.set_ylabel("Power(kw)", color='orange')
    ax2.tick_params(axis='y', colors='orange')
    #画lengend
    lines = plot1 + plot2
    ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(1,1))

    plt.show()
n_fc_plot()

Cr_plot()


power_plot_function()
SOC_plot_function()
print('the hydrogen consumption is ',(m_fc.x+((1 - SOE_i[N-1].x) * E_cap) / H_heat_value / n_fc_max))