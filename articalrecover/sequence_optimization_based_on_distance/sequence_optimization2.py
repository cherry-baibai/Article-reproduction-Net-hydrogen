from gurobipy import *
import matplotlib.pyplot as plt
import xlrd

#读取第一步优化而获得的数据
def read_excel_xls(path):
    v_ave_i = []
    v_ave_i1d = []
    v_ave_i2 = []
    delta_t_i = []
    f_i_drag = []
    E_i_seg = []
    workbook = xlrd.open_workbook(path) #打开工作簿
    sheet = workbook.sheet_names() #获取工作簿中所有的表格
    worksheet = workbook.sheet_by_name(sheet[0])#获取所有表格中的第一个表格
    for i in range(0, worksheet.nrows):
        for j in range(1, worksheet.ncols):
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

read_excel_xls('the_first_optimal_consequence.xls')