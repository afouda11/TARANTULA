import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

state_label = {'2p1s3pz':'IBr[3$d_{Br}^{1}$,$\sigma_{pz}^{1}$]$^{+2}$',
		 '2p2t3pz':'IBr[3$d_{Br}^{1}$,$\sigma_{s}^{1}$]$^{+2}$', 
         '3p1d1pz':'IBr[$\sigma_{s}^{1}$,$\sigma_{pz}^{1}$,$\pi_{px}^{1}$]$^{+3}$',          
         '3p2d1pz':'IBr[$\sigma_{s}^{2}$,$\sigma_{pz}^{1}$]$^{+3}$',
         '1p': 'IBr[3$p_{Br}^{1}$]$^{+1}$'} 

data = {'population':{}, 'bondlength':{}, 'energy':{}, 'kenergy':{}, '0_charge':{}, '1_charge':{}, '0_spin':{}, '1_spin':{}}

spawn_start = np.loadtxt('spawn_start.txt')

for i in data:
	for j in range(5):
		#print(j)
		data[i][j] = np.loadtxt('outputs/'+i+'_'+str(j)+'.txt')
		#print(np.shape(data[i][j]))

total_pop = {}
for j in range(1,3):
    total_pop[j] = []
    for k in range(1,21):
        if k == 1:
            total_pop[j]  = data['population'][j][:,k]
        else:
            total_pop[j] += data['population'][j][:,k]
for j in range(3,5):
    total_pop[j] = []
    for k in range(1,191):
        if k == 1:
            total_pop[j]  = data['population'][j][:,k]
        else:
            total_pop[j] += data['population'][j][:,k]

au2fs = 0.02418884254
au2ev = 27.2114

time = data['population'][0][:,0]

color_1 = '#7fbf7b'
color_2 = '#af8dc3'
color_3 = '#66FF00'
color_4 = '#FF007F'

t_spawn    = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000,
		      55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

spawn_start = [10000] 
for i in range(2,20):
	spawn_start = np.append(spawn_start, np.repeat(t_spawn[i], i))
# 			np.repeat(15000, 2), np.repeat(20000, 3), np.repeat(25000, 4), 
# 			   np.repeat(30000, 5),  np.repeat(35000, 6),  np.repeat(40000, 7),  np.repeat(45000, 8),  np.repeat(50000, 9),
# 		       np.repeat(55000, 10), np.repeat(60000, 11), np.repeat(65000, 12), np.repeat(70000, 13), np.repeat(75000, 14), 
# 			   np.repeat(80000, 15), np.repeat(85000, 16), np.repeat(90000, 17), np.repeat(95000, 18), np.repeat(100000, 19)]
# 
# spawn_start = np.array([spawn_start])
print(len(spawn_start))

state_Br_charge = {}
state_I_charge  = {}
total_state_charge  = {}
total_Br_charge = []
total_I_charge  = []
total_charge    = []

state_Br_spin = {}
state_I_spin  = {}
total_state_spin  = {}
total_Br_spin = []
total_I_spin  = []
total_spin    = []

state_kenergy    = {}
state_bondlength = {}
total_kenergy    = []
total_bondlength = []

MEDIUM_SIZE = 14
BIGGER_SIZE = 14.5
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(4,2,sharex=True)
fig.set_figwidth(12)
fig.set_figheight(14)

ax1.plot(time*au2fs, data['bondlength'][0][:,1], linewidth=1.3, color='k', linestyle='-',  label=state_label['1p'])
ax2.plot(time*au2fs, data['bondlength'][0][:,1], linewidth=1.3, color='k', linestyle='-',  label=state_label['1p'])

for i in range(1,21):
    if i == 1:
        ax1.plot(time[t_spawn[i-1]:]*au2fs, data['bondlength'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='-', label=state_label['2p1s3pz'])
        ax2.plot(time[t_spawn[i-1]:]*au2fs, data['bondlength'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='-', label=state_label['2p2t3pz'])
    if i > 1:
        ax1.plot(time[t_spawn[i-1]:]*au2fs, data['bondlength'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='-')
        ax2.plot(time[t_spawn[i-1]:]*au2fs, data['bondlength'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='-')


for i in range(1,191):
	if i == 1:          
		ax1.plot(time[spawn_start[i-1]:]*au2fs, data['bondlength'][3][spawn_start[i-1]:,i], linewidth=1.3, color=color_3, linestyle=':',   label=state_label['3p1d1pz'])
		ax2.plot(time[spawn_start[i-1]:]*au2fs, data['bondlength'][4][spawn_start[i-1]:,i], linewidth=1.3, color=color_4, linestyle=':',   label=state_label['3p2d1pz'])
	else:
		ax1.plot(time[spawn_start[i-1]:]*au2fs, data['bondlength'][3][spawn_start[i-1]:,i], linewidth=1.3, color=color_3, linestyle=':')         
		ax2.plot(time[spawn_start[i-1]:]*au2fs, data['bondlength'][4][spawn_start[i-1]:,i], linewidth=1.3, color=color_4, linestyle=':')
#         ax1.plot(time*au2fs, data['bondlength'][3][:,i], linewidth=1.3, color=color_3, linestyle=':')
#         ax2.plot(time*au2fs, data['bondlength'][4][:,i], linewidth=1.3, color=color_4, linestyle=':')



ax3.plot(time*au2fs, data['kenergy'][0][:,1]*au2ev, linewidth=1.3, color='k', linestyle='-',  label=state_label['1p'])
ax4.plot(time*au2fs, data['kenergy'][0][:,1]*au2ev, linewidth=1.3, color='k', linestyle='-',  label=state_label['1p'])

for i in range(1,21):
    if i == 1:
        ax3.plot(time[t_spawn[i-1]:]*au2fs, data['kenergy'][1][t_spawn[i-1]:,i]*au2ev, linewidth=1.3, color=color_1, linestyle='-', label=state_label['2p1s3pz'])
        ax4.plot(time[t_spawn[i-1]:]*au2fs, data['kenergy'][2][t_spawn[i-1]:,i]*au2ev, linewidth=1.3, color=color_2, linestyle='-', label=state_label['2p2t3pz'])
    if i > 1:
        ax3.plot(time[t_spawn[i-1]:]*au2fs, data['kenergy'][1][t_spawn[i-1]:,i]*au2ev, linewidth=1.3, color=color_1, linestyle='-')
        ax4.plot(time[t_spawn[i-1]:]*au2fs, data['kenergy'][2][t_spawn[i-1]:,i]*au2ev, linewidth=1.3, color=color_2, linestyle='-')


for i in range(1,191):
    if i == 1:          
        ax3.plot(time[int(spawn_start[i-1]):]*au2fs, data['kenergy'][3][int(spawn_start[i-1]):,i]*au2ev, linewidth=1.3, color=color_3, linestyle=':',   label=state_label['3p1d1pz'])
        ax4.plot(time[int(spawn_start[i-1]):]*au2fs, data['kenergy'][4][int(spawn_start[i-1]):,i]*au2ev, linewidth=1.3, color=color_4, linestyle=':',   label=state_label['3p2d1pz'])
    else:
        ax3.plot(time[int(spawn_start[i-1]):]*au2fs, data['kenergy'][3][int(spawn_start[i-1]):,i]*au2ev, linewidth=1.3, color=color_3, linestyle=':')
        ax4.plot(time[int(spawn_start[i-1]):]*au2fs, data['kenergy'][4][int(spawn_start[i-1]):,i]*au2ev, linewidth=1.3, color=color_4, linestyle=':')

for i in range(1,21):
    if i == 1:
        ax5.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='-', label=state_label['2p1s3pz'])
        ax6.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='-', label=state_label['2p2t3pz'])
    if i > 1:
        ax5.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='-')
        ax6.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='-')
for i in range(1,191):
    if i == 1:          
        ax5.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_3, linestyle=':', label=state_label['3p1d1pz'])
        ax6.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_4, linestyle=':', label=state_label['3p2d1pz'])
    else:                                                    
        ax5.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_3, linestyle=':')
        ax6.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_4, linestyle=':')


for i in range(1,21):
    if i == 1:
        ax7.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='-', label=state_label['2p1s3pz'])
        ax8.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='-', label=state_label['2p2t3pz'])
    if i > 1:
        ax7.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='-')
        ax8.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='-')
for i in range(1,191):
    if i == 1:          
        ax7.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_3, linestyle=':', label=state_label['3p1d1pz'])
        ax8.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_4, linestyle=':', label=state_label['3p2d1pz'])
    else:                                                    
        ax7.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_3, linestyle=':')
        ax8.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_4, linestyle=':')


trans = mtransforms.ScaledTranslation(-20/72, 7/200, fig.dpi_scale_trans)
ax1.text(0.0, 1.0, 'a)', transform=ax1.transAxes + trans)
ax3.text(0.0, 1.0, 'b)', transform=ax3.transAxes + trans)
ax5.text(0.0, 1.0, 'c)', transform=ax5.transAxes + trans)
ax7.text(0.0, 1.0, 'd)', transform=ax7.transAxes + trans)

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")
#ax5.legend(loc="center right")
#ax6.legend(loc="center right")

ax1.set_ylabel("Bond length ($\AA$)")
ax3.set_ylabel("Kinetic Energy eV")
ax5.set_ylabel("Br Charge")
ax7.set_ylabel("I Charge")

ax1.set_title("Channel 1")
ax2.set_title("Channel 2")
ax7.set_xlabel("Time (fs)")
ax8.set_xlabel("Time (fs)")
plt.tight_layout()
plt.savefig("pngs/spawn_all.png",dpi=300)

# fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2,sharex=False,sharey=False)
# fig.set_figwidth(10)
# fig.set_figheight(8.5)
# for i in range(1,11):
#     if i == 1:
#         ax1.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='--', label=state_label['2p1s3pz']+" Br")
#         ax1.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle=':', label=state_label['2p1s3pz']+" I")
#         ax3.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='--', label=state_label['2p2t3pz']+" Br")
#         ax3.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle=':', label=state_label['2p2t3pz']+" I")
#     if i > 1:
#         ax1.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle='--')
#         ax1.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][1][t_spawn[i-1]:,i], linewidth=1.3, color=color_1, linestyle=':')
#         ax3.plot(time[t_spawn[i-1]:]*au2fs, data['0_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle='--')
#         ax3.plot(time[t_spawn[i-1]:]*au2fs, data['1_charge'][2][t_spawn[i-1]:,i], linewidth=1.3, color=color_2, linestyle=':')
# for i in range(1,46):
#     if i == 1:          
#         ax2.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_1, linestyle='--', label=state_label['3p1d1pz']+" Br")
#         ax2.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_1, linestyle=':', label=state_label['3p1d1pz']+" I")
#         ax4.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_2, linestyle='--', label=state_label['3p2d1pz']+" Br")
#         ax4.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_2, linestyle=':', label=state_label['3p2d1pz']+" I")
#     else:                                                    
#         ax2.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_1, linestyle='--')
#         ax2.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][3][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_1, linestyle=':')
#         ax4.plot(time[int(spawn_start[i-1]):]*au2fs, data['0_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_2, linestyle='--')
#         ax4.plot(time[int(spawn_start[i-1]):]*au2fs, data['1_charge'][4][int(spawn_start[i-1]):,i], linewidth=1.3, color=color_2, linestyle=':')
# ax1.set_ylabel("Atomic Charge")
# ax3.set_ylabel("Atomic Charge")
# ax1.set_xlabel("Time (fs)")
# ax2.set_xlabel("Time (fs)")
# ax3.set_xlabel("Time (fs)")
# ax4.set_xlabel("Time (fs)")
# ax1.legend(loc="center left")
# ax2.legend(loc="center left")
# ax3.legend(loc="center right")
# ax4.legend(loc="center right")
# plt.tight_layout()
# plt.savefig("pngs/charges_full_2.png",dpi=300)

                 #1  2  3  4   5   6   7   8   9   10  11  12  13  14   15   16   17   18 
# second_step = [1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106, 121, 137, 154, 172]
# 
# second_step_sum = np.zeros(100001)
# 
# for j in range(5):
#     if j == 0:
# 
#         state_Br_charge[j]    = data['0_charge'][j][:,1] 
#         state_I_charge[j]     = data['1_charge'][j][:,1] 
#         total_state_charge[j] = state_Br_charge[j] + state_I_charge[j]
#         state_Br_spin[j]    = data['0_spin'][j][:,1] 
#         state_I_spin[j]     = data['1_spin'][j][:,1] 
#         total_state_spin[j] = state_Br_spin[j] + state_I_spin[j]
#         state_kenergy[j]      = data['kenergy'][j][:,1]     
#         state_bondlength[j]   = data['bondlength'][j][:,1] 
# 
#     if j == 1 or j == 2:
#         for k in range(1,11):
#             if k == 1:
#                 total_state_charge[j] = (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_charge[j]    = data['0_charge'][j][:,k] * data['population'][j][:,k]
#                 state_I_charge[j]     = data['1_charge'][j][:,k] * data['population'][j][:,k]
# 
#                 total_state_spin[j] = (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_spin[j]    = data['0_spin'][j][:,k] * data['population'][j][:,k]
#                 state_I_spin[j]     = data['1_spin'][j][:,k] * data['population'][j][:,k]
# 
#                 state_kenergy[j]     = data['kenergy'][j][:,k] * data['population'][j][:,k] 
#                 state_bondlength[j]  = data['bondlength'][j][:,k] * data['population'][j][:,k]
# 
#             else:
#                 total_state_charge[j] += (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_charge[j]  += (data['0_charge'][j][:,k] * data['population'][j][:,k])
#                 state_I_charge[j]   += (data['1_charge'][j][:,k] * data['population'][j][:,k])
# 
#                 total_state_spin[j] += (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_spin[j]  += (data['0_spin'][j][:,k] * data['population'][j][:,k])
#                 state_I_spin[j]   += (data['1_spin'][j][:,k] * data['population'][j][:,k])
# 
#                 state_kenergy[j]    += (data['kenergy'][j][:,k] * data['population'][j][:,k]) 
#                 state_bondlength[j] += (data['bondlength'][j][:,k] * data['population'][j][:,k])
# 
# 
#     if j == 3 or j == 4:
#         for k in second_step:
#             if k == 1:
#                 total_state_charge[j] = (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_charge[j]    = data['0_charge'][j][:,k] * data['population'][j][:,k]
#                 state_I_charge[j]     = data['1_charge'][j][:,k] * data['population'][j][:,k]
# 
#                 total_state_spin[j] = (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_spin[j]    = data['0_spin'][j][:,k] * data['population'][j][:,k]
#                 state_I_spin[j]     = data['1_spin'][j][:,k] * data['population'][j][:,k]
# 
#                 state_kenergy[j]      = data['kenergy'][j][:,k] * data['population'][j][:,k] 
#                 state_bondlength[j]   = data['bondlength'][j][:,k] * data['population'][j][:,k]
# 
#             else:
#                 total_state_charge[j] += (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_charge[j]    += (data['0_charge'][j][:,k] * data['population'][j][:,k])
#                 state_I_charge[j]     += (data['1_charge'][j][:,k] * data['population'][j][:,k])
# 
#                 total_state_spin[j] += (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
#                 state_Br_spin[j]    += (data['0_spin'][j][:,k] * data['population'][j][:,k])
#                 state_I_spin[j]     += (data['1_spin'][j][:,k] * data['population'][j][:,k])
# 
#                 state_kenergy[j]      += (data['kenergy'][j][:,k] * data['population'][j][:,k]) 
#                 state_bondlength[j]   += (data['bondlength'][j][:,k] * data['population'][j][:,k])
# 
# 
# for j in range(1,5):
#     
#     if j == 1 or j== 2:
#         total_state_charge[j] /=  total_pop[j]
#         state_Br_charge[j] /= total_pop[j]
#         state_I_charge[j]  /= total_pop[j]
#         total_state_spin[j] /=  total_pop[j]
#         state_Br_spin[j] /= total_pop[j]
#         state_I_spin[j]  /= total_pop[j]
#         state_kenergy[j]    /= total_pop[j]
#         state_bondlength[j] /= total_pop[j]
#     if j == 3 or j== 4:
#         for i in second_step:
#             second_step_sum += data['population'][j][:,i]
#         #second_step_sum = (data['population'][j][:,1] + data['population'][j][:,2] + data['population'][j][:,4] + data['population'][j][:,7] + data['population'][j][:,11] + data['population'][j][:,16] + data['population'][j][:,22] + data['population'][j][:,29] + data['population'][j][:,37])
#         total_state_charge[j] /=second_step_sum
#         state_Br_charge[j]  /= second_step_sum
#         state_I_charge[j]   /= second_step_sum
#         total_state_spin[j] /=second_step_sum
#         state_Br_spin[j]  /= second_step_sum
#         state_I_spin[j]   /= second_step_sum
#         state_kenergy[j]    /= second_step_sum
#         state_bondlength[j] /= second_step_sum
# 
# total_charge = (total_state_charge[0] * data['population'][0][:,1]) + (total_state_charge[1] * total_pop[1]) + (total_state_charge[2] * total_pop[2]) + (total_state_charge[3] * total_pop[3]) + (total_state_charge[4] * total_pop[4])
# total_charge /= data['population'][1][:,21]
# total_Br_charge    = (state_Br_charge[0] * data['population'][0][:,1]) + (state_Br_charge[1] * total_pop[1]) + (state_Br_charge[2] * total_pop[2]) + (state_Br_charge[3] * total_pop[3]) + (state_Br_charge[4] * total_pop[4])
# total_Br_charge /= data['population'][1][:,21]
# total_I_charge     = (state_I_charge[0] * data['population'][0][:,1]) + (state_I_charge[1] * total_pop[1]) + (state_I_charge[2] * total_pop[2]) + (state_I_charge[3] * total_pop[3]) + (state_I_charge[4] * total_pop[4]) 
# total_I_charge /= data['population'][1][:,21]
# 
# total_spin = (total_state_spin[0] * data['population'][0][:,1]) + (total_state_spin[1] * total_pop[1]) + (total_state_spin[2] * total_pop[2]) + (total_state_spin[3] * total_pop[3]) + (total_state_spin[4] * total_pop[4])
# total_spin /= data['population'][1][:,21]
# total_Br_spin    = (state_Br_spin[0] * data['population'][0][:,1]) + (state_Br_spin[1] * total_pop[1]) + (state_Br_spin[2] * total_pop[2]) + (state_Br_spin[3] * total_pop[3]) + (state_Br_spin[4] * total_pop[4])
# total_Br_spin /= data['population'][1][:,21]
# total_I_spin     = (state_I_spin[0] * data['population'][0][:,1]) + (state_I_spin[1] * total_pop[1]) + (state_I_spin[2] * total_pop[2]) + (state_I_spin[3] * total_pop[3]) + (state_I_spin[4] * total_pop[4]) 
# total_I_spin /= data['population'][1][:,21]
# 
# total_kenergy      = (state_kenergy[0] * data['population'][0][:,1]) + (state_kenergy[1] * total_pop[1]) + (state_kenergy[2] * total_pop[2]) + (state_kenergy[3] * total_pop[3]) + (state_kenergy[4] * total_pop[4])
# total_kenergy /= data['population'][1][:,21]
# total_bondlength   = (state_bondlength[0] * data['population'][0][:,1]) + (state_bondlength[1] * total_pop[1]) + (state_bondlength[2] * total_pop[2]) + (state_bondlength[3] * total_pop[3]) + (state_bondlength[4] * total_pop[4])
# total_bondlength /= data['population'][1][:,21]
# 
# total_charge[:5000] = total_state_charge[0][:5000]  
# total_Br_charge[:5000] = state_Br_charge[0][:5000] 
# total_I_charge[:5000] = state_I_charge[0][:5000]
# total_spin[:5000] = total_state_spin[0][:5000]  
# total_Br_spin[:5000] = state_Br_spin[0][:5000] 
# total_I_spin[:5000] = state_I_spin[0][:5000]
# total_kenergy[:5000] = state_kenergy[0][:5000]
# total_bondlength[:5000] = state_bondlength[0][:5000]
# 
# total_charge[5000:10000] = (total_state_charge[0][5000:10000] *  data['population'][0][5000:10000,1]) + (total_state_charge[1][5000:10000] *  total_pop[1][5000:10000]) + (total_state_charge[2][5000:10000] * total_pop[2][5000:10000])
# total_charge[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# total_Br_charge[5000:10000] = (state_Br_charge[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_Br_charge[1][5000:10000] *  total_pop[1][5000:10000]) + (state_Br_charge[2][5000:10000] * total_pop[2][5000:10000])
# total_Br_charge[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# total_I_charge[5000:10000] = (state_I_charge[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_I_charge[1][5000:10000] *  total_pop[1][5000:10000]) + (state_I_charge[2][5000:10000] * total_pop[2][5000:10000])
# total_I_charge[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# 
# total_spin[5000:10000] = (total_state_spin[0][5000:10000] *  data['population'][0][5000:10000,1]) + (total_state_spin[1][5000:10000] *  total_pop[1][5000:10000]) + (total_state_spin[2][5000:10000] * total_pop[2][5000:10000])
# total_spin[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# total_Br_spin[5000:10000] = (state_Br_spin[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_Br_spin[1][5000:10000] *  total_pop[1][5000:10000]) + (state_Br_spin[2][5000:10000] * total_pop[2][5000:10000])
# total_Br_spin[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# total_I_spin[5000:10000] = (state_I_spin[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_I_spin[1][5000:10000] *  total_pop[1][5000:10000]) + (state_I_spin[2][5000:10000] * total_pop[2][5000:10000])
# total_I_spin[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# 
# total_kenergy[5000:10000] = (state_kenergy[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_kenergy[1][5000:10000] *  total_pop[1][5000:10000]) + (state_kenergy[2][5000:10000] * total_pop[2][5000:10000])
# total_kenergy[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# total_bondlength[5000:10000] = (state_bondlength[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_bondlength[1][5000:10000] *  total_pop[1][5000:10000]) + (state_bondlength[2][5000:10000] * total_pop[2][5000:10000])
# total_bondlength[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
# 
# print(total_charge[0])
# 
# plt.rcParams['agg.path.chunksize'] = 10000
# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2,sharex=False,sharey=False)
# fig.set_figwidth(8)
# fig.set_figheight(12)
# 
# ax1.plot(time*au2fs, data['population'][0][:,1],  linewidth=1.3, color='k', linestyle='-',  label=state_label['1p'])
# ax1.plot(time*au2fs, total_pop[1], linewidth=1.3, color=color_1, linestyle='-',             label=state_label['2p1s3pz'])
# ax1.plot(time*au2fs, total_pop[2], linewidth=1.3, color=color_2, linestyle='-',             label=state_label['2p2t3pz'])
# ax1.plot(time*au2fs, total_pop[3], linewidth=1.3, color=color_1, linestyle=':',             label=state_label['3p1d1pz'])
# ax1.plot(time*au2fs, total_pop[4], linewidth=1.3, color=color_2, linestyle=':',             label=state_label['3p2d1pz'])
# ax1.plot(time*au2fs, data['population'][1][:,11], linewidth=1.3, color='orange', linestyle='-',  label='Total')
# 
# ax2.plot(time*au2fs, total_state_charge[0], linewidth=1.0, color='k',      linestyle='-', label=state_label['1p'])
# ax2.plot(time*au2fs, total_state_charge[1], linewidth=1.0, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
# ax2.plot(time*au2fs, total_state_charge[2], linewidth=1.0, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
# ax2.plot(time*au2fs, total_state_charge[3], linewidth=1.0, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
# ax2.plot(time*au2fs, total_state_charge[4], linewidth=1.0, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
# ax2.plot(time*au2fs, total_charge,    linewidth=1.0, color='orange', linestyle='-', label='Total')
# 
# ax3.plot(time*au2fs, state_Br_charge[0], linewidth=1.0, color='k',      linestyle='-', label=state_label['1p'])
# ax3.plot(time*au2fs, state_Br_charge[1], linewidth=1.0, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
# ax3.plot(time*au2fs, state_Br_charge[2], linewidth=1.0, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
# ax3.plot(time*au2fs, state_Br_charge[3], linewidth=1.0, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
# ax3.plot(time*au2fs, state_Br_charge[4], linewidth=1.0, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
# ax3.plot(time*au2fs, total_Br_charge,    linewidth=1.0, color='orange', linestyle='-', label='Total')
# 
# ax4.plot(time*au2fs, state_I_charge[0], linewidth=1.0, color='k',      linestyle='-', label=state_label['1p'])
# ax4.plot(time*au2fs, state_I_charge[1], linewidth=1.0, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
# ax4.plot(time*au2fs, state_I_charge[2], linewidth=1.0, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
# ax4.plot(time*au2fs, state_I_charge[3], linewidth=1.0, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
# ax4.plot(time*au2fs, state_I_charge[4], linewidth=1.0, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
# ax4.plot(time*au2fs, total_I_charge,    linewidth=1.0, color='orange', linestyle='-', label='Total')
# 
# ax5.plot(time*au2fs, state_Br_spin[0], linewidth=1.0, color='k',      linestyle='-', label=state_label['1p'])
# ax5.plot(time*au2fs, state_Br_spin[1], linewidth=1.0, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
# ax5.plot(time*au2fs, state_Br_spin[2], linewidth=1.0, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
# ax5.plot(time*au2fs, state_Br_spin[3], linewidth=1.0, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
# ax5.plot(time*au2fs, state_Br_spin[4], linewidth=1.0, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
# ax5.plot(time*au2fs, total_Br_spin,    linewidth=1.0, color='orange', linestyle='-', label='Total')
# 
# ax6.plot(time*au2fs, state_I_spin[0], linewidth=1.0, color='k',      linestyle='-', label=state_label['1p'])
# ax6.plot(time*au2fs, state_I_spin[1], linewidth=1.0, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
# ax6.plot(time*au2fs, state_I_spin[2], linewidth=1.0, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
# ax6.plot(time*au2fs, state_I_spin[3], linewidth=1.0, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
# ax6.plot(time*au2fs, state_I_spin[4], linewidth=1.0, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
# ax6.plot(time*au2fs, total_I_spin,    linewidth=1.0, color='orange', linestyle='-', label='Total')
# 
# ax7.plot(time*au2fs, state_bondlength[0], linewidth=1.0, color='k',      linestyle='-', label=state_label['1p'])
# ax7.plot(time*au2fs, state_bondlength[1], linewidth=1.0, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
# ax7.plot(time*au2fs, state_bondlength[2], linewidth=1.0, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
# ax7.plot(time*au2fs, state_bondlength[3], linewidth=1.0, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
# ax7.plot(time*au2fs, state_bondlength[4], linewidth=1.0, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
# ax7.plot(time*au2fs, total_bondlength,    linewidth=1.0, color='orange', linestyle='-', label='Total')
# 
# ax8.plot(time*au2fs, state_kenergy[0]*au2ev, linewidth=1.0, color='k',      linestyle='-', label=state_label['1p'])
# ax8.plot(time*au2fs, state_kenergy[1]*au2ev, linewidth=1.0, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
# ax8.plot(time*au2fs, state_kenergy[2]*au2ev, linewidth=1.0, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
# ax8.plot(time*au2fs, state_kenergy[3]*au2ev, linewidth=1.0, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
# ax8.plot(time*au2fs, state_kenergy[4]*au2ev, linewidth=1.0, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
# ax8.plot(time*au2fs, total_kenergy*au2ev,    linewidth=1.0, color='orange', linestyle='-', label='Total')
# 
# ax1.set_xticklabels([])
# ax2.set_xticklabels([])
# ax3.set_xticklabels([])
# ax4.set_xticklabels([])
# ax5.set_xticklabels([])
# ax6.set_xticklabels([])
# ax1.set_ylim(10**-5, 1.2)
# #ax3.set_ylim(0.24, 2.3)
# #ax4.set_ylim(0.24, 2.3)
# #ax5.set_ylim(-2.1, 0.1)
# #ax6.set_ylim(-2.1, 0.1)
# ax1.set_yscale("log")
# ax1.set_ylabel("Population Fraction")
# ax2.set_ylabel("Total Weighted Charge Average")
# ax3.set_ylabel("Br Weighted Charge Average")
# ax4.set_ylabel("I Weighted Charge Average")
# ax5.set_ylabel("Br Weighted Spin Average")
# ax6.set_ylabel("I Weighted Spin Average")
# ax7.set_ylabel("Weighted Bondlength Average")
# ax8.set_ylabel("Weighted Kinetic Energy Average")
# ax7.set_xlabel("Time (fs)")
# ax8.set_xlabel("Time (fs)")
# ax1.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("pngs/population_all.png",dpi=300)

# fig, (ax1) = plt.subplots()
# fig.set_figwidth(6)
# fig.set_figheight(5)
# ax1.plot(time*au2fs, data['population'][0][:,1], linewidth=1.3, color='k', linestyle='-',  label=state_label['1p'])
# 
# ax1.set_xlim(0, 3)
# ax1.set_ylabel("Population fraction")
# ax1.set_xlabel("Time (fs)")
# ax1.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig("pngs/population_initial.png",dpi=300)
# 
# fig, (ax1) = plt.subplots()
# fig.set_figwidth(6.5)
# fig.set_figheight(5)
# ax1.plot(time*au2fs, data['population'][0][:,1], linewidth=1.3, color='k', linestyle='-',   label=state_label['1p'])
# ax1.plot(time*au2fs, total_pop[1], linewidth=1.3, color=color_1, linestyle='-',             label=state_label['2p1s3pz'])
# ax1.plot(time*au2fs, total_pop[2], linewidth=1.3, color=color_2, linestyle='-',             label=state_label['2p2t3pz'])
# ax1.plot(time*au2fs, total_pop[3], linewidth=1.3, color=color_1, linestyle=':',             label=state_label['3p1d1pz'])
# ax1.plot(time*au2fs, total_pop[4], linewidth=1.3, color=color_2, linestyle=':',             label=state_label['3p2d1pz'])
# ax1.plot(time*au2fs, data['population'][1][:,11], linewidth=1.3, color='orange', linestyle='-',  label='Total')
# 
# #ax1.set_xlim(-2, 120)
# ax1.set_ylim(10**-5, 1.2)
# ax1.set_ylabel("Population fraction")
# ax1.set_yscale("log")
# ax1.set_xlabel("Time (fs)")
# ax1.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("pngs/population_full.png",dpi=300)
# 
# print(total_pop[3].max())
# print(total_pop[4].max())
# 

