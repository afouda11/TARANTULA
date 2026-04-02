import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

state_label = {'2p1s3pz':'IBr[3$d_{Br}^{1}$,$\sigma_{pz}^{1}$]$^{+2}$',
		 '2p2t3pz':'IBr[3$d_{Br}^{1}$,$\sigma_{s}^{1}$]$^{+2}$', 
         '3p1d1pz':'IBr[$\sigma_{s}^{1}$,$\sigma_{pz}^{1}$,$\pi_{px}^{1}$]$^{+3}$',          
         '3p2d1pz':'IBr[$\sigma_{s}^{2}$,$\sigma_{pz}^{1}$]$^{+3}$',
         '1p': 'IBr[3$p_{Br}^{1}$]$^{+1}$'} 

data = {'population':{}, 'bondlength':{}, 'energy':{}, 'kenergy':{}, '0_charge':{}, '1_charge':{}, '0_spin':{}, '1_spin':{}}

t_spawn    = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000,
		      55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

spawn_start = [10000] 
for i in range(2,20):
	spawn_start = np.append(spawn_start, np.repeat(t_spawn[i], i))

for i in data:
    for j in range(5):
        data[i][j] = np.loadtxt('outputs/'+i+'_'+str(j)+'.txt')

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

for i in data:
    for j in range(5):
        data[i][j] = np.loadtxt('outputs/'+i+'_'+str(j)+'.txt')

au2fs = 0.02418884254
au2ev = 27.2114

time = data['population'][0][:,0]

color_1 = '#7fbf7b'
color_2 = '#af8dc3'



second_step = [1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106, 121, 137, 154, 172]
fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True)
fig.set_figwidth(8)
fig.set_figheight(5)

ax1.plot(time*au2fs, data['population'][0][:,1], linewidth=1.3, color='k', linestyle='-',  label=state_label['1p'])

for i in range(1,21):
	ax1.plot(time[t_spawn[i-1]-1:]*au2fs, data['population'][1][t_spawn[i-1]-1:,i], linewidth=1.8, color=color_1, linestyle='-')
	ax1.plot(time[t_spawn[i-1]-1:]*au2fs, data['population'][2][t_spawn[i-1]-1:,i], linewidth=1.8, color=color_2, linestyle='-')


for i in range(1,191):
    ax1.plot(time[spawn_start[i-1]-1:]*au2fs, data['population'][3][spawn_start[i-1]-1:,i], linewidth=1.8, color=color_1, linestyle=':')
    ax1.plot(time[spawn_start[i-1]-1:]*au2fs, data['population'][4][spawn_start[i-1]-1:,i], linewidth=1.8, color=color_2, linestyle=':')

ax2.plot(time*au2fs, data['population'][0][:,1],  linewidth=1.8, color='k', linestyle='-',  label=state_label['1p'])
ax2.plot(time*au2fs, total_pop[1], linewidth=1.8, color=color_1, linestyle='-',             label=state_label['2p1s3pz'])
ax2.plot(time*au2fs, total_pop[2], linewidth=1.8, color=color_2, linestyle='-',             label=state_label['2p2t3pz'])
ax2.plot(time*au2fs, total_pop[3], linewidth=1.8, color=color_1, linestyle=':',             label=state_label['3p1d1pz'])
ax2.plot(time*au2fs, total_pop[4], linewidth=1.8, color=color_2, linestyle=':',             label=state_label['3p2d1pz'])
ax2.plot(time*au2fs, data['population'][1][:,21], linewidth=1.8, color='orange', linestyle='--',  label='Total')


trans = mtransforms.ScaledTranslation(-20/72, 7/60, fig.dpi_scale_trans)
ax1.text(0.0, 1.0, 'a)', transform=ax1.transAxes + trans)
ax2.text(0.0, 1.0, 'b)', transform=ax2.transAxes + trans)

ax1.set_ylim(10**-5, 1.2)
ax1.set_yscale("log")
ax1.set_ylabel("Population Fraction")
ax1.set_xlabel("Time (fs)")
ax2.set_xlabel("Time (fs)")
ax2.legend(loc="lower center")
plt.tight_layout()
plt.savefig("pngs/population_all.png",dpi=300)




