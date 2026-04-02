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
#second_step = [2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106, 121, 137, 154, 172]
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


for j in range(5):
    if j == 0:

        state_Br_charge[j]    = data['0_charge'][j][:,1] 
        state_I_charge[j]     = data['1_charge'][j][:,1] 
        total_state_charge[j] = state_Br_charge[j] + state_I_charge[j]
        state_Br_spin[j]    = data['0_spin'][j][:,1] 
        state_I_spin[j]     = data['1_spin'][j][:,1] 
        total_state_spin[j] = state_Br_spin[j] + state_I_spin[j]
        state_kenergy[j]      = data['kenergy'][j][:,1]     
        state_bondlength[j]   = data['bondlength'][j][:,1] 

    if j == 1 or j == 2:
        for k in range(1,21):
            if k == 1:
                total_state_charge[j] = (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
                state_Br_charge[j]    = data['0_charge'][j][:,k] * data['population'][j][:,k]
                state_I_charge[j]     = data['1_charge'][j][:,k] * data['population'][j][:,k]

                total_state_spin[j] = (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
                state_Br_spin[j]    = data['0_spin'][j][:,k] * data['population'][j][:,k]
                state_I_spin[j]     = data['1_spin'][j][:,k] * data['population'][j][:,k]

                state_kenergy[j]     = data['kenergy'][j][:,k] * data['population'][j][:,k] 
                state_bondlength[j]  = data['bondlength'][j][:,k] * data['population'][j][:,k]

            else:
                total_state_charge[j] += (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
                state_Br_charge[j]  += (data['0_charge'][j][:,k] * data['population'][j][:,k])
                state_I_charge[j]   += (data['1_charge'][j][:,k] * data['population'][j][:,k])

                total_state_spin[j] += (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
                state_Br_spin[j]  += (data['0_spin'][j][:,k] * data['population'][j][:,k])
                state_I_spin[j]   += (data['1_spin'][j][:,k] * data['population'][j][:,k])

                state_kenergy[j]    += (data['kenergy'][j][:,k] * data['population'][j][:,k]) 
                state_bondlength[j] += (data['bondlength'][j][:,k] * data['population'][j][:,k])


    if j == 3 or j == 4:
        for k in second_step:
            if k == 1:
                total_state_charge[j] = (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
                state_Br_charge[j]    = data['0_charge'][j][:,k] * data['population'][j][:,k]
                state_I_charge[j]     = data['1_charge'][j][:,k] * data['population'][j][:,k]

                total_state_spin[j] = (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
                state_Br_spin[j]    = data['0_spin'][j][:,k] * data['population'][j][:,k]
                state_I_spin[j]     = data['1_spin'][j][:,k] * data['population'][j][:,k]

                state_kenergy[j]      = data['kenergy'][j][:,k] * data['population'][j][:,k] 
                state_bondlength[j]   = data['bondlength'][j][:,k] * data['population'][j][:,k]

            else:
                total_state_charge[j] += (data['0_charge'][j][:,k] + data['1_charge'][j][:,k]) * data['population'][j][:,k] 
                state_Br_charge[j]    += (data['0_charge'][j][:,k] * data['population'][j][:,k])
                state_I_charge[j]     += (data['1_charge'][j][:,k] * data['population'][j][:,k])

                total_state_spin[j] += (data['0_spin'][j][:,k] + data['1_spin'][j][:,k]) * data['population'][j][:,k] 
                state_Br_spin[j]    += (data['0_spin'][j][:,k] * data['population'][j][:,k])
                state_I_spin[j]     += (data['1_spin'][j][:,k] * data['population'][j][:,k])

                state_kenergy[j]      += (data['kenergy'][j][:,k] * data['population'][j][:,k]) 
                state_bondlength[j]   += (data['bondlength'][j][:,k] * data['population'][j][:,k])


for j in range(1,5):
    
    if j == 1 or j== 2:
        total_state_charge[j] /=  total_pop[j]
        state_Br_charge[j] /= total_pop[j]
        state_I_charge[j]  /= total_pop[j]
        total_state_spin[j] /=  total_pop[j]
        state_Br_spin[j] /= total_pop[j]
        state_I_spin[j]  /= total_pop[j]
        state_kenergy[j]    /= total_pop[j]
        state_bondlength[j] /= total_pop[j]
    if j == 3 or j== 4:
        second_step_sum = np.zeros(100001)
        for i in second_step:
            second_step_sum += data['population'][j][:,i]
        total_state_charge[j] /=second_step_sum
        state_Br_charge[j]  /= second_step_sum
        state_I_charge[j]   /= second_step_sum
        total_state_spin[j] /=second_step_sum
        state_Br_spin[j]  /= second_step_sum
        state_I_spin[j]   /= second_step_sum
        state_kenergy[j]    /= second_step_sum
        state_bondlength[j] /= second_step_sum

total_charge = (total_state_charge[0] * data['population'][0][:,1]) + (total_state_charge[1] * total_pop[1]) + (total_state_charge[2] * total_pop[2]) + (total_state_charge[3] * total_pop[3]) + (total_state_charge[4] * total_pop[4])
total_charge /= data['population'][1][:,21]
total_Br_charge    = (state_Br_charge[0] * data['population'][0][:,1]) + (state_Br_charge[1] * total_pop[1]) + (state_Br_charge[2] * total_pop[2]) + (state_Br_charge[3] * total_pop[3]) + (state_Br_charge[4] * total_pop[4])
total_Br_charge /= data['population'][1][:,21]
total_I_charge     = (state_I_charge[0] * data['population'][0][:,1]) + (state_I_charge[1] * total_pop[1]) + (state_I_charge[2] * total_pop[2]) + (state_I_charge[3] * total_pop[3]) + (state_I_charge[4] * total_pop[4]) 
total_I_charge /= data['population'][1][:,21]

total_spin = (total_state_spin[0] * data['population'][0][:,1]) + (total_state_spin[1] * total_pop[1]) + (total_state_spin[2] * total_pop[2]) + (total_state_spin[3] * total_pop[3]) + (total_state_spin[4] * total_pop[4])
total_spin /= data['population'][1][:,21]
total_Br_spin    = (state_Br_spin[0] * data['population'][0][:,1]) + (state_Br_spin[1] * total_pop[1]) + (state_Br_spin[2] * total_pop[2]) + (state_Br_spin[3] * total_pop[3]) + (state_Br_spin[4] * total_pop[4])
total_Br_spin /= data['population'][1][:,21]
total_I_spin     = (state_I_spin[0] * data['population'][0][:,1]) + (state_I_spin[1] * total_pop[1]) + (state_I_spin[2] * total_pop[2]) + (state_I_spin[3] * total_pop[3]) + (state_I_spin[4] * total_pop[4]) 
total_I_spin /= data['population'][1][:,21]

total_kenergy      = (state_kenergy[0] * data['population'][0][:,1]) + (state_kenergy[1] * total_pop[1]) + (state_kenergy[2] * total_pop[2]) + (state_kenergy[3] * total_pop[3]) + (state_kenergy[4] * total_pop[4])
total_kenergy /= data['population'][1][:,21]
total_bondlength   = (state_bondlength[0] * data['population'][0][:,1]) + (state_bondlength[1] * total_pop[1]) + (state_bondlength[2] * total_pop[2]) + (state_bondlength[3] * total_pop[3]) + (state_bondlength[4] * total_pop[4])
total_bondlength /= data['population'][1][:,21]

total_charge[:5000] = total_state_charge[0][:5000]  
total_Br_charge[:5000] = state_Br_charge[0][:5000] 
total_I_charge[:5000] = state_I_charge[0][:5000]
total_spin[:5000] = total_state_spin[0][:5000]  
total_Br_spin[:5000] = state_Br_spin[0][:5000] 
total_I_spin[:5000] = state_I_spin[0][:5000]
total_kenergy[:5000] = state_kenergy[0][:5000]
total_bondlength[:5000] = state_bondlength[0][:5000]

total_charge[5000:10000] = (total_state_charge[0][5000:10000] *  data['population'][0][5000:10000,1]) + (total_state_charge[1][5000:10000] *  total_pop[1][5000:10000]) + (total_state_charge[2][5000:10000] * total_pop[2][5000:10000])
total_charge[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
total_Br_charge[5000:10000] = (state_Br_charge[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_Br_charge[1][5000:10000] *  total_pop[1][5000:10000]) + (state_Br_charge[2][5000:10000] * total_pop[2][5000:10000])
total_Br_charge[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
total_I_charge[5000:10000] = (state_I_charge[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_I_charge[1][5000:10000] *  total_pop[1][5000:10000]) + (state_I_charge[2][5000:10000] * total_pop[2][5000:10000])
total_I_charge[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])

total_spin[5000:10000] = (total_state_spin[0][5000:10000] *  data['population'][0][5000:10000,1]) + (total_state_spin[1][5000:10000] *  total_pop[1][5000:10000]) + (total_state_spin[2][5000:10000] * total_pop[2][5000:10000])
total_spin[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
total_Br_spin[5000:10000] = (state_Br_spin[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_Br_spin[1][5000:10000] *  total_pop[1][5000:10000]) + (state_Br_spin[2][5000:10000] * total_pop[2][5000:10000])
total_Br_spin[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
total_I_spin[5000:10000] = (state_I_spin[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_I_spin[1][5000:10000] *  total_pop[1][5000:10000]) + (state_I_spin[2][5000:10000] * total_pop[2][5000:10000])
total_I_spin[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])

total_kenergy[5000:10000] = (state_kenergy[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_kenergy[1][5000:10000] *  total_pop[1][5000:10000]) + (state_kenergy[2][5000:10000] * total_pop[2][5000:10000])
total_kenergy[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])
total_bondlength[5000:10000] = (state_bondlength[0][5000:10000] *  data['population'][0][5000:10000,1]) + (state_bondlength[1][5000:10000] *  total_pop[1][5000:10000]) + (state_bondlength[2][5000:10000] * total_pop[2][5000:10000])
total_bondlength[5000:10000] /= (data['population'][0][5000:10000,1] + total_pop[1][5000:10000] + total_pop[2][5000:10000])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex=False,sharey=False)
fig.set_figwidth(8)
fig.set_figheight(6.5)

ax1.plot(time*au2fs, state_bondlength[0],  linewidth=1.8, color='k', linestyle='-',  label=state_label['1p'])
ax1.plot(time*au2fs, state_bondlength[1], linewidth=1.8, color=color_1, linestyle='-',             label=state_label['2p1s3pz'])
ax1.plot(time*au2fs, state_bondlength[2], linewidth=1.8, color=color_2, linestyle='-',             label=state_label['2p2t3pz'])
ax1.plot(time*au2fs, state_bondlength[3], linewidth=1.8, color=color_1, linestyle=':',             label=state_label['3p1d1pz'])
ax1.plot(time*au2fs, state_bondlength[4], linewidth=1.8, color=color_2, linestyle=':',             label=state_label['3p2d1pz'])
ax1.plot(time*au2fs, total_bondlength,    linewidth=1.8, color='orange', linestyle='--',  label='Total')

ax2.plot(time*au2fs, state_kenergy[0]*au2ev, linewidth=1.8, color='k',      linestyle='-', label=state_label['1p'])
ax2.plot(time*au2fs, state_kenergy[1]*au2ev, linewidth=1.8, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
ax2.plot(time*au2fs, state_kenergy[2]*au2ev, linewidth=1.8, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
ax2.plot(time*au2fs, state_kenergy[3]*au2ev, linewidth=1.8, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
ax2.plot(time*au2fs, state_kenergy[4]*au2ev, linewidth=1.8, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
ax2.plot(time*au2fs, total_kenergy*au2ev,    linewidth=1.8, color='orange', linestyle='--', label='Total')

ax3.plot(time*au2fs, state_Br_charge[0], linewidth=1.8, color='k',      linestyle='-', label=state_label['1p'])
ax3.plot(time*au2fs, state_Br_charge[1], linewidth=1.8, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
ax3.plot(time*au2fs, state_Br_charge[2], linewidth=1.8, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
ax3.plot(time*au2fs, state_Br_charge[3], linewidth=1.8, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
ax3.plot(time*au2fs, state_Br_charge[4], linewidth=1.8, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
ax3.plot(time*au2fs, total_Br_charge,    linewidth=1.8, color='orange', linestyle='--', label='Total')

ax4.plot(time*au2fs, state_I_charge[0], linewidth=1.8, color='k',      linestyle='-', label=state_label['1p'])
ax4.plot(time*au2fs, state_I_charge[1], linewidth=1.8, color=color_1,  linestyle='-', label=state_label['2p1s3pz'])
ax4.plot(time*au2fs, state_I_charge[2], linewidth=1.8, color=color_2,  linestyle='-', label=state_label['2p2t3pz'])
ax4.plot(time*au2fs, state_I_charge[3], linewidth=1.8, color=color_1,  linestyle=':', label=state_label['3p1d1pz'])
ax4.plot(time*au2fs, state_I_charge[4], linewidth=1.8, color=color_2,  linestyle=':', label=state_label['3p2d1pz'])
ax4.plot(time*au2fs, total_I_charge,    linewidth=1.8, color='orange', linestyle='--', label='Total')


trans = mtransforms.ScaledTranslation(-20/72, 7/60, fig.dpi_scale_trans)
ax1.text(0.0, 1.0, 'a)', transform=ax1.transAxes + trans)
ax2.text(0.0, 1.0, 'b)', transform=ax2.transAxes + trans)
ax3.text(0.0, 1.0, 'c)', transform=ax3.transAxes + trans)


print(total_Br_charge[len(total_Br_charge)-1])
print(total_I_charge[len(total_I_charge)-1])
print(total_Br_charge[len(total_Br_charge)-1]+total_I_charge[len(total_I_charge)-1])


ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_ylim(0.20, 2.25)
ax4.set_ylim(0.20, 2.25)
ax1.set_ylabel("Bond length ($\AA$)")
ax2.set_ylabel("Weight. Avg. Kinetic Energy")
ax3.set_ylabel("Weight. Avg. Br Charge")
ax4.set_ylabel("Weight. Avg. I Charge")
ax3.set_xlabel("Time (fs)")
ax4.set_xlabel("Time (fs)")
ax1.legend(loc="upper left")
plt.tight_layout()
plt.savefig("pngs/weight_averaged.png",dpi=300)




